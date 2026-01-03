//! GPU runtime abstractions with Metal/wgpu backend.
//!
//! This module provides GPU execution for compute kernels using wgpu,
//! which maps to Metal on macOS and Vulkan/DX12 on other platforms.

use crate::planner::{GpuMatmulPlan, GpuPlanner};
use anyhow::{anyhow, bail, ensure, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use kernelforge_kernels::config::{ActivationKind, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use ndarray::Array2;
use pollster::block_on;
use std::num::NonZeroU64;
use std::sync::mpsc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use tracing::info;

/// Result of a GPU kernel execution with timing information.
#[derive(Debug, Clone)]
pub struct GpuExecutionResult {
    pub output: Array2<f32>,
    pub gpu_time_ms: f64,
    pub cpu_dispatch_time_ms: f64,
    pub total_time_ms: f64,
}

impl GpuExecutionResult {
    /// Calculate effective GFLOP/s based on matmul operation.
    pub fn gflops(&self, m: usize, n: usize, k: usize) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / 1e9;
        if self.gpu_time_ms > 0.0 {
            gflops / (self.gpu_time_ms / 1000.0)
        } else {
            0.0
        }
    }
    
    /// Calculate memory bandwidth in GB/s.
    pub fn bandwidth_gbps(&self, m: usize, n: usize, k: usize) -> f64 {
        // Bytes read: A (m*k) + B (k*n), Bytes written: C (m*n)
        let bytes = ((m * k + k * n + m * n) * std::mem::size_of::<f32>()) as f64;
        let gb = bytes / 1e9;
        if self.gpu_time_ms > 0.0 {
            gb / (self.gpu_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

pub struct GpuExecutor {
    planner: GpuPlanner,
    context: GpuContext,
}

impl GpuExecutor {
    pub fn new(planner: GpuPlanner) -> Result<Self> {
        let context = GpuContext::new()?;
        Ok(Self { planner, context })
    }

    /// Execute matmul on GPU and return result with timing.
    pub fn execute_matmul(
        &self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
        tiling: Option<(u32, u32, u32)>,
    ) -> Result<Array2<f32>> {
        let result = self.execute_matmul_timed(problem, inputs, tiling)?;
        Ok(result.output)
    }

    /// Execute layer norm on GPU. Expects gamma/beta to be shape (1, features).
    pub fn execute_layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array2<f32>,
        beta: &Array2<f32>,
        epsilon: f32,
        workgroup_y: u32,
    ) -> Result<Array2<f32>> {
        self.context
            .run_layer_norm(input, gamma, beta, epsilon, workgroup_y)
    }
    
    /// Execute matmul on GPU with detailed timing information.
    pub fn execute_matmul_timed(
        &self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
        tiling: Option<(u32, u32, u32)>,
    ) -> Result<GpuExecutionResult> {
        if inputs.bias.is_some() {
            bail!("GPU matmul does not yet support bias");
        }
        if !matches!(inputs.activation, ActivationKind::None) {
            bail!("GPU matmul does not yet support fused activation");
        }

        let plan = self.planner.plan_matmul(problem, tiling)?;
        self.context.run_matmul_timed(plan, inputs)
    }
    
    /// Get information about the GPU device.
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.context.device_info
    }
}

impl Default for GpuExecutor {
    fn default() -> Self {
        Self::new(GpuPlanner).expect("gpu executor initialization")
    }
}

/// Information about the GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: String,
    pub supports_timestamps: bool,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    device_info: GpuDeviceInfo,
    timestamp_query_set: Option<wgpu::QuerySet>,
    timestamp_period: f32,
}

impl GpuContext {
    fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no suitable GPU adapter found"))?;
        
        let adapter_info = adapter.get_info();
        let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        // Request timestamp query feature if available
        let required_features = if supports_timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("KernelForge GPU Device"),
                required_features,
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;
        
        // Create timestamp query set if supported
        let timestamp_query_set = if supports_timestamps {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp_queries"),
                ty: wgpu::QueryType::Timestamp,
                count: 2, // Start and end timestamps
            }))
        } else {
            None
        };
        
        let timestamp_period = if supports_timestamps {
            queue.get_timestamp_period()
        } else {
            0.0
        };
        
        let device_info = GpuDeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            supports_timestamps,
        };

        Ok(Self { 
            device, 
            queue, 
            device_info,
            timestamp_query_set,
            timestamp_period,
        })
    }

    fn run_matmul_timed(&self, plan: GpuMatmulPlan, inputs: &MatmulInputs<'_>) -> Result<GpuExecutionResult> {
        let problem = plan.problem;
        ensure!(inputs.lhs.ncols() == problem.k, "lhs columns must equal K");
        ensure!(inputs.rhs.nrows() == problem.k, "rhs rows must equal K");
        ensure!(inputs.lhs.nrows() == problem.m, "lhs rows must equal M");
        ensure!(inputs.rhs.ncols() == problem.n, "rhs columns must equal N");

        let total_start = Instant::now();

        let lhs = inputs.lhs.to_owned();
        let rhs = inputs.rhs.to_owned();

        let lhs_data = lhs.into_raw_vec();
        let rhs_data = rhs.into_raw_vec();

        let lhs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lhs"),
                contents: cast_slice(&lhs_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rhs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rhs"),
                contents: cast_slice(&rhs_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_elements = problem.m * problem.n;
        let output_size = (output_elements * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Timestamp resolve buffer (if supported)
        let timestamp_buffer = if self.timestamp_query_set.is_some() {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_buffer"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        
        let timestamp_staging = if self.timestamp_query_set.is_some() {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_staging"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let params = ShaderParams {
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
            _pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("matmul_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(
                                    std::mem::size_of::<ShaderParams>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let wg_m = plan.workgroup_m.max(1).min(16);
        let wg_n = plan.workgroup_n.max(1).min(16);
        let tile_k = plan.tile_k.max(1);
        info!(
            workgroup_m = wg_m,
            workgroup_n = wg_n,
            tile_k,
            "gpu matmul plan applied"
        );

        let shader_src = matmul_shader_source(wg_m, wg_n, tile_k);
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            });

        let workgroup_x = (problem.m as u32).div_ceil(wg_m);
        let workgroup_y = (problem.n as u32).div_ceil(wg_n);

        let dispatch_start = Instant::now();
        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        // Configure timestamp writes if supported
        let timestamp_writes = self.timestamp_query_set.as_ref().map(|qs| {
            wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x.max(1), workgroup_y.max(1), 1);
        }
        
        // Resolve timestamps if available
        if let (Some(qs), Some(ts_buf)) = (&self.timestamp_query_set, &timestamp_buffer) {
            encoder.resolve_query_set(qs, 0..2, ts_buf, 0);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        
        // Copy timestamp buffer to staging
        if let (Some(ts_buf), Some(ts_staging)) = (&timestamp_buffer, &timestamp_staging) {
            encoder.copy_buffer_to_buffer(
                ts_buf, 
                0, 
                ts_staging, 
                0, 
                2 * std::mem::size_of::<u64>() as u64
            );
        }
        
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        
        let cpu_dispatch_time_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        // Read output data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| anyhow!("failed to receive GPU map signal"))??;
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        // Read GPU timestamps if available
        let gpu_time_ms = if let Some(ts_staging) = &timestamp_staging {
            let ts_slice = ts_staging.slice(..);
            let (ts_sender, ts_receiver) = mpsc::channel();
            ts_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = ts_sender.send(res);
            });
            self.device.poll(wgpu::Maintain::Wait);
            
            // Check if mapping succeeded
            match ts_receiver.recv().map_err(|_| anyhow!("timestamp recv failed"))? {
                Ok(_) => {
                    let ts_data = ts_slice.get_mapped_range();
                    let timestamps: &[u64] = cast_slice(&ts_data);
                    let start = timestamps[0];
                    let end = timestamps[1];
                    drop(ts_data);
                    ts_staging.unmap();
                    
                    // Convert to milliseconds using timestamp period
                    let delta_ns = (end - start) as f64 * self.timestamp_period as f64;
                    delta_ns / 1_000_000.0
                }
                Err(_) => {
                    // Fallback to CPU timing if timestamp mapping failed
                    cpu_dispatch_time_ms
                }
            }
        } else {
            // No timestamp support, use CPU timing as approximation
            cpu_dispatch_time_ms
        };
        
        let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let output = Array2::from_shape_vec((problem.m, problem.n), result)
            .map_err(|err| anyhow!("failed to shape GPU output: {err}"))?;
            
        Ok(GpuExecutionResult {
            output,
            gpu_time_ms,
            cpu_dispatch_time_ms,
            total_time_ms,
        })
    }

    fn run_layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array2<f32>,
        beta: &Array2<f32>,
        epsilon: f32,
        workgroup_y: u32,
    ) -> Result<Array2<f32>> {
        let rows = input.nrows() as u32;
        let cols = input.ncols() as u32;

        ensure!(
            gamma.shape() == &[1, input.ncols()] && beta.shape() == &[1, input.ncols()],
            "gamma/beta must be shape (1, features)"
        );

        let input_vec = input.as_standard_layout().to_owned().into_raw_vec();
        let gamma_vec = gamma.as_standard_layout().to_owned().into_raw_vec();
        let beta_vec = beta.as_standard_layout().to_owned().into_raw_vec();

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ln_input"),
                contents: cast_slice(&input_vec),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gamma_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ln_gamma"),
                contents: cast_slice(&gamma_vec),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let beta_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ln_beta"),
                contents: cast_slice(&beta_vec),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size =
            (input_vec.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ln_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ln_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = LayerNormParams {
            m: rows,
            n: cols,
            epsilon,
            _pad: 0.0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ln_params"),
                contents: cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ln_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(
                                    std::mem::size_of::<LayerNormParams>() as u64
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layernorm_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let wg_y = workgroup_y.max(1).min(64);
        let shader_src = layer_norm_shader_source(wg_y);
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layernorm_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("layernorm_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("layernorm_encoder"),
            });

        let workgroups_x = rows;
        let workgroups_y = (cols + wg_y - 1) / wg_y;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layernorm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| anyhow!("failed to receive GPU map signal"))??;
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let output = Array2::from_shape_vec((rows as usize, cols as usize), result)
            .map_err(|err| anyhow!("failed to shape GPU layernorm output: {err}"))?;

        Ok(output)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormParams {
    m: u32,
    n: u32,
    epsilon: f32,
    _pad: f32,
}

fn matmul_shader_source(workgroup_m: u32, workgroup_n: u32, tile_k: u32) -> String {
    format!(
        r#"
struct Params {{
  size_m: u32,
  size_n: u32,
  size_k: u32,
  _padding: u32,
}}

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn lhs_index(m: u32, k: u32) -> u32 {{
  return m * params.size_k + k;
}}

fn rhs_index(k: u32, n: u32) -> u32 {{
  return k * params.size_n + n;
}}

fn out_index(m: u32, n: u32) -> u32 {{
  return m * params.size_n + n;
}}

@compute @workgroup_size({wg_m}, {wg_n}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let m = gid.x;
  let n = gid.y;
  if (m >= params.size_m || n >= params.size_n) {{
    return;
  }}

  var acc: f32 = 0.0;
  var k_outer: u32 = 0u;
  loop {{
    if (k_outer >= params.size_k) {{
        break;
    }}
    let k_limit = min(params.size_k, k_outer + {tile_k}u);
    var k: u32 = k_outer;
    loop {{
        if (k >= k_limit) {{
            break;
        }}
        acc = acc + lhs[lhs_index(m, k)] * rhs[rhs_index(k, n)];
        k = k + 1u;
    }}
    k_outer = k_outer + {tile_k}u;
  }}

  output[out_index(m, n)] = acc;
}}
"#,
        wg_m = workgroup_m,
        wg_n = workgroup_n,
        tile_k = tile_k
    )
}

fn layer_norm_shader_source(workgroup_y: u32) -> String {
    format!(
        r#"
struct Params {{
  size_m: u32,
  size_n: u32,
  epsilon: f32,
  _pad: f32,
}}

@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read> gamma: array<f32>;
@group(0) @binding(2)
var<storage, read> beta: array<f32>;
@group(0) @binding(3)
var<storage, read_write> output: array<f32>;
@group(0) @binding(4)
var<uniform> params: Params;

fn idx(m: u32, n: u32) -> u32 {{
  return m * params.size_n + n;
}}

@compute @workgroup_size(1, {wg_y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let row = gid.x;
  let col = gid.y;
  if (row >= params.size_m || col >= params.size_n) {{
    return;
  }}

  var mean: f32 = 0.0;
  for (var i: u32 = 0u; i < params.size_n; i = i + 1u) {{
    mean = mean + input[idx(row, i)];
  }}
  mean = mean / f32(params.size_n);

  var var_acc: f32 = 0.0;
  for (var i: u32 = 0u; i < params.size_n; i = i + 1u) {{
    let d = input[idx(row, i)] - mean;
    var_acc = var_acc + d * d;
  }}
  let inv_std = inverseSqrt(var_acc / f32(params.size_n) + params.epsilon);

  let x = input[idx(row, col)];
  let g = gamma[col];
  let b = beta[col];
  output[idx(row, col)] = (x - mean) * inv_std * g + b;
}}
"#,
        wg_y = workgroup_y
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_gpu_execution_result_gflops() {
        let output = Array2::zeros((64, 64));
        let result = GpuExecutionResult {
            output,
            gpu_time_ms: 1.0, // 1ms
            cpu_dispatch_time_ms: 1.5,
            total_time_ms: 2.0,
        };
        
        // 2 * 64 * 64 * 64 = 524288 FLOPs = 0.000524288 GFLOP
        // At 1ms, that's 0.524288 GFLOP/s
        let gflops = result.gflops(64, 64, 64);
        assert!((gflops - 0.524288).abs() < 0.01);
    }
}
