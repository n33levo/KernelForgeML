//! GPU runtime abstractions.

use crate::planner::{GpuMatmulPlan, GpuPlanner};
use anyhow::{anyhow, bail, ensure, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use kernelforge_kernels::config::{ActivationKind, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use ndarray::Array2;
use pollster::block_on;
use std::num::NonZeroU64;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const MATMUL_SHADER: &str = include_str!("shaders/matmul.wgsl");

pub struct GpuExecutor {
    planner: GpuPlanner,
    context: GpuContext,
}

impl GpuExecutor {
    pub fn new(planner: GpuPlanner) -> Result<Self> {
        let context = GpuContext::new()?;
        Ok(Self { planner, context })
    }

    pub fn execute_matmul(
        &self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<Array2<f32>> {
        if inputs.bias.is_some() {
            bail!("GPU matmul does not yet support bias");
        }
        if !matches!(inputs.activation, ActivationKind::None) {
            bail!("GPU matmul does not yet support fused activation");
        }

        let plan = self.planner.plan_matmul(problem)?;
        self.context.run_matmul(plan, inputs)
    }
}

impl Default for GpuExecutor {
    fn default() -> Self {
        Self::new(GpuPlanner).expect("gpu executor initialization")
    }
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
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

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("KernelForge GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        Ok(Self { device, queue })
    }

    fn run_matmul(&self, plan: GpuMatmulPlan, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>> {
        let problem = plan.problem;
        ensure!(inputs.lhs.ncols() == problem.k, "lhs columns must equal K");
        ensure!(inputs.rhs.nrows() == problem.k, "rhs rows must equal K");
        ensure!(inputs.lhs.nrows() == problem.m, "lhs rows must equal M");
        ensure!(inputs.rhs.ncols() == problem.n, "rhs columns must equal N");

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

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            });

        let workgroup_x = (problem.m as u32).div_ceil(16);
        let workgroup_y = (problem.n as u32).div_ceil(16);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x.max(1), workgroup_y.max(1), 1);
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

        Array2::from_shape_vec((problem.m, problem.n), result)
            .map_err(|err| anyhow!("failed to shape GPU output: {err}"))
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
