# KernelForgeML

**Assured LLM-Guided ML Compiler Optimizer**

A Rust ML compiler framework where **LLM proposes optimizations → verifier checks correctness → system learns best plans**. Built with MLIR for IR, wgpu/Metal for GPU execution on macOS.

## Overview

KernelForgeML demonstrates a novel approach to ML compiler optimization:

1. **LLM Proposes**: An LLM (or heuristic) suggests optimization knobs and pass ordering
2. **Verifier Checks**: Every suggestion is verified against a CPU reference with numeric tolerance
3. **System Learns**: Accepted plans are cached and reused for similar workloads

This is the "Assured" part: **no optimization is applied unless it passes verification**.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Optimizer      │────▶│  Verifier        │────▶│  Best Plans     │
│  (LLM/Heuristic)│     │  (CPU vs GPU)    │     │  Cache          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Key Features

### 🎯 LLM-Guided Optimization
- Heuristic optimizer (no network, deterministic baseline)
- LLM optimizer (OpenAI-compatible API) with automatic fallback
- Structured JSON output with pass order + tuning knobs

### ✅ Assured Verification
- CPU reference vs GPU execution comparison
- Configurable numeric tolerance (absolute + relative error)
- Shape/type invariant checking
- Machine-readable audit reports

### 🧪 Mutation Testing
- Automatic generation of "broken" optimization plans
- Verifies the test suite catches real bugs
- Regression corpus for continuous testing

### 🖥️ Mac GPU First-Class Support
- wgpu → Metal backend for Apple Silicon (M1/M2/M3/M4)
- GPU timestamp queries for accurate kernel timing
- `diagnose-gpu` command for quick health checks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      High-Level IR (MLIR)                       │
│    MatMul, Attention, MLP, LayerNorm with tensor metadata       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM-Guided Optimizer                          │
│  • Proposes pass order    • Suggests tuning knobs               │
│  • Falls back to heuristic if no API key                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Assurance Harness                           │
│  • CPU reference computation    • Numeric tolerance checking    │
│  • Audit report generation      • Best plans cache              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌───────────────┬─────────────────────────────┬───────────────────┐
│ CPU Backend   │        GPU Backend          │  Mutation Tests   │
│ (Rayon)       │        (wgpu/Metal)         │  (Regression)     │
└───────────────┴─────────────────────────────┴───────────────────┘
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `optimizer/` | **NEW** LLM proposer, verifier, mutation testing |
| `ir/` | MLIR IR builder, custom dialect for transformer ops |
| `kernels/` | Kernel implementations (matmul, attention, layernorm) |
| `autotune/` | Grid-search autotuner with persistent JSON cache |
| `backend-cpu/` | CPU planner and Rayon-based executor |
| `backend-gpu/` | GPU executor with wgpu/Metal, timestamp queries |
| `compiler/` | Pass pipeline, session management, CLI |
| `benchmarks/` | CLI entrypoint for all commands |

## Setup (macOS)

```bash
# Install LLVM 18
brew install llvm@18

# Clone and setup
git clone https://github.com/your-username/KernelForgeML
cd KernelForgeML
source scripts/env.sh

# Build
cargo build --workspace

# Run tests
cargo test --workspace
```

## Quick Start

### 1. Diagnose GPU

```bash
cargo run -p kernelforge_benchmarks -- diagnose-gpu
```

Output:
```
=== KernelForgeML GPU Diagnostics ===

GPU Device: Apple M4
Backend: Metal
Timestamp Queries: Supported

--- Running GPU Smoke Test ---
GPU smoke test: 64x64x128 matmul, max_abs_error=0.00e0, gpu_time=0.000ms, passed=true

✓ GPU smoke test PASSED
```

### 2. Run LLM-Guided Optimization

```bash
# Without LLM (uses heuristic)
cargo run -p kernelforge_benchmarks -- optimize-with-llm --iterations 3

# With LLM (set API key first)
export KERNELFORGE_LLM_API_KEY="sk-..."
cargo run -p kernelforge_benchmarks -- optimize-with-llm --iterations 5
```

### 3. Verify a Specific Plan

```bash
cargo run -p kernelforge_benchmarks -- verify-plan --plan my_plan.json
```

### 4. Run Mutation Testing

```bash
cargo run -p kernelforge_benchmarks -- mutate-and-test
```

Output:
```
=== KernelForgeML Mutation Testing ===

Generated 6 mutants
✓ Mutant killed: tile_k = 0
✓ Mutant killed: vector_width = 3 (not power of 2)
✓ Mutant killed: layernorm_epsilon = 0.0
...

Total mutants: 6, Killed: 6, Escaped: 0
```

## All Commands

| Command | Description |
|---------|-------------|
| `diagnose-gpu` | Print GPU adapter info and run smoke test |
| `optimize-with-llm` | Run LLM-guided optimization loop |
| `verify-plan` | Verify a specific plan from JSON |
| `mutate-and-test` | Run mutation testing on the optimizer |
| `emit-mlir` | Emit MLIR for a transformer block |
| `show-passes` | List available optimization passes |
| `optimize-ir` | Show before/after IR optimization |
| `benchmark-matmul` | Benchmark a matmul kernel |
| `benchmark-suite` | Run the evaluation suite |

## Optimization Knobs

The `OptimizationKnobs` struct contains tunable parameters:

```rust
pub struct OptimizationKnobs {
    pub tile_m: usize,           // Tile size for M dimension
    pub tile_n: usize,           // Tile size for N dimension  
    pub tile_k: usize,           // Tile size for K dimension
    pub vector_width: usize,     // SIMD vector width
    pub enable_fuse_matmul_activation: bool,
    pub enable_fuse_mlp: bool,
    pub enable_fold_constants: bool,
    pub layernorm_epsilon: f32,
}
```

## Available Passes

| Pass | Description |
|------|-------------|
| `fold-constants` | Fold compile-time constant expressions |
| `fuse-matmul-activation` | Fuse activation into matmul ops |
| `fuse-mlp-block` | Fuse MLP block pattern |
| `tile-matmul` | Apply cache-aware tiling |
| `vectorize-layernorm` | Vectorize layer normalization |
| `eliminate-dead-ops` | Remove unused operations |

## View Optimization Passes

```bash
cargo run -p kernelforge_benchmarks -- show-passes
```

## LLM Configuration

To use the LLM optimizer instead of the heuristic:

```bash
export KERNELFORGE_LLM_API_KEY="sk-..."           # Required
export KERNELFORGE_LLM_ENDPOINT="https://..."     # Optional (default: OpenAI)
export KERNELFORGE_LLM_MODEL="gpt-4o-mini"        # Optional
```

## IR Optimization Passes

Real compiler optimizations adapted for ML:

- **FuseMatmulActivation**: Fuses activation into matmul (e.g., Linear+GELU → FusedLinearGELU)
- **TileMatmul**: Annotates for cache-efficient tiling
- **FoldConstants**: Propagates compile-time constants (scale factors, epsilon)
- **VectorizeLayerNorm**: Marks for SIMD execution
- **EliminateDeadOps**: Removes unused operations

## GPU Backend Details

GPU backend uses wgpu with timestamp queries for accurate kernel timing:

```rust
let result = executor.execute_matmul_timed(problem, &inputs)?;
println!("GPU time: {:.2}ms, GFLOP/s: {:.2}", 
    result.gpu_time_ms, 
    result.gflops(m, n, k));
```

### Why Mac GPU?

- **wgpu** provides a clean abstraction over Metal on macOS
- Apple Silicon GPUs are ubiquitous for ML development
- Timestamp queries work on Metal, enabling accurate kernel timing
- The `diagnose-gpu` command verifies everything works

## Project Goals

This project demonstrates:

1. How ML compilers represent and transform computation graphs
2. How LLMs can propose optimizations that are verified for correctness
3. How mutation testing validates the assurance harness
4. How backends lower high-level ops to hardware-specific code

It's designed to be presentable and understandable, with a clean flow from proposal → verification → caching.

## Related Work

- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework (used via Melior)
- [TVM](https://tvm.apache.org/) - End-to-end ML compiler stack
- [Triton](https://github.com/openai/triton) - OpenAI's GPU programming language
- [LLM Compiler](https://arxiv.org/abs/2407.02524) - Meta's code optimization LLM

## License

MIT
