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
- Full transformer block GPU execution:
  - Q/K/V projections on GPU with plan-driven tiling
  - QK^T scores matmul on GPU
  - Softmax on CPU (hybrid by design for numeric stability)
  - Context matmul (scores × V) on GPU
  - LayerNorm on GPU with configurable workgroup size
- CPU reference vs GPU execution comparison
- Configurable numeric tolerance (absolute + relative error)
- Structured error reporting with stage/shape/knobs on failures
- Machine-readable audit reports

### 🧪 Mutation Testing
- 9 mutants tested: 6 validation + 3 plausible numeric mutants
- **Plausible mutants** that pass validation but fail numeric checks:
  - Uniform attention weights (ignores Q·K content)
  - Missing attention scale (1/√d_k)
  - Biased layernorm variance (no gamma/beta)
- Per-stage error tracking (Q/K/V/attention/final)
- Automatic regression corpus generation
- Kill reason classification (Validation, NumericMismatch, RegressionCase)

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

=== Part 1: Plan Validation Mutants ===
Generated 6 plan mutants
✓ Mutant killed (validation): tile_k = 0
✓ Mutant killed (validation): vector_width = 3 (not power of 2)
✓ Mutant killed (validation): layernorm_epsilon = 0.0
✓ Mutant killed (validation): Unknown pass in pass_order
✓ Mutant killed (validation): target = 'tpu' (unsupported)
✓ Mutant killed (validation): tile_m = 1024 (too large)

=== Part 2: Plausible Mutants (Numeric Equivalence) ===
Generated 3 plausible mutants
✓ Plausible mutant killed (NumericMismatch): uniform attention weights (ignores content)
  errors: q=0.00e0, k=0.00e0, v=0.00e0, attn=1.49e-3, final=6.07e-2
✓ Plausible mutant killed (NumericMismatch): missing attention scale (1/sqrt(d_k))
  errors: q=0.00e0, k=0.00e0, v=0.00e0, attn=6.84e-3, final=2.75e-1
✓ Plausible mutant killed (NumericMismatch): biased layernorm variance (no gamma/beta)
  errors: q=0.00e0, k=0.00e0, v=0.00e0, attn=0.00e0, final=1.13e-1

Total killed: 9, Escaped: 0
Regression corpus saved to: reports/regression_corpus.json
```

## End-to-End Compiler Workflow

The `compile-transformer` command provides a complete, single-command workflow for compiling a transformer block:

```bash
# Compile a transformer block for GPU
cargo run -p kernelforge_benchmarks -- compile-transformer \
    --seq 128 --d-model 256 --target gpu

# Or for CPU
cargo run -p kernelforge_benchmarks -- compile-transformer \
    --seq 64 --d-model 128 --target cpu
```

### What Happens

1. **Build Workload**: Creates a `TransformerMicroblock` with:
   - Input matrix: `[seq_len × d_model]`
   - Q/K/V projection weights: `[d_model × d_model]`
   - LayerNorm parameters: `[d_model]`

2. **Detect Hardware**: Queries GPU capabilities (Apple M1/M2/M3/M4)

3. **Propose Optimization Plan**: LLM or heuristic suggests:
   - Tile sizes (`tile_m`, `tile_n`, `tile_k`)
   - Vector width for SIMD
   - Pass ordering (fuse-matmul-activation, tile-matmul, etc.)

4. **Validate Plan**: Checks plan constraints:
   - Tile sizes are valid (non-zero, power of 2 for vector width)
   - Target is supported (`cpu` or `gpu`)
   - All passes are known

5. **Compute Reference**: CPU computes the correct output:
   - Q = Input × W_Q, K = Input × W_K, V = Input × W_V
   - Attention = softmax(Q × K^T / √d_k) × V
   - Output = LayerNorm(Attention)

6. **GPU Verify**: Full transformer block runs on GPU:
   - Q/K/V projections: GPU matmul with plan tiling
   - Scores (QK^T): GPU matmul
   - Softmax: CPU (hybrid for numeric stability)
   - Context (scores × V): GPU matmul
   - LayerNorm: GPU with plan workgroup width
   - End-to-end tolerance: 2e-1 (accounts for hybrid softmax)

7. **Emit Results**: Saves plan and audit report to JSON

### Example Output

```
=== KernelForgeML Transformer Compiler ===

Step 1: Building transformer microblock workload...
  Workload: seq_len=128, d_model=256, d_k=64

Step 2: Detecting hardware...
  Target: gpu
  GPU: Apple M4

Step 3: Proposing optimization plan...
  Using heuristic optimizer
  Proposed plan: tile_m=64, tile_n=64, tile_k=32

Step 4: Validating plan...
  ✓ Plan is valid

Step 5: Computing CPU reference...
  ✓ Reference computed

Step 6: Running GPU verification (Q projection matmul)...
  GPU Q projection max error: 0.00e0
  GPU time: 0.000ms
  ✓ GPU output matches CPU reference

Step 6b: Running full transformer on GPU (softmax on CPU)...
  End-to-end GPU max error vs CPU: 2.40e-2
  ✓ GPU end-to-end output matches CPU reference (<= 2.00e-1)

Step 7: Emitting results...
  Plan saved to: transformer_plan.json
  Audit saved to: transformer_audit.json

✓ Compilation successful!
```

### Additional Options

| Flag | Description |
|------|-------------|
| `--seq` | Sequence length (default: 64) |
| `--d-model` | Model dimension (default: 128) |
| `--target` | `gpu` or `cpu` (default: gpu) |
| `--optimize-with` | `heuristic` or `llm` (default: heuristic) |
| `--seed` | Random seed for reproducibility |
| `--plan-output` | Path for output plan JSON |
| `--audit-output` | Path for audit report JSON |

## All Commands

| Command | Description |
|---------|-------------|
| `compile-transformer` | **End-to-end compiler workflow for a transformer block** |
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

GPU backend uses wgpu with Metal for Apple Silicon:

- **Plan-driven execution**: Optimization knobs directly control GPU behavior
  - `tile_m/n/k` → GPU workgroup sizes (capped at 16×16 for Metal limits)
  - `vector_width` → SIMD width for CPU kernels
  - Fusion flags → enable/disable matmul+activation and MLP fusion
- **Hybrid transformer execution**:
  - Heavy matmuls (Q/K/V projections, scores, context) run on GPU
  - Softmax runs on CPU for numeric stability (documented by design)
  - LayerNorm runs on GPU with configurable workgroup width
- **Structured error reporting**:
  - GPU failures include stage, shape, error detail, and applied plan knobs
  - Example: "GPU execution failed at stage 'matmul' (shape 128x256x512): workgroup size exceeded device limits. Plan knobs: tile_m=64, tile_n=64, tile_k=32, workgroup_m=16, workgroup_n=16"

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
