# Architecture

Quick overview of how KernelForgeML is structured and why.

## Design Goals

1. Fast iteration: change IR, recompile, benchmark in <30 seconds
2. Extensible: easy to add new kernels, backends, or IR ops
3. Safe: Rust's type system catches bugs at compile time
4. Transparent: easy to see what's happening (no magic)

## Component Overview

### IR Layer (`crates/ir`)

Uses MLIR through the `melior` bindings. Defines a high-level transformer dialect with ops like:
- `matmul` - matrix multiplication
- `attention` - scaled dot-product attention
- `mlp` - feedforward layer
- `layer_norm` - layer normalization

The IR builder is just a wrapper around MLIR's builder API. Nothing fancy, just makes it easier to construct common transformer patterns.

**Why MLIR?** Didn't want to write my own IR from scratch. MLIR gives you parsing, printing, passes, and type checking for free.

### Kernel Library (`crates/kernels`)

Three matmul implementations:
1. **Reference** - just calls `ndarray::dot()`. Slow but correct.
2. **Blocked** - tiled matmul with configurable block sizes. Better cache locality.
3. **Parallel** - uses rayon to parallelize across rows. Good for wide matrices.

All kernels implement the same `MatmulKernel` trait so they're pluggable.

**Why three variants?** Wanted to have something for the autotuner to choose between. In practice, the reference kernel wins most of the time for small sizes because the overhead of tiling/parallelism dominates.

### Autotuner (`crates/autotune`)

Dead simple grid search:
1. Run each kernel variant
2. Measure time
3. Pick the fastest
4. Cache result to JSON

The cache is persistent across runs so you don't re-benchmark the same problem size.

**Why not something smarter?** Grid search is easy to implement and understand. For a research project, that's more important than optimal performance. Could swap in Bayesian optimization or whatever later.

### CPU Backend (`crates/backend-cpu`)

Just a thin wrapper around the kernel library and autotuner. The "planner" picks which kernel to use, the "runtime" actually executes it.

Uses rayon for parallelism because it's built into Rust and works well.

### GPU Backend (`crates/backend-gpu`)

Uses `wgpu` to dispatch compute shaders with proper timestamp query support for accurate GPU timing.

**Why wgpu?** It's cross-platform (Metal, Vulkan, DX12) and integrates well with Rust. Could have used raw Metal APIs but wgpu gives better portability.

The runtime includes:
- Timestamp queries for accurate kernel timing (where supported)
- Device info reporting (backend, name, timestamp support)
- Structured execution results with timing breakdown

### Evaluation + Compiler (`crates/compiler`)

`crates/compiler` is the glue: it assembles the pass pipeline, manages backend executors, and exposes the CLI. A new `eval` module provides a reusable evaluation suite—right now it benchmarks three transformer-style matmuls, checks numerical fidelity, and emits JSON reports (latency, GFLOP/s, bandwidth, error norms). Reports can be diffed against a previous baseline to spot regressions.

### Benchmark Harness (`crates/benchmarks`)

Thin CLI wrapper built with `clap`. Commands:

- `diagnose-gpu` – print GPU adapter info and run smoke test.
- `optimize-with-llm` – run LLM-guided optimization loop.
- `verify-plan` – verify a specific optimization plan from JSON.
- `mutate-and-test` – run mutation testing on the optimizer.
- `emit-mlir` – print the module's MLIR.
- `show-passes` – list available optimization passes in the pipeline.
- `optimize-ir` – run IR optimization and show before/after diff.
- `benchmark-matmul` – run a single matmul through compile + execute.
- `benchmark-suite` – run the evaluation suite and write JSON.

### Optimizer (`crates/optimizer`)

The LLM-guided optimization engine:

- **Optimizer trait** with `HeuristicOptimizer` (no network) and `LlmOptimizer` (OpenAI-compatible API).
- **Verifier** compares GPU output against CPU reference with numeric tolerance.
- **Mutator** generates broken plans to test the verifier catches real bugs.
- **BestPlansCache** persists verified optimization plans.

## Data Flow

```
User defines transformer block
         ↓
IR builder emits MLIR
         ↓
LLM/Heuristic proposes optimization plan
         ↓
Verifier checks plan against CPU reference
         ↓
Backend selection (CPU or GPU)
         ↓
CPU: Autotuner picks kernel → Kernel executes
GPU: wgpu shader dispatch with timing
         ↓
Results + audit report
```

## Why This Architecture?

**Separation of concerns:** IR, kernels, and backends are independent. Can swap backends without touching IR.

**Testability:** Each component has its own tests. The matmul kernels have unit tests that verify correctness.

**Extensibility:** Adding a new kernel is just implementing a trait. Adding a new backend is implementing planner + runtime.

## What's Missing

**Fusion passes:** The IR supports fusion (e.g., matmul + bias + activation in one op) but the lowering passes aren't implemented. Right now each op is standalone.

**Scheduling:** No control over tiling, unrolling, vectorization, etc. The blocked kernel has hardcoded tile sizes. Could expose this as a tuning knob.

**Advanced autotuning:** Grid search is slow. Could use cost models, machine learning, or adaptive sampling.

**GPU optimization:** The wgpu shader is naive. Could use shared memory, tensor cores, etc.

**Multi-platform:** Only tested on macOS. Linux should work but haven't tried. Windows probably needs different setup.

## Performance Notes

Current numbers on M-series Mac:
- Small matmuls (512x512x1024): ~0.5 GFLOP/s
- Reference BLAS libraries: ~100+ GFLOP/s

So there's a 200x gap. But that's expected for reference implementations. The point of this project is the infrastructure, not peak performance.

Ways to close the gap:
1. Use SIMD intrinsics (std::simd is stabilizing)
2. Better tiling strategies
3. Parallelize K loop, not just M
4. Call out to optimized BLAS for large sizes

## References

- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) - Good intro to MLIR
- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm) - Classic matmul optimization guide
- [Halide tutorials](https://halide-lang.org/tutorials/) - Scheduling separations
- [TVM docs](https://tvm.apache.org/docs/) - Production ML compiler architecture
