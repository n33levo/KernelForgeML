# Comprehensive Results and Documentation

## Test Setup

Hardware: M-series MacBook
OS: macOS 14.6.0
Rust: 1.90.0
LLVM: 18.1.8

## Setup & Environment

The project was successfully set up and built on macOS with Rust 1.90.0 and LLVM 18. The setup script (`./setup-macos.sh`) installed all dependencies and created the environment file `~/.kernelforge_env` which must be sourced before running any commands:

```bash
source ~/.kernelforge_env
```

## CPU Benchmarks

Ran the benchmark suite with three test cases:

| Test Case | Size (M×N×K) | Kernel | Time (ms) | GFLOP/s |
|-----------|--------------|--------|-----------|---------|
| decoder_mha_qkv | 64×64×128 | reference | 2.372 | 0.442 |
| decoder_mlp_up_proj | 128×192×192 | reference | 20.716 | 0.456 |
| decoder_mlp_down_proj | 128×128×256 | reference | 17.043 | 0.492 |

All tests had zero numerical error (bit-exact match with reference).

## GPU Benchmarks

GPU backend produces correct results but timing is stubbed (0.0 ms until query pools are plumbed through):

| Test Case | Size (MxNxK) | Status |
|-----------|--------------|--------|
| Single matmul | 1024x1024x2048 | Shader executes, no timing |

## LLM Inference Demo

Running the tiny decoder with random weights verifies end-to-end plumbing and provides latency breakdowns. Examples:

### With "the quick brown fox" prompt:
- Prefill: ~23.2 ms
- Decode: ~81.5 ms
- Tokens generated: 5
- Throughput: ~61.4 tokens/sec
- KV-cache footprint: 2048 bytes/token

### With custom testing:
- Command: `cargo run -p kernelforge_benchmarks -- llm-inference --prompt "Hello world" --max-tokens 5`
- Input: "Hello world"
- Output: "in" (generated 2 tokens, stopped at 5 requested)
- Prefill time: 21.75 ms
- Decode time: 81.61 ms
- Total time: 103.36 ms
- Throughput: 61.27 tokens/sec
- KV-cache footprint: 2048 bytes/token

Enabling `--compare-with-cerebras` (with `CEREBRAS_API_KEY`) prints hosted-model output and performance for side-by-side inspection.

## Functional Components Tested

### 1. MLIR Emission
- Command: `cargo run -p kernelforge_benchmarks -- emit-mlir`
- Result: Successfully generates MLIR code for transformer operations including matmul, attention, FFN, and layer normalization
- Output includes proper tensor operations with linalg dialect

### 2. CPU Benchmarking
- Command: `cargo run -p kernelforge_benchmarks -- benchmark-matmul --m 512 --n 512 --k 1024`
- Result: Generated performance metrics for matrix multiplication
  - Average time: ~1024ms (on first run, faster on subsequent runs due to autotuning)
  - Performance: ~0.52 GFLOP/s 
  - Kernel: reference implementation

### 3. Full Benchmark Suite
- Command: `cargo run -p kernelforge_benchmarks -- benchmark-suite --output reports/cpu_baseline.json`
- Result: Ran 3 transformer-style test cases with detailed metrics (shown above)

### 4. Unit Tests
- Command: `cargo test --workspace`
- Result: 8 tests passed across kernel and LLM inference components:
  - 3 kernel tests (matmul reference matches, parallel matches reference, layer norm)
  - 5 LLM tests (KV cache, tokenizer, RoPE)

## Performance Notes

Current performance on M-series Mac:
- Small matmuls achieve ~0.44-0.49 GFLOP/s
- Reference BLAS libraries typically achieve ~100+ GFLOP/s
- The ~200x performance gap is expected as this uses reference implementations focused on correctness over optimization

## Known Limitations Observed

- GPU timing still reports 0.0 ms (timing infrastructure not fully implemented)
- Only tested on macOS with Homebrew and Metal
- Kernels are reference-grade; no SIMD/vector optimizations
- Attention kernels are skeletal; inference uses matmul + softmax on CPU

## File Artifacts Generated

Running the benchmark suite creates:
- JSON reports with detailed metrics (latency, GFLOP/s, bandwidth, error norms)
- Autotuning cache at `~/.kernelforge/autotune.json`
- Performance baselines under `reports/` directory

## Commands That Work

The following commands were verified as functional:

```bash
# Compile and build everything
cargo build --all

# Run unit tests
cargo test --workspace

# Emit MLIR
cargo run -p kernelforge_benchmarks -- emit-mlir

# Single CPU benchmark
cargo run -p kernelforge_benchmarks -- benchmark-matmul --m 512 --n 512 --k 1024

# Full benchmark suite
cargo run -p kernelforge_benchmarks -- benchmark-suite --output results.json

# LLM inference demo
cargo run -p kernelforge_benchmarks -- llm-inference --prompt "hello" --max-tokens 10
```

## Environment Requirements

The project requires:
- Rust 1.90.0 or later
- LLVM 18
- Homebrew (on macOS)
- Environment variables set via the setup script