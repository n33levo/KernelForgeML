# KernelForgeML

A Rust + MLIR compiler framework for ML kernels, focused on transformer operations. Built this to learn how ML compilers work under the hood and experiment with kernel optimization strategies.

## What is this?

Most ML frameworks are Python with some C++ kernels. I wanted to see what it would look like to build a compiler stack in Rust that could:
- Define transformer ops (matmul, attention, etc.) in a high-level IR
- Use MLIR for the IR layer
- Auto-tune kernel variants
- Support both CPU and GPU backends

Think of it as a toy version of what XLA or TVM do, but way simpler and Rust-first.

## Structure

```
crates/
  ir/             - MLIR IR builder and transformer dialect
  kernels/        - Kernel implementations (matmul, attention helpers, layernorm)
  autotune/       - Grid-search autotuner with persistent cache
  backend-cpu/    - CPU executor + planner (rayon powered)
  backend-gpu/    - Experimental GPU executor (wgpu/Metal)
  compiler/       - Pass pipeline, session management, CLI wiring, eval harness
  bench​marks/    - CLI entrypoint (`cargo run …`)
  llm-inference/  - Minimal transformer decoder with KV-cache, RoPE, safetensors
```

## Setup (macOS)

Need Rust and LLVM 18:

```bash
# Install LLVM
brew install llvm@18

# Set env vars (add to your .zshrc)
export PATH="/opt/homebrew/opt/llvm@18/bin:$PATH"
export LLVM_SYS_180_PREFIX="/opt/homebrew/opt/llvm@18"
export MLIR_SYS_180_PREFIX="/opt/homebrew/opt/llvm@18"
export LLVM_CONFIG_PATH="/opt/homebrew/opt/llvm@18/bin/llvm-config"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Or just run the setup script
./setup-macos.sh && source ~/.kernelforge_env
```

## Usage

```bash
# Build everything
cargo build --all

# Run tests
cargo test

# Emit MLIR for a transformer block
cargo run -p kernelforge_benchmarks -- emit-mlir

# Benchmark a matmul (CPU)
cargo run -p kernelforge_benchmarks -- benchmark-matmul \
  --m 512 --n 512 --k 1024

# Benchmark on GPU (Metal)
cargo run -p kernelforge_benchmarks -- --target gpu \
  benchmark-matmul --m 512 --n 512 --k 1024

# Run the curated evaluation suite (writes JSON report)
cargo run -p kernelforge_benchmarks -- benchmark-suite \
  --output reports/cpu_baseline.json

# Run an end-to-end decoder (random weights by default)
cargo run -p kernelforge_benchmarks -- llm-inference \
  --prompt "the quick brown fox" \
  --max-tokens 20

# Compare local inference vs. Cerebras hosted model (requires API key)
CEREBRAS_API_KEY=<key> cargo run -p kernelforge_benchmarks -- llm-inference \
  --prompt "hello" --max-tokens 32 --compare-with-cerebras
```

## What works

- MLIR emission for transformer blocks (`emit-mlir`)
- CPU matmul kernels (reference, blocked, parallel) with autotuned selection
- Evaluation suite that benchmarks representative transformer GEMMs and emits JSON reports (`reports/cpu_baseline.json`)
- LLM decoder inference with KV-cache, RoPE, safetensors loading, and optional Cerebras API comparison
- GPU/Metal shader dispatch (produces correct output; timing stubbed)
- Continuous integration via GitHub Actions (fmt, clippy, tests)

## What doesn't work / TODO

- GPU timing still reports `0.0 ms` (needs timestamp queries)
- Attention kernel lowering is skeletal; inference uses matmul + softmax implemented on CPU
- Fusion passes not implemented yet (IR has attributes, no lowering)
- Only tested on macOS so far
- Random-weight LLM demo is for plumbing verification; needs checkpoint export to be useful

## Performance

Latest evaluation suite on an Apple Silicon MacBook (macOS 14.6) using the tiny transformer cases:

| Case | Shape (M×N×K) | Kernel | Latency (ms) | GFLOP/s |
|------|----------------|--------|--------------|---------|
| decoder_mha_qkv | 64×64×128 | reference | 2.372 | 0.442 |
| decoder_mlp_up_proj | 128×192×192 | reference | 20.716 | 0.456 |
| decoder_mlp_down_proj | 128×128×256 | reference | 17.043 | 0.492 |

Numbers live in `reports/cpu_baseline.json`. Autotuning caches choices under `~/.kernelforge/autotune.json`.

## Why Rust + MLIR?

- Rust: Memory safety, good FFI, fast compilation
- MLIR: Modular IR infrastructure, don't have to write my own from scratch
- Learning: Wanted to understand how compiler stacks work

Most ML compilers use Python + C++. Rust gives you safety without GC overhead, which is nice for a compiler.

## Related Work

- [MLIR](https://mlir.llvm.org/) - The IR framework this uses
- [TVM](https://tvm.apache.org/) - Production ML compiler
- [XLA](https://www.tensorflow.org/xla) - TensorFlow's compiler
- [Halide](https://halide-lang.org/) - Image processing DSL with scheduling

This is way simpler than any of those but borrows ideas from all of them.

## Known Issues

- Only works on macOS right now (needs Homebrew, Metal)
- GPU timing infrastructure not done
- Performance isn't amazing (reference kernels)
- No batching support yet

See [docs/architecture.md](docs/architecture.md) for more details on design decisions.
See [docs/run_summary.md](docs/run_summary.md) for comprehensive test results and running documentation.

## License

MIT
