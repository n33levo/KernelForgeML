# Project Status

Quick summary of what's done and what works.

## Completed

- MLIR IR builder for transformer ops (matmul, attention, MLP, layer norm)
- Three CPU matmul kernels (reference, blocked, parallel) wired through an autotuner with persistent JSON cache
- CPU planner/executor (rayon) and GPU executor stub (wgpu/Metal)
- Benchmark CLI covering MLIR emission, single-op benchmarking, and evaluation suite report generation
- Evaluation harness with regression-friendly JSON output + diff support
- LLM decoder crate with RoPE, KV-cache, safetensors loading, Cerebras API comparison helper
- End-to-end tests/unit tests for kernels, KV-cache, RoPE, tokenizer, etc. (`cargo test --workspace`)

## Tested On

macOS 14.6, Apple Silicon, Rust 1.90.0, LLVM 18.1.8

## Works

```bash
# Build
cargo build --all

# Tests
cargo test --workspace  # kernel + LLM inference unit tests passing

# MLIR emission
cargo run -p kernelforge_benchmarks -- emit-mlir

# CPU benchmark
cargo run -p kernelforge_benchmarks -- benchmark-matmul --m 512 --n 512 --k 1024
# Result: prints autotuned kernel profile (GFLOP/s, latency)

# GPU benchmark
cargo run -p kernelforge_benchmarks -- --target gpu benchmark-matmul --m 512 --n 512 --k 1024
# Result: shader runs (timing = 0.0 placeholder)

# Full suite
cargo run -p kernelforge_benchmarks -- benchmark-suite --output reports/cpu_baseline.json
# Result: JSON report with GFLOP/s, bandwidth, error norms

# LLM inference demo
cargo run -p kernelforge_benchmarks -- llm-inference \
  --prompt "the quick brown fox" --max-tokens 20
# Result: prints prefill/decode times, tokens/sec, KV-cache bytes/token

# Optional hosted comparison (requires CEREBRAS_API_KEY)
cargo run -p kernelforge_benchmarks -- llm-inference \
  --prompt "hello world" --max-tokens 32 --compare-with-cerebras
```

## Known Issues

- GPU timing still stubbed (no query pools yet)
- Only tested on macOS (Linux/Windows unverified)
- Kernels are reference-grade; SIMD/vector optimizations TODO
- Attention kernels in compiler lowering are minimal; inference relies on matmul loops
- No IR fusion passes wired in (attributes present but not lowered)

## Next Steps

1. GPU timing + more efficient shaders (shared memory, tiling)
2. SIMD-aware CPU kernels and broader autotune search space
3. Flesh out attention lowering/fusion
4. Add proper checkpoint exporter to feed real model weights into `llm-inference`
5. Expand CI matrix beyond macOS (Linux runners)

## Note on "Eval"

The "eval" in this project is the benchmark evaluation suite that measures matmul performance, not LLM model evaluation. It's testing the compiler/kernel framework itself, not running inference on language models.
