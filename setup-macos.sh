#!/bin/bash
# KernelForgeML - macOS Setup Script
# This script sets up the development environment for KernelForgeML on macOS

set -e

echo "=== KernelForgeML macOS Setup ==="
echo ""

# Check for Rust
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "✅ Rust installed"
else
    echo "✅ Rust found: $(rustc --version)"
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install from https://brew.sh"
    exit 1
else
    echo "✅ Homebrew found"
fi

# Install LLVM 18
if [ ! -d "/opt/homebrew/opt/llvm@18" ]; then
    echo "📦 Installing LLVM 18..."
    brew install llvm@18
    echo "✅ LLVM 18 installed"
else
    echo "✅ LLVM 18 already installed"
fi

# Install zstd (required for LLVM linking)
if ! brew list zstd &> /dev/null; then
    echo "📦 Installing zstd..."
    brew install zstd
    echo "✅ zstd installed"
else
    echo "✅ zstd already installed"
fi

# Set environment variables
echo ""
echo "=== Setting Environment Variables ==="
ENV_FILE="$HOME/.kernelforge_env"

cat > "$ENV_FILE" << 'EOF'
# KernelForgeML Environment Variables
export PATH="$HOME/.cargo/bin:/opt/homebrew/opt/llvm@18/bin:$PATH"
export LLVM_SYS_180_PREFIX="/opt/homebrew/opt/llvm@18"
export MLIR_SYS_180_PREFIX="/opt/homebrew/opt/llvm@18"
export LLVM_CONFIG_PATH="/opt/homebrew/opt/llvm@18/bin/llvm-config"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
EOF

echo "✅ Environment file created at: $ENV_FILE"
echo ""
echo "To activate the environment, run:"
echo "  source $ENV_FILE"
echo ""
echo "Or add the following to your ~/.zshrc or ~/.bash_profile:"
echo "  source $ENV_FILE"
echo ""

# Load environment for this session
source "$ENV_FILE"

# Verify LLVM
echo "=== Verifying LLVM ==="
if command -v llvm-config &> /dev/null; then
    echo "✅ LLVM version: $(llvm-config --version)"
else
    echo "❌ llvm-config not found in PATH"
    exit 1
fi

# Create autotune cache directory
mkdir -p "$HOME/.kernelforge"
echo "✅ Created cache directory: $HOME/.kernelforge"

# Build the project
echo ""
echo "=== Building KernelForgeML ==="
cargo build --all

echo ""
echo "=== Running Tests ==="
cargo test --workspace

echo ""
echo "✅ Build and tests complete!"
echo ""
echo "=== Quick Start ==="
echo ""
echo "Available commands:"
echo ""
echo "1. Diagnose GPU:"
echo "   cargo run -p kernelforge_benchmarks -- diagnose-gpu"
echo ""
echo "2. Run LLM-guided optimization:"
echo "   cargo run -p kernelforge_benchmarks -- optimize-with-llm --iterations 3"
echo ""
echo "3. Run mutation testing:"
echo "   cargo run -p kernelforge_benchmarks -- mutate-and-test"
echo ""
echo "4. Emit MLIR for transformer ops:"
echo "   cargo run -p kernelforge_benchmarks -- emit-mlir"
echo ""
echo "5. CPU benchmark:"
echo "   cargo run -p kernelforge_benchmarks -- benchmark-matmul --m 512 --n 512 --k 1024"
echo ""
echo "6. GPU benchmark (Metal):"
echo "   cargo run -p kernelforge_benchmarks -- --target gpu benchmark-matmul --m 256 --n 256 --k 512"
echo ""
