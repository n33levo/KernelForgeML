#!/bin/bash
# KernelForgeML environment setup for LLVM 18 / MLIR
export LLVM_HOME="/opt/homebrew/opt/llvm@18"
export PATH="$LLVM_HOME/bin:$PATH"
export LLVM_SYS_180_PREFIX="$LLVM_HOME"
export MLIR_SYS_180_PREFIX="$LLVM_HOME"
export LLVM_CONFIG_PATH="$LLVM_HOME/bin/llvm-config"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
