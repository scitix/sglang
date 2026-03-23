#!/bin/bash
# Build the standalone UVM activation allocator shared library.
#
# Source:  sgl-kernel/csrc/memory/uvm_allocator.cu
# Output:  python/sglang/srt/utils/libuvm_allocator.so
#
# Usage:
#   cd sgl-kernel
#   bash build_uvm.sh            # auto-detect GPU arch
#   bash build_uvm.sh sm_89      # Ada Lovelace (RTX 4090)
#   bash build_uvm.sh sm_90      # Hopper (H100/H200)
#   CUDA_HOME=/usr/local/cuda bash build_uvm.sh

set -e

SGLANG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(dirname "${BASH_SOURCE[0]}")/csrc/memory/uvm_allocator.cu"
OUT="$SGLANG_ROOT/python/sglang/srt/utils/libuvm_allocator.so"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ── auto-detect GPU SM architecture ──────────────────────────────────────────
if [ -n "$1" ]; then
    ARCH="$1"
else
    ARCH=""
    if command -v nvidia-smi &>/dev/null; then
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | head -1 | tr -d '.')
        [ -n "$CC" ] && ARCH="sm_${CC}"
    fi
    # fallback: Hopper (H100/H200)
    [ -z "$ARCH" ] && ARCH="sm_90"
fi
COMPUTE="${ARCH/sm_/compute_}"

echo "[sgl-kernel] Building UVM allocator"
echo "  Source : $SRC"
echo "  Output : $OUT"
echo "  CUDA   : $CUDA_HOME"
echo "  Arch   : $ARCH"

"$CUDA_HOME/bin/nvcc" \
    -shared \
    -O2 \
    -std=c++17 \
    -gencode "arch=${COMPUTE},code=${ARCH}" \
    --compiler-options '-fPIC' \
    "$SRC" \
    -o "$OUT"

echo "[sgl-kernel] Done: $OUT"
