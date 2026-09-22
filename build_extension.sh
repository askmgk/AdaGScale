#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the AdaGScale Conda environment before building." >&2
  exit 1
fi
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA nvcc was not found in the active Conda environment." >&2
  exit 1
fi

# CUDA 11.6 needs a compatible host compiler for .cu files. The Conda GCC
# package supplies it; use the system C++ compiler for the Python binding and
# final link so the host Python headers and libraries resolve correctly.
CUDA_CC=""
for candidate in "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
                 "$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-gcc"; do
  if [[ -x "$candidate" ]]; then
    CUDA_CC="$candidate"
    break
  fi
done
if [[ -n "$CUDA_CC" ]]; then
  SYSTEM_CXX="$(command -v g++ || true)"
  if [[ -z "$SYSTEM_CXX" ]]; then
    echo "A system g++ compiler is required to link the extension." >&2
    exit 1
  fi
  export CC="$CUDA_CC"
  export CXX="$SYSTEM_CXX"
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
PY

python setup.py build_ext --inplace
python - <<'PY'
import torch
import adagscale_cuda
print("AdaGScale extension import: OK")
PY
