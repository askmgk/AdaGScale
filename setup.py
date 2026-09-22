import warnings
from distutils import sysconfig as distutils_sysconfig
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Conda's Python CFLAGS include a C-only warning option. This extension is C++.
_cflags = distutils_sysconfig.get_config_var("CFLAGS")
if _cflags:
    distutils_sysconfig.get_config_vars()["CFLAGS"] = " ".join(
        flag for flag in _cflags.split() if flag != "-Wstrict-prototypes"
    )

# PyTorch 1.12 has no compiler bounds entry for CUDA 11.6. The build script
# selects the CUDA host compiler explicitly, so this warning is uninformative.
warnings.filterwarnings(
    "ignore",
    message=r"There are no .* version bounds defined for CUDA version 11\.6",
    category=UserWarning,
    module=r"torch\.utils\.cpp_extension",
)

def _is_cuda() -> bool:
    return torch.version.cuda is not None


ext_modules = []

# Compiler flags.
# Accelerate compilation speed and prevent compiler from performing negative optimizations!
CXX_FLAGS = ["-g", "-O1", "-std=c++17"]
# The bundled GLM headers and inherited FlashGS kernels trigger these two
# benign CUDA frontend diagnostics. Leave all other diagnostics visible.
NVCC_FLAGS = ["-O1", '-Xptxas="-O1"', "--diag-suppress=20012,177"]


def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

if _is_cuda():
    ext_modules.append(
        CUDAExtension(
            name="adagscale_cuda",
            sources=glob("csrc/cuda_rasterizer/*.cu") + glob("csrc/pybind.cpp"),
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
    )

setup(name="adagscale_cuda",
      version="0.1.0",
      ext_modules=ext_modules,
      cmdclass={"build_ext": BuildExtension}
      )
