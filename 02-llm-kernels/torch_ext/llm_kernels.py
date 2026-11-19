"""Loads the CUDA softmax / LayerNorm extension (compiled on first import).

    import llm_kernels
    y = llm_kernels.softmax(x)                      # over the last dimension
    z = llm_kernels.layer_norm(x, weight, bias)     # eps defaults to 1e-5

Forward only, float32, CUDA tensors. The first import compiles the extension with
torch.utils.cpp_extension (about a minute); later imports reuse the build.
"""
import os

from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)            # 02-llm-kernels
_REPO = os.path.dirname(_KERNELS)            # repository root

_ext = load(
    name="llm_kernels_ext",
    sources=[os.path.join(_HERE, "csrc", "llm_kernels.cu")],
    extra_include_paths=[
        _KERNELS,
        os.path.join(_KERNELS, "common"),
        os.path.join(_REPO, "common"),
    ],
    # No -std flag: cpp_extension picks the C++ standard this PyTorch needs
    # (recent releases require C++20).
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=os.environ.get("LLM_KERNELS_VERBOSE") == "1",
)

softmax = _ext.softmax
layer_norm = _ext.layer_norm

__all__ = ["softmax", "layer_norm"]
