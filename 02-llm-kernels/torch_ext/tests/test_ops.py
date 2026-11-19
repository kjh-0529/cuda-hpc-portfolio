"""Correctness tests for the extension against PyTorch (computed in float64).

Run from 02-llm-kernels/torch_ext:  python -m pytest -q tests
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_kernels  # noqa: E402  (compiles the extension on first import)

# Shapes cover every dispatcher path: warp per row (<=1024), block per row
# (<=8192), online softmax / Welford (>8192), and cols % 4 != 0 fallbacks.
SHAPES = [
    (8, 64), (8, 256), (4, 1000), (3, 5, 1024),   # warp per row, incl. 3-D input
    (16, 2048), (16, 4096), (2, 8192),            # block per row
    (2, 32768),                                   # too long for registers
    (7, 130), (5, 3001),                          # cols % 4 != 0
    (1, 4),                                       # tiny
]


@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda") * 5
    ref = torch.softmax(x.double(), dim=-1).float()
    torch.testing.assert_close(llm_kernels.softmax(x), ref, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_matches_torch(shape):
    torch.manual_seed(1)
    cols = shape[-1]
    x = 100 + 5 * torch.rand(*shape, device="cuda")  # large mean, small spread
    w = 1 + 0.5 * torch.rand(cols, device="cuda")
    b = 0.5 * torch.rand(cols, device="cuda") - 0.25
    ref = F.layer_norm(x.double(), (cols,), w.double(), b.double(), eps=1e-5).float()
    torch.testing.assert_close(llm_kernels.layer_norm(x, w, b, 1e-5), ref, atol=5e-4, rtol=1e-3)


def test_misaligned_view_is_handled():
    # A view starting 4 bytes into its storage is not 16-byte aligned; the
    # extension must copy it rather than issue misaligned float4 loads.
    base = torch.randn(4 * 1024 + 1, device="cuda")
    x = base[1:].view(4, 1024)
    assert x.data_ptr() % 16 != 0
    torch.testing.assert_close(llm_kernels.softmax(x), torch.softmax(x, -1), atol=1e-6, rtol=1e-4)


def test_non_contiguous_input():
    x = torch.randn(256, 64, device="cuda").t()  # [64, 256], not contiguous
    torch.testing.assert_close(llm_kernels.softmax(x), torch.softmax(x, -1), atol=1e-6, rtol=1e-4)


def test_runs_on_current_stream():
    s = torch.cuda.Stream()
    x = torch.randn(512, 1024, device="cuda")
    with torch.cuda.stream(s):
        y = llm_kernels.softmax(x)
    s.synchronize()
    torch.testing.assert_close(y, torch.softmax(x, -1), atol=1e-6, rtol=1e-4)


def test_rejects_bad_inputs():
    with pytest.raises(RuntimeError, match="CUDA"):
        llm_kernels.softmax(torch.randn(4, 8))
    with pytest.raises(RuntimeError, match="float32"):
        llm_kernels.softmax(torch.randn(4, 8, device="cuda").half())
    with pytest.raises(RuntimeError, match="weight"):
        x = torch.randn(4, 8, device="cuda")
        llm_kernels.layer_norm(x, torch.ones(7, device="cuda"), torch.zeros(8, device="cuda"))
