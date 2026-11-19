# LLM Kernels: Softmax & LayerNorm (planned)

Memory-bandwidth-bound kernels used in transformer inference, exposed to PyTorch as a C++/CUDA extension.

Planned steps:

1. Reduction: shared memory, then warp shuffle (`__shfl_down_sync`)
2. Softmax: naive 3-pass, then single-pass online softmax
3. LayerNorm: Welford's algorithm for mean and variance in one pass
4. FP16 / BF16 inputs
5. PyTorch extension (`torch.utils.cpp_extension`), compared with `torch.softmax` and `torch.nn.LayerNorm`
6. Optional: the same softmax in Triton

Metric: achieved bandwidth (GB/s) and % of the GPU's peak bandwidth, per sequence length.
