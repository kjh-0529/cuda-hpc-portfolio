# Distributed Training with PyTorch DDP (planned)

A small GPT-style language model (tens of millions of parameters) trained on public text, scaled from one GPU to two with PyTorch DistributedDataParallel. The same questions I work on in MPI codes, applied to AI training: how much does communication cost, and how close to linear does it scale?

Target hardware: 2x Tesla T4 (Kaggle).

Planned steps:

1. Single-GPU baseline: model, data pipeline, tokens/s, GPU memory, loss curve
2. Mixed precision (FP16 AMP) to use the T4's Tensor Cores; compare speed and memory
3. DDP on 2 GPUs with `torchrun` and NCCL; throughput and scaling efficiency from 1 to 2 GPUs
4. Communication analysis: gradient bucket size, gradient accumulation with `no_sync()` to communicate less often, PyTorch Profiler timeline
5. Optional: FSDP to compare memory savings

Metrics: tokens/s per configuration, 1 → 2 GPU scaling efficiency, matching loss curves for 1 and 2 GPUs (same convergence), profiler timeline.
