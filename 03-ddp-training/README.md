# Distributed Training with PyTorch DDP

A small GPT-style language model trained on one GPU and then two with PyTorch DistributedDataParallel. The same questions I work on in MPI codes, applied to AI training: how much does communication cost, and how close to linear does it scale?

Status: code ready and checked on CPU; GPU measurements pending. Target hardware: 2x Tesla T4 (Kaggle).

## Setup

- **Model** ([`model.py`](model.py)): decoder-only transformer written from scratch. 6 layers, 8 heads, width 512, context 256 tokens, pre-norm blocks, tied input/output embeddings, PyTorch `scaled_dot_product_attention`. **19.2M parameters, about 125 MFLOPs per training token.**
- **Data** ([`data.py`](data.py)): byte-level (vocabulary of 256 byte values, no tokenizer), Tiny Shakespeare by default (1.1 MB, public domain), 90/10 train/validation split. Each GPU samples random windows with its own seed, so no sampler or sharding is needed.
- **Training** ([`train.py`](train.py)): AdamW, linear warm-up then cosine decay, gradient clipping. One script for one GPU (`python train.py`) and several (`torchrun --nproc_per_node=N train.py`). Options for FP16 mixed precision (`--amp fp16`), gradient accumulation with or without `no_sync()`, DDP bucket size, and a `torch.profiler` trace.
- **Metrics**: throughput in tokens/s, measured after 20 warm-up steps with evaluation time excluded; MFU (model FLOPs utilization) against the T4's dense peak (8.1 TFLOPS FP32, 65 TFLOPS FP16); peak GPU memory; training and validation loss curves. Each run writes a JSON file; [`summarize.py`](summarize.py) turns them into a table.

## Experiments

| # | Question | Runs |
|---|---|---|
| 1 | Single-GPU baseline | 1 GPU, FP32 |
| 2 | What does mixed precision buy on the T4's Tensor Cores? | 1 GPU, FP16 |
| 3 | How well does throughput scale to 2 GPUs? | 2 GPUs, FP16, same per-GPU batch as run 2 → scaling efficiency |
| 4 | Does 2-GPU training converge like 1-GPU training? | 1 GPU with 2 accumulation steps vs 2 GPUs, same global batch → compare loss curves |
| 5 | What does communication cost, and can it be hidden? | 2 GPUs with gradient accumulation, with and without `no_sync()`; different bucket sizes; profiler trace |

Scaling efficiency = tokens/s on N GPUs ÷ (N × tokens/s on 1 GPU).

## Checks done before the GPU runs

- Model: 19.2M parameters as designed; initial loss 5.57, matching ln(256) = 5.55 for a uniform guess over 256 bytes; a tiny configuration learns a repetitive text (loss 5.57 → 0.23 in 60 CPU steps).
- Found and fixed a bug that would have stopped the first GPU step: the target tensor is a strided slice, so `view()` failed; it now uses `reshape()`.
- DDP logic with 2 CPU processes (`torchrun`, gloo backend): 1 process with 2 accumulation steps and 2 processes with 1 step, i.e. the same global batch, reached the same validation loss (4.173 vs 4.171) after 40 steps. The `no_sync()` path and small buckets also ran. The script falls back to CPU only for such logic checks; CPU numbers are not measurements.

## Results

Pending (Tesla T4 x2).

## Run

On Kaggle, set the accelerator to **GPU T4 x2** and enable internet (the corpus is downloaded on first use).

```bash
python train.py --amp none --out results/1gpu_fp32.json
python train.py --amp fp16 --out results/1gpu_fp16.json
torchrun --nproc_per_node=2 train.py --amp fp16 --out results/2gpu_fp16.json
python train.py --amp fp16 --grad-accum 2 --out results/1gpu_fp16_accum2.json
python summarize.py results/*.json
```
