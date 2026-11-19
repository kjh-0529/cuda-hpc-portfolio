# Distributed Training with PyTorch DDP

A small GPT-style language model trained on one GPU and then two with PyTorch DistributedDataParallel. The same questions I work on in MPI codes, applied to AI training: how much does communication cost, and how close to linear does it scale?

Status: experiments 1–4 measured on 2x Tesla T4 (Kaggle): FP16 is 4.0x faster than FP32, 2 GPUs scale at 93%, and 2-GPU training converges like 1-GPU training. Experiment 5 (communication) next.

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

2x Tesla T4 on Kaggle, PyTorch 2.11 (CUDA 13.0), 300 steps per run. Raw data: [`results/kaggle-t4x2.csv`](results/kaggle-t4x2.csv).

| Run | GPUs | Precision | Global batch (tokens) | tokens/s | ms/step | MFU | Peak memory | Final val loss |
|---|---|---|---|---|---|---|---|---|
| 1 GPU FP32 | 1 | FP32 | 8,192 | 27,120 | 302.1 | 41.7% of 8.1 TFLOPS | 1.98 GB | 2.047 |
| 1 GPU FP16 | 1 | FP16 AMP | 8,192 | 107,597 | 76.1 | 20.6% of 65 TFLOPS | 1.29 GB | 2.048 |
| **2 GPUs FP16** | 2 | FP16 AMP | 16,384 | **200,867** | 81.6 | 19.2% | 1.37 GB | **1.898** |
| 1 GPU FP16, 2 accumulation steps | 1 | FP16 AMP | 16,384 | 96,789 | 169.3 | 18.5% | 1.37 GB | **1.895** |

**1. Mixed precision: 4.0x faster, 35% less memory, same loss.** FP16 autocast moves the matrix multiplies onto the T4's Tensor Cores: 27,120 → 107,597 tokens/s, peak memory 1.98 → 1.29 GB, and the final validation loss is unchanged (2.047 vs 2.048).

**2. MFU tells a different story than speed.** FP32 uses 41.7% of the 8.1 TFLOPS FP32 peak. The SGEMM project showed that the T4 sustains only about 4.1 TFLOPS in FP32 under load, so the training loop runs at roughly 80% of what cuBLAS reaches. FP16 uses only 20.6% of the 65 TFLOPS Tensor Core peak: once the matrix multiplies are 8x faster, the parts that did not speed up (LayerNorm, softmax, GELU, the optimizer step, Python and data sampling) take a much larger share of each step. This is where fused kernels like the ones in project 2 pay off.

**3. Two GPUs scale at 93%.** With the same per-GPU batch, 2 GPUs process 200,867 tokens/s versus 2 x 107,597 = 215,194 for perfect scaling. Each step takes 81.6 ms instead of 76.1 ms; the 5.5 ms difference is mostly the all-reduce of 19.2M FP32 gradients (77 MB) between the two T4s, which are connected over PCIe without NVLink. DDP already overlaps most of that transfer with the backward pass by sending gradients in buckets while later layers are still computing.

**4. Two GPUs converge like one.** At the same global batch of 16,384 tokens, 2-GPU DDP reached a final validation loss of 1.898 and 1 GPU with 2 accumulation steps reached 1.895, within run-to-run noise. Data parallelism changed the speed, not the result. (Both are lower than the 8,192-token runs simply because they saw twice as many tokens in 300 steps.)

Measurement caveat: the runs were executed one after another on the same notebook. The accumulation run, which came last, is about 10% slower per micro-step than the earlier FP16 run, although it does the same work per micro-step and fewer optimizer steps. The likely cause is the T4 lowering its clock as it heats up, as measured in the SGEMM project. That means later runs, including the 2-GPU run, may be slightly penalized, and the 93% scaling efficiency is, if anything, a conservative estimate.

## Run

On Kaggle, set the accelerator to **GPU T4 x2** and enable internet (the corpus is downloaded on first use).

```bash
python train.py --amp none --out results/1gpu_fp32.json
python train.py --amp fp16 --out results/1gpu_fp16.json
torchrun --nproc_per_node=2 train.py --amp fp16 --out results/2gpu_fp16.json
python train.py --amp fp16 --grad-accum 2 --out results/1gpu_fp16_accum2.json
python summarize.py results/*.json
```
