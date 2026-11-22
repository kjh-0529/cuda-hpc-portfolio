# Memory Bandwidth Benchmarking (STREAM-style)

How much of the hardware's memory bandwidth can code actually reach, and what decides it? Measured with STREAM-style kernels written from scratch: GPU first (Tesla T4), then CPU.

Status: GPU part measured on a Tesla T4 (all results verified); CPU part planned.

## GPU: `gpu_stream`

[`gpu_stream.cu`](gpu_stream.cu) runs five kernels on three arrays of doubles, 268 MB each (2^25 elements, as in BabelStream's default), the same kernels STREAM and BabelStream use:

| Kernel | Operation | Bytes moved per element |
|---|---|---|
| Copy | `c = a` | 16 |
| Mul | `b = s * c` | 16 |
| Add | `c = a + b` | 24 |
| Triad | `a = b + s * c` | 24 |
| Dot | `sum += a * b` | 16 |

It sweeps two things:

- **Thread-block size:** 64, 128, 256, 512, 1024 threads.
- **Where the arrays live:**
  - `device`: `cudaMalloc`, GPU memory.
  - `managed`: `cudaMallocManaged`, written first by the CPU. The first kernel has to migrate the pages to the GPU; that run is reported separately as `first_touch_copy`, later runs use GPU-resident pages.
  - `host`: pinned host memory mapped into the GPU (zero-copy), so every access crosses PCIe.

Every result is compared with the **theoretical peak**, computed from the memory clock and bus width the device reports (320 GB/s on a T4), and with **`cudaMemcpy`**, the baseline project 2 used. That closes a question project 2 left open: PyTorch's softmax ran at 242–244 GB/s, faster than `cudaMemcpy`'s 232 GB/s, so the copy was not the true ceiling.

Before timing, each configuration runs every kernel once from known values and checks the results (expected values are computed in the kernels' order with a 1e-12 relative tolerance, since 0.4 × 0.1 is not exactly 0.04 in binary and the GPU may fuse multiply-adds). Times are medians of batches after a GPU warm-up, the method used in the other projects.

## Predictions, written before measuring

- **Device memory:** Copy and Triad close to `cudaMemcpy` (about 230 GB/s, 72% of peak) or slightly above it, as PyTorch's softmax was. Add and Triad (two reads, one write) may edge out Copy and Mul.
- **Block size:** flat from 128 to 1024. Even 64 should reach full occupancy, since a Turing SM holds up to 16 blocks, 16 x 64 = 1,024 threads. Dot, with its fixed 1,024-block grid and block-level reduction, is most likely to be slower at small sizes.
- **Managed memory:** once the pages are on the GPU, the same as device memory. The first touch should be several times slower: the pages move over PCIe through page faults.
- **Mapped host memory:** limited by PCIe 3.0 x16, about 12 GB/s, roughly 4% of the GPU's peak.

## GPU results (Tesla T4)

Theoretical peak 320.1 GB/s (from the reported memory clock and bus width). Raw data: [`results/kaggle-t4.csv`](results/kaggle-t4.csv). Block size made almost no difference (next section), so this table shows the 1,024-thread runs.

| Kernel | Device memory | Managed (pages on GPU) | Mapped host memory |
|---|---|---|---|
| `cudaMemcpy` | 232.0 GB/s (72.5%) | — | — |
| Copy | 244.2 GB/s (76.3%) | 243.9 GB/s | 20.8 GB/s (6.5%) |
| Mul | 244.1 GB/s (76.3%) | 244.0 GB/s | 20.9 GB/s |
| Add | 254.2 GB/s (79.4%) | 255.7 GB/s | 18.2 GB/s |
| Triad | 255.3 GB/s (79.8%) | 256.0 GB/s | 18.2 GB/s |
| Dot | **271.4 GB/s (84.8%)** | 271.8 GB/s | 12.4 GB/s (3.9%) |
| Managed, first touch (Copy) | — | **3.6–4.4 GB/s** | — |

Checked against the predictions:

| Prediction | Result | |
|---|---|---|
| Copy and Triad close to `cudaMemcpy` or slightly above | Copy 244, Triad 255 vs. 232 GB/s | right: all kernels beat `cudaMemcpy` |
| Add and Triad edge out Copy and Mul | 254–256 vs. 244 GB/s | right |
| Block size flat from 128 to 1,024, 64 also full occupancy | within 2% from 64 to 1,024 | right |
| Dot slower at small block sizes | flat (269–274 GB/s) | wrong |
| Managed memory equals device memory once pages are on the GPU | same within 1% | right |
| First touch several times slower | **55–67x slower** (122–149 ms vs. 2.2 ms) | right direction, far larger |
| Mapped host memory about 12 GB/s | 12.4 GB/s per direction (see below) | right |

What it shows:

- **The more a kernel reads relative to what it writes, the closer it gets to the peak.** Read-only Dot reaches 85% of peak, Add and Triad (two reads per write) about 80%, Copy and Mul (one read per write) 76%. The likely reason is that switching the memory bus between reading and writing costs time, so writes are expensive; this was not measured directly.
- **`cudaMemcpy` is not the ceiling; it is below every hand-written kernel.** This answers the question from project 2, where PyTorch's softmax (242–244 GB/s) beat `cudaMemcpy` (232 GB/s): a plain copy kernel reaches 244 GB/s on this card, so PyTorch's softmax runs at copy-kernel speed, and project 2's "% of copy" numbers are about 5% generous. The realistic ceilings on a T4 are about 244 GB/s for one read per write and 255 GB/s for two.
- **Block size does not matter for streaming kernels on this GPU.** Even 64-thread blocks fill the SMs (16 blocks x 64 = 1,024 threads), and the kernels have no data reuse that a block size could change.
- **Managed memory is free once the data is on the GPU, and very expensive before.** Steady-state runs match device memory, but the first kernel after the CPU wrote the arrays ran at 3.6–4.4 GB/s, a third of what PCIe carries: pages move on demand through page faults, which is far slower than one bulk transfer (the usual explanation; the fault handling itself was not profiled). Moving them ahead of time (`cudaMemPrefetchAsync`) is the usual fix; it was not measured here.
- **Mapped host memory is limited by PCIe, about 12.4 GB/s in each direction.** The T4 sits on PCIe 3.0 x16, which carries traffic both ways at once. Dot only reads and reaches 12.4 GB/s; Add and Triad read 537 MB in the same time (12.3 GB/s read); Copy reads and writes equal amounts, so its 21 GB/s is about 10.8 GB/s in each direction at the same time. Everything over PCIe is about 20x slower than GPU memory.

## Run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build ../build -j
../build/04-memory-bandwidth/gpu_stream | tee results/kaggle-t4.csv
```

Options: `--n ELEMENTS`, `--modes device,managed,host`, `--tbsizes 64,128,256,512,1024`.

## Background

The methodology comes from HPC benchmarking work: STREAM on CPUs and BabelStream on GPUs, comparing compilers, thread placement, block sizes and memory types. Everything here is new code, run on public cloud hardware; no material from that work is included.
