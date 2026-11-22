# Memory Bandwidth Benchmarking (STREAM-style)

How much of the hardware's memory bandwidth can code actually reach, and what decides it? Measured with STREAM-style kernels written from scratch: GPU first (Tesla T4), then CPU.

Status: GPU benchmark implemented; results pending. CPU part planned.

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

## Run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build ../build -j
../build/04-memory-bandwidth/gpu_stream | tee results/kaggle-t4.csv
```

Options: `--n ELEMENTS`, `--modes device,managed,host`, `--tbsizes 64,128,256,512,1024`.

## Background

The methodology comes from HPC benchmarking work: STREAM on CPUs and BabelStream on GPUs, comparing compilers, thread placement, block sizes and memory types. Everything here is new code, run on public cloud hardware; no material from that work is included.
