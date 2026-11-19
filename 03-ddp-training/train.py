"""Train the small GPT on one GPU or several with DistributedDataParallel.

Single GPU:   python train.py --amp fp16
Two GPUs:     torchrun --nproc_per_node=2 train.py --amp fp16

Global batch = micro_batch x grad_accum x number of GPUs. Throughput is measured
after `--warmup-steps` steps, excluding evaluation, and written with the loss
curve to a JSON file (--out) for comparison between runs.
"""
import argparse
import contextlib
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import RandomChunks, load_bytes
from model import GPT, GPTConfig

# Dense peak throughput of a Tesla T4, used for model FLOPs utilization (MFU).
T4_PEAK_TFLOPS = {"none": 8.1, "fp16": 65.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup-steps", type=int, default=20, help="steps excluded from timing")
    p.add_argument("--micro-batch", type=int, default=32, help="sequences per GPU per micro-step")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--amp", choices=["none", "fp16"], default="none")
    p.add_argument("--bucket-cap-mb", type=float, default=25.0, help="DDP gradient bucket size")
    p.add_argument("--sync-every-micro-step", action="store_true",
                   help="all-reduce on every micro-step instead of using no_sync()")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--profile", action="store_true", help="trace a few steps with torch.profiler")
    p.add_argument("--peak-tflops", type=float, default=None, help="for MFU; default: T4 peak")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default=None, help="JSON results file (rank 0)")
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--block-size", type=int, default=256)
    return p.parse_args()


def lr_at(step, total, base, warmup=50):
    """Linear warm-up, then cosine decay to 10% of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    # Without a GPU the script still runs on CPU (gloo backend), which is only
    # meant for checking the training and DDP logic, not for measurements.
    use_cuda = torch.cuda.is_available()
    if distributed:
        dist.init_process_group("nccl" if use_cuda else "gloo")
        rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    else:
        rank, local_rank = 0, 0
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    is_main = rank == 0
    torch.manual_seed(args.seed)  # identical initial weights on every rank

    # Rank 0 downloads the corpus first; the others wait, then read it.
    if is_main:
        load_bytes(args.data)
    if distributed:
        dist.barrier()
    train_data, val_data = load_bytes(args.data)

    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                    d_model=args.d_model)
    model = GPT(cfg).to(device)
    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                    bucket_cap_mb=args.bucket_cap_mb, gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=use_cuda)
    use_fp16 = args.amp == "fp16" and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16)

    train_iter = RandomChunks(train_data, cfg.block_size, args.micro_batch, device, args.seed + rank)
    tokens_per_step = args.micro_batch * cfg.block_size * args.grad_accum * world

    if is_main:
        name = torch.cuda.get_device_name(device) if use_cuda else "CPU (logic check only)"
        print(f"device: {name} x {world} | PyTorch {torch.__version__}")
        print(f"model: {raw_model.num_params() / 1e6:.1f}M params | amp={args.amp} | "
              f"global batch {args.micro_batch * args.grad_accum * world} seqs "
              f"({tokens_per_step} tokens/step)")

    @torch.no_grad()
    def evaluate():
        model.eval()
        it = RandomChunks(val_data, cfg.block_size, args.micro_batch, device, 10_000 + rank)
        total = torch.zeros((), device=device)
        for _ in range(args.eval_batches):
            x, y = it.next()
            with autocast():
                _, loss = raw_model(x, y)
            total += loss.float()
        total /= args.eval_batches
        if distributed:
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        model.train()
        return total.item()

    prof = None
    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
            schedule=torch.profiler.schedule(wait=args.warmup_steps, warmup=2, active=4, repeat=1))
        prof.start()

    log, evals = [], []
    timed_time, timed_tokens, t0 = 0.0, 0, None
    model.train()
    for step in range(args.steps):
        if step == args.warmup_steps:
            sync()
            t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)

        for micro in range(args.grad_accum):
            x, y = train_iter.next()
            last = micro == args.grad_accum - 1
            # Skip the gradient all-reduce on all but the last micro-step.
            sync_ctx = (contextlib.nullcontext() if (last or not distributed or args.sync_every_micro_step)
                        else model.no_sync())
            with sync_ctx:
                with autocast():
                    _, loss = model(x, y)
                scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if prof is not None:
            prof.step()

        if step >= args.warmup_steps:
            timed_tokens += tokens_per_step
        if step % args.log_every == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": loss.item()})
            if is_main:
                print(f"step {step:4d} | loss {loss.item():.4f}")
        if (step > 0 and step % args.eval_every == 0) or step == args.steps - 1:
            if t0 is not None:
                sync()
                timed_time += time.perf_counter() - t0
            v = evaluate()
            evals.append({"step": step, "val_loss": v})
            if is_main:
                print(f"step {step:4d} | val loss {v:.4f}")
            if t0 is not None:
                sync()
                t0 = time.perf_counter()

    if prof is not None:
        prof.stop()
        if is_main:
            os.makedirs("results", exist_ok=True)
            trace = f"results/trace_{world}gpu_{args.amp}.json"
            prof.export_chrome_trace(trace)
            print(prof.key_averages().table(
                sort_by="cuda_time_total" if use_cuda else "cpu_time_total", row_limit=15))
            print(f"trace written to {trace} (open in chrome://tracing or ui.perfetto.dev)")

    tok_s = timed_tokens / timed_time if timed_time > 0 else float("nan")
    peak = (args.peak_tflops or T4_PEAK_TFLOPS[args.amp]) * 1e12
    mfu = raw_model.flops_per_token() * (tok_s / world) / peak
    mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else 0.0
    result = {
        "gpus": world, "amp": args.amp, "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum, "global_batch_tokens": tokens_per_step,
        "bucket_cap_mb": args.bucket_cap_mb, "no_sync": not args.sync_every_micro_step,
        "params_m": round(raw_model.num_params() / 1e6, 2), "steps": args.steps,
        "tokens_per_s": round(tok_s), "tokens_per_s_per_gpu": round(tok_s / world),
        "mfu_pct": round(100 * mfu, 1), "peak_mem_gb": round(mem_gb, 2),
        "ms_per_step": round(1000 * tokens_per_step / tok_s, 1),
        "final_val_loss": evals[-1]["val_loss"] if evals else None,
        "train_log": log, "val_log": evals,
    }
    if is_main:
        print(f"\n{tok_s:,.0f} tokens/s ({tok_s / world:,.0f} per GPU) | "
              f"{result['ms_per_step']} ms/step | MFU {result['mfu_pct']}% | "
              f"peak memory {mem_gb:.2f} GB | final val loss {result['final_val_loss']:.4f}")
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
            print(f"results written to {args.out}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
