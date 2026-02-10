"""Pion — Training with WSD schedule, BF16, gradient checkpointing"""

import os, sys, argparse, time, math
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm import tqdm

from model import Pion, PionConfig, get_model, fmt
from data import create_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="small", choices=["nano", "micro", "small", "base", "large", "xl"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--memory_test", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_soft_cap", action="store_true", help="Disable logit soft-capping to use SDPA (much less memory)")
    return p.parse_args()


def wsd_lr(step, warmup_steps, max_steps, lr, min_lr):
    """Warmup-Stable-Decay schedule. 5% warmup, 75% stable, 20% decay."""
    warmup_end = warmup_steps
    decay_start = int(max_steps * 0.80)

    if step < warmup_end:
        return lr * step / max(warmup_end, 1)
    elif step < decay_start:
        return lr
    else:
        progress = (step - decay_start) / max(max_steps - decay_start, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (lr - min_lr)


def enable_gradient_checkpointing(model):
    for layer in model.layers:
        layer._orig_forward = layer.forward

        def make_ckpt_forward(module):
            def ckpt_forward(x, mask=None, kv_cache=None):
                def custom_fwd(*inputs):
                    return module._orig_forward(*inputs)
                return torch.utils.checkpoint.checkpoint(custom_fwd, x, mask, kv_cache, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(layer)


@torch.no_grad()
def evaluate(model, data_dir, seq_len, batch_size, device, max_batches=20):
    model.eval()
    loader = create_dataloader(data_dir=data_dir, batch_size=batch_size, seq_len=seq_len,
                               num_workers=0, split="val", streaming=False, pin_memory=False)
    total_loss = 0
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, labels=x)
        total_loss += out["loss"].item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The meaning of life is", max_tokens=64, temperature=0.8):
    """Generate a sample during training to monitor quality."""
    model.eval()
    ids = tokenizer.encode(prompt).ids
    generated = list(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    kv_caches = model.create_kv_caches()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(input_ids, kv_caches=kv_caches)

    for _ in range(max_tokens):
        logits = out["logits"][:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tok_id = next_tok.item()
        if tok_id == 0:
            break
        generated.append(tok_id)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(next_tok, kv_caches=kv_caches)

    model.train()
    return tokenizer.decode(generated)


def memory_test(args):
    print("=" * 60)
    print("MEMORY TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No CUDA available")
        return False

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    use_gc = not args.no_gradient_checkpointing
    print(f"\n  Model: {args.model_size}")
    print(f"  Batch: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Grad accum: {args.grad_accum}")
    use_soft_cap = not getattr(args, 'no_soft_cap', False)
    print(f"  Grad checkpoint: {use_gc}")
    print(f"  Soft-capping: {use_soft_cap}")

    model = get_model(args.model_size, use_soft_cap=use_soft_cap).to(device).to(torch.bfloat16)
    params = model.count_parameters()
    print(f"  Params: {fmt(params)}")

    if use_gc:
        enable_gradient_checkpointing(model)

    if args.compile:
        model = torch.compile(model)

    mem_model = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  After model: {mem_model:.2f} GB")

    x = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device=device)

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x, labels=x)
        loss = out["loss"]
    mem_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  After forward: {mem_fwd:.2f} GB")

    loss.backward()
    mem_bwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  After backward: {mem_bwd:.2f} GB")

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  Peak: {mem_bwd:.2f} GB / {total_mem:.1f} GB ({100*mem_bwd/total_mem:.0f}%)")

    if mem_bwd < total_mem * 0.9:
        print("  PASS")
        return True
    elif mem_bwd < total_mem:
        print("  WARNING — tight fit")
        return True
    else:
        print("  FAIL — OOM")
        return False


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    use_soft_cap = not getattr(args, 'no_soft_cap', False)
    model = get_model(args.model_size, use_soft_cap=use_soft_cap).to(device).to(torch.bfloat16)
    params = model.count_parameters()
    print(f"Pion-{args.model_size}: {fmt(params)} params")
    if not use_soft_cap:
        print("Soft-capping: off (using SDPA)")

    use_gc = not args.no_gradient_checkpointing
    if use_gc:
        enable_gradient_checkpointing(model)
        print("Gradient checkpointing: on")

    if args.compile:
        print("Compiling...")
        model = torch.compile(model)

    # Optimizer
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if "norm" in name or "bias" in name else decay).append(p)

    use_fused = device.type == "cuda"
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), fused=use_fused)

    # Data
    print(f"Loading data from {args.data_dir}...")
    dataloader = create_dataloader(data_dir=args.data_dir, batch_size=args.batch_size,
                                   seq_len=args.seq_len, num_workers=args.num_workers)
    data_iter = iter(dataloader)

    # Tokenizer for sample generation
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tok_path = args.tokenizer or str(Path(__file__).parent.parent / "Tokenizers" / "pile" / "pile_tokenizer.json")
        if Path(tok_path).exists():
            tokenizer = Tokenizer.from_file(tok_path)
    except Exception:
        pass

    # Resume
    start_step = 0
    tokens_seen = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"Resumed from step {start_step} ({fmt(tokens_seen)} tokens)")

    save_dir = Path(args.save_dir) / args.model_size
    save_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"Effective batch: {args.batch_size * args.grad_accum} ({fmt(tokens_per_step)} tok/step)")
    print(f"Schedule: WSD (warmup={args.warmup_steps}, stable until {int(args.max_steps*0.8)}, decay to {args.max_steps})")
    print("-" * 60)

    model.train()
    opt.zero_grad()
    running_loss = 0.0
    smooth_loss = None
    t0 = time.time()

    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps,
                desc="training", unit="step", dynamic_ncols=True)

    for step in pbar:
        lr = wsd_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        step_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, labels=x)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            step_loss += out["loss"].item() / args.grad_accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        opt.zero_grad()
        tokens_seen += tokens_per_step

        smooth_loss = step_loss if smooth_loss is None else 0.95 * smooth_loss + 0.05 * step_loss
        running_loss += step_loss

        elapsed = time.time() - t0
        tok_per_sec = tokens_per_step / max(elapsed, 1e-6) if elapsed > 0 else 0
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        ppl = math.exp(min(smooth_loss, 20))

        pbar.set_postfix_str(
            f"loss={smooth_loss:.3f} ppl={ppl:.1f} lr={lr:.1e} {fmt(tok_per_sec)}t/s {fmt(tokens_seen)}tok {mem:.1f}GB"
        )
        t0 = time.time()

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, args.data_dir, args.seq_len, args.batch_size, device, args.eval_steps)
            val_ppl = math.exp(min(val_loss, 20))
            tqdm.write(f"  eval | loss {val_loss:.4f} | ppl {val_ppl:.1f}")

            if tokenizer is not None:
                text = generate_sample(model, tokenizer, device)
                preview = text[:200].replace("\n", " ")
                tqdm.write(f"  sample: {preview}")

        if (step + 1) % args.save_interval == 0:
            path = save_dir / f"pion_{args.model_size}_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": model.config if hasattr(model, "config") else None,
                "tokens_seen": tokens_seen,
            }, path)
            tqdm.write(f"  saved {path}")

    pbar.close()

    path = save_dir / f"pion_{args.model_size}_final.pt"
    torch.save({
        "step": args.max_steps,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": model.config if hasattr(model, "config") else None,
        "tokens_seen": tokens_seen,
    }, path)

    val_loss = evaluate(model, args.data_dir, args.seq_len, args.batch_size, device, args.eval_steps)
    print(f"\nFinal | loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.1f}")
    print(f"Total tokens: {fmt(tokens_seen)}")
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    if args.memory_test:
        success = memory_test(args)
        sys.exit(0 if success else 1)
    else:
        train(args)
