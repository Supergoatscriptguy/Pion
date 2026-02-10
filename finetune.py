"""Pion — Supervised fine-tuning on instruction data"""

import os, sys, argparse, time, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm import tqdm

from model import Pion, PionConfig, get_model, fmt
from sft_data import create_sft_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min_lr", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--no_soft_cap", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_lr(step, warmup_steps, max_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
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
def evaluate(model, tokenizer, seq_len, batch_size, device, max_batches=20):
    model.eval()
    loader = create_sft_dataloader(tokenizer, batch_size=batch_size, seq_len=seq_len,
                                    num_workers=0, split="val", max_samples=max_batches * batch_size)
    total_loss = 0
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, labels=labels)
        total_loss += out["loss"].item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="What is the meaning of life?", max_tokens=100, temperature=0.7):
    model.eval()
    formatted = "<|user|>\n" + prompt + "\n<|assistant|>\n"
    ids = tokenizer.encode(formatted).ids
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


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Load pretrained checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_soft_cap = not getattr(args, 'no_soft_cap', False)

    if "config" in ckpt and ckpt["config"] is not None:
        config = ckpt["config"]
        if hasattr(config, 'use_soft_cap'):
            config.use_soft_cap = use_soft_cap
        model = Pion(config)
    else:
        state = ckpt["model"]
        d_model = state["embed.weight"].shape[1]
        sizes = {384: "nano", 512: "micro", 640: "small", 768: "base", 1024: "large", 1536: "xl"}
        model = get_model(sizes.get(d_model, "small"), use_soft_cap=use_soft_cap)

    model.load_state_dict(ckpt["model"])
    model = model.to(device).to(torch.bfloat16)
    params = model.count_parameters()
    print(f"Loaded pretrained: {fmt(params)} params from step {ckpt.get('step', '?')}")

    use_gc = not args.no_gradient_checkpointing
    if use_gc:
        enable_gradient_checkpointing(model)
        print("Gradient checkpointing: on")

    # Tokenizer
    from tokenizers import Tokenizer
    tok_path = args.tokenizer or str(Path(__file__).parent / "pile_tokenizer.json")
    if not Path(tok_path).exists():
        tok_path = str(Path(__file__).parent.parent / "Tokenizers" / "pile" / "pile_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Tokenizer: {tokenizer.get_vocab_size()} vocab")

    # Optimizer — lower weight decay for SFT
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
    print(f"Loading OpenHermes 2.5...")
    dataloader = create_sft_dataloader(tokenizer, batch_size=args.batch_size,
                                        seq_len=args.seq_len, num_workers=args.num_workers,
                                        split="train", max_samples=args.max_samples)
    data_iter = iter(dataloader)

    save_dir = Path(args.save_dir) / "sft"
    save_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"Effective batch: {args.batch_size * args.grad_accum} ({fmt(tokens_per_step)} tok/step)")
    print(f"Schedule: cosine (warmup={args.warmup_steps}, max={args.max_steps})")
    print(f"LR: {args.lr} -> {args.min_lr}")
    print("-" * 60)

    model.train()
    opt.zero_grad()
    smooth_loss = None
    t0 = time.time()

    pbar = tqdm(range(args.max_steps), desc="sft", unit="step", dynamic_ncols=True)

    for step in pbar:
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
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
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, labels=labels)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            step_loss += out["loss"].item() / args.grad_accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        opt.zero_grad()

        smooth_loss = step_loss if smooth_loss is None else 0.95 * smooth_loss + 0.05 * step_loss
        elapsed = time.time() - t0
        tok_per_sec = tokens_per_step / max(elapsed, 1e-6)
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        ppl = math.exp(min(smooth_loss, 20))

        pbar.set_postfix_str(
            f"loss={smooth_loss:.3f} ppl={ppl:.1f} lr={lr:.1e} {fmt(tok_per_sec)}t/s {mem:.1f}GB"
        )
        t0 = time.time()

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, tokenizer, args.seq_len, args.batch_size, device, args.eval_steps)
            val_ppl = math.exp(min(val_loss, 20))
            tqdm.write(f"  eval | loss {val_loss:.4f} | ppl {val_ppl:.1f}")

            text = generate_sample(model, tokenizer, device)
            preview = text[:300].replace("\n", " ")
            tqdm.write(f"  sample: {preview}")

        if (step + 1) % args.save_interval == 0:
            path = save_dir / f"pion_sft_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "config": model.config if hasattr(model, "config") else None,
            }, path)
            tqdm.write(f"  saved {path}")

    pbar.close()

    path = save_dir / f"pion_sft_final.pt"
    torch.save({
        "step": args.max_steps,
        "model": model.state_dict(),
        "config": model.config if hasattr(model, "config") else None,
    }, path)

    print(f"\nGenerating final samples...")
    for prompt in ["What is machine learning?", "Write a haiku about coding.", "Explain gravity to a 5 year old."]:
        text = generate_sample(model, tokenizer, device, prompt=prompt)
        print(f"\n  Q: {prompt}")
        response = text.split("<|assistant|>\n")[-1] if "<|assistant|>" in text else text
        print(f"  A: {response[:300]}")

    print(f"\nDone! Checkpoint saved to {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
