"""Pion — Interactive text generation"""

import sys
from pathlib import Path
from generate import load_checkpoint, load_tokenizer, generate

import torch


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    args = p.parse_args()

    if args.checkpoint is None:
        ckpt_dir = Path(__file__).parent / "checkpoints"
        if not ckpt_dir.exists():
            print("No checkpoints/ directory. Train a model first.")
            sys.exit(1)
        pts = sorted(ckpt_dir.rglob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not pts:
            print("No .pt files found. Train a model first.")
            sys.exit(1)
        args.checkpoint = str(pts[0])
        print(f"Auto-selected: {args.checkpoint}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)

    temp = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens

    print(f"\nPion Chat — generate text from any prompt")
    print(f"Commands: /quit, /temp <n>, /topk <n>, /topp <n>, /tokens <n>")
    print(f"Settings: temp={temp}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}\n")

    while True:
        try:
            prompt = input("you > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        if prompt in ["/quit", "/q", "/exit"]:
            break

        if prompt.startswith("/temp "):
            try:
                temp = float(prompt.split()[1])
                print(f"  temperature = {temp}")
            except (ValueError, IndexError):
                print("  usage: /temp 0.8")
            continue

        if prompt.startswith("/topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"  top_k = {top_k}")
            except (ValueError, IndexError):
                print("  usage: /topk 50")
            continue

        if prompt.startswith("/topp "):
            try:
                top_p = float(prompt.split()[1])
                print(f"  top_p = {top_p}")
            except (ValueError, IndexError):
                print("  usage: /topp 0.9")
            continue

        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  max_tokens = {max_tokens}")
            except (ValueError, IndexError):
                print("  usage: /tokens 128")
            continue

        print()
        result = generate(model, tokenizer, prompt, max_new_tokens=max_tokens,
                         temperature=temp, top_k=top_k, top_p=top_p,
                         device=device, stream=True)
        print(f"[{result['generated_tokens']} tokens, {result['tok_per_sec']:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
