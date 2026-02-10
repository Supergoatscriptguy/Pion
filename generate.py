"""Pion — Text generation with KV-cache, top-k/top-p sampling"""

import argparse
import time
import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import Pion, PionConfig, get_model, fmt


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt and ckpt["config"] is not None:
        config = ckpt["config"]
        model = Pion(config)
    else:
        state = ckpt["model"]
        d_model = state["embed.weight"].shape[1]
        sizes = {384: "nano", 512: "micro", 640: "small", 768: "base", 1024: "large", 1536: "xl"}
        model = get_model(sizes.get(d_model, "small"))

    model.load_state_dict(ckpt["model"])
    model = model.to(device).to(torch.bfloat16).eval()

    step = ckpt.get("step", "?")
    tokens = ckpt.get("tokens_seen", 0)
    params = model.count_parameters()
    print(f"Loaded Pion ({fmt(params)} params) from step {step} ({fmt(tokens)} tokens seen)")
    return model


def load_tokenizer(path=None):
    if path is None:
        path = str(Path(__file__).parent.parent / "Tokenizers" / "pile" / "pile_tokenizer.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.8,
             top_k=50, top_p=0.9, device=None, stream=False):
    if device is None:
        device = next(model.parameters()).device

    ids = tokenizer.encode(prompt).ids
    generated = list(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    kv_caches = model.create_kv_caches()

    # Prefill
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(input_ids, kv_caches=kv_caches)

    t0 = time.perf_counter()
    gen_count = 0

    for _ in range(max_new_tokens):
        logits = out["logits"][:, -1, :]

        if temperature > 0:
            logits = logits / temperature
        else:
            # greedy
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tok_id = next_tok.item()
            if tok_id == 0:
                break
            generated.append(tok_id)
            gen_count += 1
            if stream:
                print(tokenizer.decode([tok_id]), end="", flush=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(next_tok, kv_caches=kv_caches)
            continue

        # Top-k
        if top_k > 0:
            threshold = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[logits < threshold] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices_to_remove = remove.scatter(1, sorted_idx, remove)
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tok_id = next_tok.item()

        if tok_id == 0:
            break

        generated.append(tok_id)
        gen_count += 1

        if stream:
            print(tokenizer.decode([tok_id]), end="", flush=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(next_tok, kv_caches=kv_caches)

    elapsed = time.perf_counter() - t0
    tok_per_sec = gen_count / elapsed if elapsed > 0 else 0

    if stream:
        print()

    return {
        "text": tokenizer.decode(generated),
        "prompt_tokens": len(ids),
        "generated_tokens": gen_count,
        "tok_per_sec": tok_per_sec,
        "elapsed": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--stream", action="store_true", default=True)
    p.add_argument("--no_stream", action="store_true")
    args = p.parse_args()

    if args.no_stream:
        args.stream = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"\nTemp: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("=" * 60)

    if args.prompt:
        print(f"\n{args.prompt}", end="")
        result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                         device=device, stream=args.stream)
        if not args.stream:
            print(result["text"])
        print(f"\n[{result['generated_tokens']} tokens, {result['tok_per_sec']:.1f} tok/s]")
    else:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt.strip():
                continue

            print()
            result = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             device=device, stream=args.stream)
            if not args.stream:
                print(result["text"])
            print(f"[{result['generated_tokens']} tokens, {result['tok_per_sec']:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
