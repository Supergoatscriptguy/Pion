"""Pion — SFT data loading for instruction fine-tuning"""

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path

IGNORE_INDEX = -100

CHAT_TEMPLATE = {
    "user_start": "<|user|>\n",
    "user_end": "\n",
    "assistant_start": "<|assistant|>\n",
    "assistant_end": "\n",
    "system_start": "<|system|>\n",
    "system_end": "\n",
}


class SFTDataset(Dataset):
    def __init__(self, tokenizer, seq_len=2048, split="train", max_samples=None):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        ds = load_dataset("teknium/OpenHermes-2.5", split="train")

        n = len(ds)
        split_idx = int(n * 0.99)
        if split == "train":
            ds = ds.select(range(split_idx))
        else:
            ds = ds.select(range(split_idx, n))

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self.data = ds
        print(f"[SFT {split}] {len(self.data):,} conversations, seq_len={seq_len}")

    def _format_conversation(self, conversations):
        input_ids = []
        labels = []

        for turn in conversations:
            role = turn["from"]
            text = turn["value"]

            if role in ("human", "user"):
                prefix = CHAT_TEMPLATE["user_start"]
                suffix = CHAT_TEMPLATE["user_end"]
                prefix_ids = self.tokenizer.encode(prefix).ids
                content_ids = self.tokenizer.encode(text).ids
                suffix_ids = self.tokenizer.encode(suffix).ids
                turn_ids = prefix_ids + content_ids + suffix_ids
                input_ids.extend(turn_ids)
                labels.extend([IGNORE_INDEX] * len(turn_ids))

            elif role in ("gpt", "assistant"):
                prefix = CHAT_TEMPLATE["assistant_start"]
                suffix = CHAT_TEMPLATE["assistant_end"]
                prefix_ids = self.tokenizer.encode(prefix).ids
                content_ids = self.tokenizer.encode(text).ids
                suffix_ids = self.tokenizer.encode(suffix).ids
                input_ids.extend(prefix_ids)
                labels.extend([IGNORE_INDEX] * len(prefix_ids))
                input_ids.extend(content_ids + suffix_ids)
                labels.extend(content_ids + suffix_ids)

            elif role == "system":
                prefix = CHAT_TEMPLATE["system_start"]
                suffix = CHAT_TEMPLATE["system_end"]
                prefix_ids = self.tokenizer.encode(prefix).ids
                content_ids = self.tokenizer.encode(text).ids
                suffix_ids = self.tokenizer.encode(suffix).ids
                turn_ids = prefix_ids + content_ids + suffix_ids
                input_ids.extend(turn_ids)
                labels.extend([IGNORE_INDEX] * len(turn_ids))

        return input_ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        conversations = row["conversations"]
        input_ids, labels = self._format_conversation(conversations)

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            labels = labels[:self.seq_len]
        elif len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            input_ids = input_ids + [1] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_sft_dataloader(tokenizer, batch_size=8, seq_len=2048, num_workers=4,
                           split="train", max_samples=None, pin_memory=True):
    dataset = SFTDataset(tokenizer, seq_len=seq_len, split=split, max_samples=max_samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    tok_path = args.tokenizer or str(Path(__file__).parent.parent / "Tokenizers" / "pile" / "pile_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    loader = create_sft_dataloader(tokenizer, batch_size=args.batch_size,
                                    seq_len=args.seq_len, max_samples=args.max_samples,
                                    num_workers=0)

    sample = next(iter(loader))
    ids = sample["input_ids"][0]
    labs = sample["labels"][0]

    print(f"\nSample shapes: input_ids={ids.shape}, labels={labs.shape}")
    print(f"Non-masked tokens: {(labs != IGNORE_INDEX).sum().item()} / {len(labs)}")

    decoded = tokenizer.decode(ids.tolist())
    print(f"\nFull text (first 500 chars):\n{decoded[:500]}")

    mask_ratio = (labs == IGNORE_INDEX).float().mean().item()
    print(f"\nMask ratio: {mask_ratio:.1%} (higher = more prompt, less response)")

    print(f"\nBenchmark (10 batches)...")
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 10:
            break
    elapsed = time.time() - t0
    print(f"  {elapsed:.2f}s | {10 * args.batch_size / elapsed:.1f} samples/s")
