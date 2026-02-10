"""Pion — Shard-based data loading for text generation"""

import json
import numpy as np
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class ShardedDataset(Dataset):
    def __init__(self, data_dir, seq_len=2048, split="train", train_ratio=0.98):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        self.shard_paths = sorted(self.data_dir.glob("shard_*.npy"))
        if not self.shard_paths:
            raise ValueError(f"No shard_*.npy files found in {data_dir}")

        n_shards = len(self.shard_paths)
        split_idx = int(n_shards * train_ratio)
        self.shard_paths = self.shard_paths[:split_idx] if split == "train" else self.shard_paths[split_idx:]

        sample = np.load(self.shard_paths[0], mmap_mode="r")
        self.sequences_per_shard = sample.shape[0]
        self.total_sequences = len(self.shard_paths) * self.sequences_per_shard

        print(f"[{split}] {len(self.shard_paths)} shards, {self.total_sequences:,} sequences")

        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  {meta.get('total_tokens', 0):,} tokens")

        self.cached_shard_idx = -1
        self.cached_shard = None

    def _load_shard(self, shard_idx):
        if shard_idx != self.cached_shard_idx:
            self.cached_shard = np.load(self.shard_paths[shard_idx])
            self.cached_shard_idx = shard_idx
        return self.cached_shard

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        shard_idx = idx // self.sequences_per_shard
        seq_idx = idx % self.sequences_per_shard
        shard = self._load_shard(shard_idx)
        tokens = shard[seq_idx]

        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        elif len(tokens) < self.seq_len:
            tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), constant_values=1)

        return {"input_ids": torch.from_numpy(tokens.astype(np.int64))}


class StreamingShardDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=2048, split="train", train_ratio=0.98,
                 shuffle=True, seed=42):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

        self.shard_paths = sorted(self.data_dir.glob("shard_*.npy"))
        if not self.shard_paths:
            raise ValueError(f"No shard_*.npy files found in {data_dir}")

        n_shards = len(self.shard_paths)
        split_idx = int(n_shards * train_ratio)
        self.shard_paths = self.shard_paths[:split_idx] if split == "train" else self.shard_paths[split_idx:]

        print(f"[{split}] {len(self.shard_paths)} shards (streaming)")

    def _get_worker_shards(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return list(self.shard_paths)
        per_worker = len(self.shard_paths) // worker_info.num_workers
        start = worker_info.id * per_worker
        if worker_info.id == worker_info.num_workers - 1:
            return self.shard_paths[start:]
        return self.shard_paths[start:start + per_worker]

    def __iter__(self):
        shards = self._get_worker_shards()

        if self.shuffle:
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = self.seed + (worker_info.id if worker_info else 0)
            rng = random.Random(worker_seed)
            shards = list(shards)
            rng.shuffle(shards)

        for shard_path in shards:
            data = np.load(shard_path)
            n_sequences = data.shape[0]

            indices = list(range(n_sequences))
            if self.shuffle:
                random.shuffle(indices)

            for seq_idx in indices:
                tokens = data[seq_idx]
                if len(tokens) > self.seq_len:
                    tokens = tokens[:self.seq_len]
                elif len(tokens) < self.seq_len:
                    tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), constant_values=1)

                yield {"input_ids": torch.from_numpy(tokens.astype(np.int64))}


def create_dataloader(data_dir, batch_size=64, seq_len=2048, num_workers=4,
                      split="train", streaming=True, pin_memory=True, prefetch_factor=2):
    if streaming:
        dataset = StreamingShardDataset(data_dir=data_dir, seq_len=seq_len, split=split,
                                        shuffle=(split == "train"))
        shuffle = False
    else:
        dataset = ShardedDataset(data_dir=data_dir, seq_len=seq_len, split=split)
        shuffle = (split == "train")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True, persistent_workers=num_workers > 0,
    )


def create_dataloaders(data_dir, batch_size=64, seq_len=2048, num_workers=4, streaming=True):
    train_loader = create_dataloader(data_dir=data_dir, batch_size=batch_size, seq_len=seq_len,
                                     num_workers=num_workers, split="train", streaming=streaming)
    val_loader = create_dataloader(data_dir=data_dir, batch_size=batch_size, seq_len=seq_len,
                                   num_workers=num_workers, split="val", streaming=streaming)
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()

    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}, Workers: {args.num_workers}")

    loader = create_dataloader(data_dir=args.data_dir, batch_size=args.batch_size,
                               seq_len=args.seq_len, num_workers=args.num_workers)

    print("\nWarmup...")
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(f"  Batch {i}: {batch['input_ids'].shape} {batch['input_ids'].dtype}")

    print(f"\nBenchmark ({args.num_batches} batches)...")
    start = time.time()
    tokens = 0
    for i, batch in enumerate(loader):
        tokens += batch["input_ids"].numel()
        if i >= args.num_batches - 1:
            break

    elapsed = time.time() - start
    print(f"\n  {elapsed:.2f}s | {tokens/elapsed/1e6:.2f}M tok/s | {args.num_batches/elapsed:.1f} batch/s")
