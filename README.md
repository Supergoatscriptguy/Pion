# Pion

Dense transformer language model. No experimental tricks, just every proven technique from recent research applied correctly. Part of the [NeuralML](https://github.com/Supergoatscriptguy) family alongside [Tachyon](https://github.com/Supergoatscriptguy/Tachyon) (MoE) and Lepton (experimental).

- **Tachyon** — Mixture of Experts, sparse routing
- **Lepton** — Experimental (thought tokens, memory, speculative decoding)
- **Pion** — Best-practices dense transformer. The boring stuff that works.

## Architecture

Everything here is backed by published results at small scale (SmolLM2, Gemma 2, Qwen3):

- **GQA** — Grouped Query Attention with 5:1 Q-to-KV head ratio. Better than MHA at this scale.
- **QK-Norm** — RMSNorm on Q and K before attention. Keeps training stable.
- **Logit Soft-Capping** — From Gemma 2. `cap * tanh(x / cap)` on attention scores. Can be disabled with `--no_soft_cap` to use FlashAttention/SDPA instead (much less memory).
- **SwiGLU** — `w2(silu(w1(x)) * w3(x))`. Industry standard FFN.
- **RoPE** — Rotary position embeddings, theta=10000.
- **Tied Embeddings** — Input/output share weights. Critical when embeddings are 20%+ of params.
- **Chunked Cross-Entropy** — Applies lm_head per chunk instead of materializing the full (N, vocab) logits tensor. Saves a ton of memory.
- **KV-Cache** — For fast autoregressive generation.

## Model Sizes

| Size | d_model | Layers | Heads (Q/KV) | d_ff | Params |
|------|---------|--------|--------------|------|--------|
| nano | 384 | 6 | 6/2 | 1024 | 29M |
| micro | 512 | 12 | 8/2 | 1408 | 60M |
| small | 640 | 16 | 10/2 | 1792 | 103M |
| base | 768 | 20 | 12/4 | 2048 | 165M |
| large | 1024 | 24 | 16/4 | 2816 | 322M |
| xl | 1536 | 28 | 16/4 | 4096 | 771M |

## Training

Pion-small (103M) was pretrained on RunPod across a couple of GPU rentals:

- **Steps 0-10K** on an A40 (48GB). Batch 64, grad_accum 1, `--no_soft_cap` with gradient checkpointing. ~4.3s/step, ~31K tok/s.
- **Steps 10K-22.5K** on an RTX A4000 (16GB). Batch 16, ~1.6s/step.
- **SFT** (3000 steps) on [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) using the same A4000. Cosine LR 2e-5, prompt masking so loss only trains on assistant responses.

Total pretraining: ~22.5K steps on ~750M tokens from [PreprocessedMIXED](https://huggingface.co/datasets/SuperGoatScriptGuy/PreprocessedMIXED). WSD learning rate schedule (warmup-stable-decay).

Checkpoints are on HuggingFace: [SuperGoatScriptGuy/Pion-SavePoint](https://huggingface.co/SuperGoatScriptGuy/Pion-SavePoint)

### Pretraining

```bash
pip install torch numpy tokenizers tqdm huggingface_hub
```

Data uses preprocessed `.npy` shards (shape `(sequences, 2048)`, token IDs). Grab some shards to test:

```python
from huggingface_hub import snapshot_download

allow_patterns = [f'shard_{i:05d}.npy' for i in range(50)]
snapshot_download(
    repo_id='SuperGoatScriptGuy/PreprocessedMIXED',
    repo_type='dataset',
    local_dir='./data',
    allow_patterns=allow_patterns,
)
```

```bash
# Memory test
python train.py --model_size nano --data_dir ./data --batch_size 32 --memory_test --no_soft_cap

# Quick local run
python train.py --model_size nano --data_dir ./data --batch_size 4 --max_steps 500 --no_soft_cap

# Real training (cloud GPU)
python train.py --model_size small --data_dir ./data --batch_size 64 --grad_accum 1 \
    --lr 3e-4 --min_lr 3e-5 --max_steps 30000 --no_soft_cap

# Resume from checkpoint
python train.py --model_size small --data_dir ./data --resume checkpoints/small/pion_small_10000.pt --no_soft_cap
```

### Fine-tuning (SFT)

```bash
pip install datasets

python finetune.py --checkpoint checkpoints/small/pion_small_22500.pt \
    --batch_size 16 --grad_accum 1 --lr 2e-5 --max_steps 3000 --no_soft_cap
```

This downloads OpenHermes 2.5 (~1.5GB) automatically and trains with prompt masking — the model only learns from the assistant's responses, not the user's questions.

### Useful Flags

- `--no_soft_cap` — Disable logit soft-capping, use SDPA/FlashAttention instead. Much less memory. Recommended for anything above nano.
- `--no_gradient_checkpointing` — Faster but uses way more VRAM.
- `--compile` — Enable torch.compile.
- `--eval_interval 500` — How often to run eval.
- `--save_interval 1000` — How often to save checkpoints.

## Generation

```bash
# Single prompt
python generate.py --checkpoint checkpoints/small/pion_small_final.pt --prompt "The meaning of life is"

# Interactive mode
python generate.py --checkpoint checkpoints/small/pion_small_final.pt

# Adjust sampling
python generate.py --checkpoint checkpoints/small/pion_small_final.pt --temperature 1.0 --top_k 100 --top_p 0.95
```

## Chat

```bash
python chat.py  # auto-selects latest checkpoint
python chat.py --checkpoint checkpoints/sft/pion_sft_final.pt
```

Commands: `/temp 0.8`, `/topk 50`, `/topp 0.9`, `/tokens 128`, `/quit`

## Files

- `model.py` — Architecture. GQA, QK-Norm, soft-capping, SwiGLU, RoPE, tied embeddings, chunked CE.
- `data.py` — Streaming shard-based data loader for pretraining.
- `train.py` — Pretraining loop. WSD schedule, BF16, gradient checkpointing.
- `finetune.py` — SFT training loop. Cosine schedule, prompt masking.
- `sft_data.py` — OpenHermes 2.5 data loading with chat template formatting.
- `generate.py` — KV-cache generation with top-k/top-p sampling.
- `chat.py` — Interactive generation wrapper.

## Tokenizer

Uses a BPE tokenizer (50K vocab) trained on The Pile. Grab it from [Tokenizers](https://github.com/Supergoatscriptguy/Tokenizers) or just use the `pile_tokenizer.json` included here.

## Notes

At 103M params, Pion-small is a proof-of-concept. It can produce coherent English after pretraining but instruction following after SFT is limited — you really need 1B+ params for that. The architecture and training pipeline are solid though, and scale up to xl (771M) without changes.

Soft-capping is cool in theory but materializes the full O(N^2) attention matrix, which OOMs on anything above nano. In practice you want `--no_soft_cap` to use SDPA, which is fused and O(N) memory. Gradient checkpointing is also basically mandatory for batch sizes above ~8 on consumer GPUs.
