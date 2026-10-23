# Hensy (330M)

**Hensy** is a pure dense (non-LLaMA, Anthropic/GPT-style) causal language model architecture built from first principles for maximum hardware FLOPs utilization (MFU).

---

## Architecture Specifications (330M)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **328.5M** (`0.329B`) | 100% dense (every parameter active on every token) |
| **Reasoning Parameters** | **170.0M** (**51.7%**) | Active transformer blocks (MHA + GeLU MLP) |
| **Embedding Parameters** | **158.6M** (**48.3%**) | Weight-tied to output LM head |
| **Layers** | **24** | Full pre-LN Transformer blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | **768** | Model width |
| **Attention Heads** | **12** | Head dimension 64 |
| **Attention Backend** | **SDPA** | PyTorch native fused FlashAttention-2 / FlashAttention-3 |
| **Positional Encoding** | **RoPE** | Rotary position embeddings with cached frequencies |
| **FFN / MLP** | **Dense GeLU** | 2 matmuls, $4\times$ expansion ($768 \to 3072 \to 768$) |
| **Normalization** | **LayerNorm** | Pre-LN architecture (unconditionally stable convergence) |
| **Vocabulary Size** | **206,464** | Multi-lingual & code vocabulary (`crayon_vocab`) |
| **Weight Tying** | **Enabled** | LM head shares weights with token embeddings |
| **Max Sequence Length** | **4,096 tokens** | Pretraining context window |

---

## 1. Pretraining (5.0 Billion Tokens)

### 4× H100 SXM (~2.4 hours, ~$32.50 compute cost)
```bash
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 run_pretrain.py \
  --data_dir /workspace/data/pretrain \
  --output_dir /workspace/checkpoints/hensy_330m \
  --total_tokens 5000000000 \
  --seq_len 4096 \
  --global_batch_tokens 4194304 \
  --micro_batch_per_gpu 8 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_ratio 0.02 \
  --grad_clip 1.0 \
  --bf16 \
  --save_every_steps 500
```

---

## 2. Supervised Fine-Tuning (SFT)

```bash
torchrun --nproc_per_node=4 run_sft.py \
  --base_model /workspace/checkpoints/hensy_330m/final \
  --data_dir /workspace/data/sft \
  --output_dir /workspace/checkpoints/hensy_sft \
  --total_tokens 20000000 \
  --seq_len 4096 \
  --global_batch_tokens 262144 \
  --micro_batch_per_gpu 4 \
  --lr_max 2e-5 \
  --lr_min 2e-6 \
  --warmup_ratio 0.03 \
  --bf16
```

---

## 3. Hugging Face Hub Repositories

- **Pretrained Base**: [Phase-Technologies/hensy-330m-pretrain](https://huggingface.co/Phase-Technologies/hensy-330m-pretrain)
- **SFT Aligned**: [Phase-Technologies/hensy-330m-sft](https://huggingface.co/Phase-Technologies/hensy-330m-sft)
