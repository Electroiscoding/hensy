---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- causal-lm
- dense-transformer
- sft
- chat
- h100
- runpod
base_model: Phase-Technologies/hensy-330m-pretrain
---

# Hensy-330M-SFT (Instruction Aligned Assistant)

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Parameters](https://img.shields.io/badge/Parameters-328.5M%20Total%20%7C%20Dense-green.svg)]()
[![Base Model](https://img.shields.io/badge/Base-5.0B%20Pretrain-purple.svg)](https://huggingface.co/Phase-Technologies/hensy-330m-pretrain)
[![Hardware](https://img.shields.io/badge/Trained%20On-4%C3%97%20NVIDIA%20H100%2080GB%20SXM5-76B900.svg?logo=nvidia)]()
[![SFT Volume](https://img.shields.io/badge/SFT-20M%20Tokens-orange.svg)]()

</div>

---

## ⚡ Overview

**Hensy-330M-SFT** is the conversational instruction-aligned checkpoint of **Hensy-330M**, a pure dense causal transformer pretrained from scratch across **5.00 Billion tokens** on a 4× NVIDIA H100 80GB SXM5 NVLink cluster. 

Post-training was performed via **Supervised Fine-Tuning (SFT)** across 20 Million tokens of multi-turn conversational dialogue with per-token assistant loss masking.

- **Base Model**: Initialized from [Phase-Technologies/hensy-330m-pretrain](https://huggingface.co/Phase-Technologies/hensy-330m-pretrain) (5.0B tokens).
- **Architecture**: 24 Pre-LayerNorm Transformer layers, 768 hidden dimension, 12 attention heads with RoPE and SDPA, 206,464 vocabulary (CRAYON Turbo), weight tying.
- **Precision**: Native BF16 with Safetensors support.

---

## 🚀 Readymade 1-Click Google Colab Code

Paste this **single cell** directly into any [Google Colab Notebook](https://colab.research.google.com/) (works on free T4 GPU or CPU):

```python
# 1. Install dependencies
!pip install -q torch huggingface_hub xerv-crayon safetensors

import sys
import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "Phase-Technologies/hensy-330m-sft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

print(f"🚀 Initializing Hensy-330M-SFT on {DEVICE} ({DTYPE})...")

# 2. Download architecture files & weights
for f in ["config.py", "model.py"]:
    if not Path(f).exists():
        hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=".")

cfg_json = hf_hub_download(repo_id=REPO_ID, filename="final/model_config.json")
st_path = hf_hub_download(repo_id=REPO_ID, filename="final/model.safetensors")

# 3. Load model & tokenizer
sys.path.insert(0, ".")
from config import HensyConfig
from model import build_hensy_330m
from tokenizer.crayon_wrapper import ForgeTokenizer

tokenizer = ForgeTokenizer()
cfg = HensyConfig.from_json(cfg_json)
model = build_hensy_330m(cfg)

weights = load_file(st_path)
weights = {k.replace("_orig_mod.", ""): v.to(DTYPE) for k, v in weights.items()}
model.load_state_dict(weights, strict=False)
model = model.to(DEVICE).to(DTYPE).eval()
print("✅ Hensy-330M-SFT loaded cleanly and ready!\n")

# 4. Interactive Chat Function
def ask_hensy(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
    formatted = f"<|USR|>{prompt.strip()}<|/USR|>\n<|ASST|>"
    input_ids = torch.tensor([tokenizer.encode(formatted, add_bos=False, add_eos=False)], device=DEVICE)
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=1.15)
        else:
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=1.15)
    
    reply = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    if "<|ASST|>" in reply:
        reply = reply.split("<|ASST|>")[-1].replace("<|/ASST|>", "").replace("<|eos|>", "").strip()
    return reply

# Test immediate inference:
print("User > What is machine learning?")
print(f"Hensy > {ask_hensy('What is machine learning?')}\n")

# Ask anything else:
while True:
    try:
        user_input = input("You > ")
        if user_input.strip().lower() in ("exit", "quit", "q"):
            break
        print(f"Hensy > {ask_hensy(user_input)}\n")
    except KeyboardInterrupt:
        break
```

---

## 🏷️ Chat Prompt Template

```text
<|USR|>Your message here<|/USR|>
<|ASST|>Assistant response here...
```

---

## a note from the creator.....

building hensy was our redemption arc and our return to discipline..... after uncovering the identity copy shortcut in our earlier experiments, we tore the training loop down to bare metal, fixed every single index shift and dtype promotion from first principles, and spun up 4x nvidia h100 sxm5s to run a clean, uncompromised 5 billion token pretrain from raw scratch.....

watching the loss drop smoothly from 12.4 down to 4.3 over 5 billion tokens at more than 565,000 tokens every second was the most rewarding sight of this entire journey..... following pretrain immediately with supervised fine-tuning cemented the turn-taking and conversational capability without needing heavy alignment crutches.....

endless gratitude and huge shoutout to the runpod.io team for providing the world-class infrastructure, blazing nvlink fabric, and uninterrupted stability that makes training real foundation models possible for independent builders on a personal budget..... to everyone reading this: build from first principles, check your loss gradients on live silicon, and keep pushing open source forward.....

— soham pal (phase technologies)
