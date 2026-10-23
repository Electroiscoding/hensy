"""
Hensy-330M-SFT • Interactive Inference & Google Colab Quickstart
Run in Google Colab (Free T4 GPU or CPU):

!pip install -q torch huggingface_hub xerv-crayon safetensors
"""

import os
import sys
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "Phase-Technologies/hensy-330m-sft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

print(f"🚀 Initializing Hensy-330M-SFT on {DEVICE} ({DTYPE})...")

for fname in ["config.py", "model.py"]:
    if not Path(fname).exists():
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=".")

sys.path.insert(0, ".")
from config import HensyConfig
from model import build_hensy_330m
from tokenizer.crayon_wrapper import ForgeTokenizer

tokenizer = ForgeTokenizer()

cfg_path = hf_hub_download(repo_id=REPO_ID, filename="final/model_config.json")
cfg = HensyConfig.from_json(cfg_path)
model = build_hensy_330m(cfg)

try:
    st_path = hf_hub_download(repo_id=REPO_ID, filename="final/model.safetensors")
    sd = load_file(st_path)
except Exception:
    bin_path = hf_hub_download(repo_id=REPO_ID, filename="final/pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu")

sd = {k.replace("_orig_mod.", ""): v.to(DTYPE) for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model = model.to(DEVICE).to(DTYPE).eval()
print("✅ Hensy-330M-SFT loaded cleanly and ready!\n")


def chat(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, repetition_penalty: float = 1.15):
    formatted = f"<|USR|>{prompt.strip()}<|/USR|>\n<|ASST|>"
    input_ids = torch.tensor([tokenizer.encode(formatted, add_bos=False, add_eos=False)], device=DEVICE)

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)
        else:
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)

    res = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    if "<|ASST|>" in res:
        res = res.split("<|ASST|>")[-1].replace("<|/ASST|>", "").replace("<|eos|>", "").strip()
    return res


if __name__ == "__main__":
    print("=" * 60)
    print("Type your message below (or type 'quit' to exit):")
    print("=" * 60)
    test_q = "What is machine learning?"
    print(f"\nUser > {test_q}")
    print(f"Hensy > {chat(test_q)}")
    print("=" * 60)
    while True:
        try:
            usr = input("\nUser > ")
            if usr.strip().lower() in ("quit", "exit", "q"):
                break
            ans = chat(usr)
            print(f"Hensy > {ans}")
        except KeyboardInterrupt:
            break
