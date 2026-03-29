"""
Script: Evaluate all trained LoRA adapters.
Usage: python scripts/evaluate_all.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import cfg
from src.training.evaluate import evaluate_adapter
from src.training.trainer import is_trained


def main():
    print(f"⏳ Loading {cfg.QWEN_MODEL}...")
    tokenizer  = AutoTokenizer.from_pretrained(cfg.QWEN_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.QWEN_MODEL,
        device_map=cfg.DEVICE, torch_dtype=torch.float16, trust_remote_code=True,
    )
    base_model.config.use_cache = False

    for name in cfg.LORA_INSTRUCTIONS:
        adapter_path = cfg.ADAPTER_DIR / name
        if not is_trained(name):
            print(f"\n⚠️  {name} — adapter not found, skipping")
            continue
        evaluate_adapter(name, str(adapter_path), base_model, tokenizer)


if __name__ == "__main__":
    main()
