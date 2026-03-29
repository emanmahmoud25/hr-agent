"""
Script: Train all 4 LoRA adapters sequentially.
Usage: python scripts/train_all.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import cfg
from src.training.trainer import train_one, is_trained

LORA_NAMES = [
    "lora1_classification",
    "lora2_skills",
    "lora3_interview",
    "lora4_improvement",
]


def main():
    print(f"⏳ Loading {cfg.QWEN_MODEL} on {cfg.DEVICE}...")
    tokenizer  = AutoTokenizer.from_pretrained(cfg.QWEN_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.QWEN_MODEL,
        device_map        = cfg.DEVICE,
        torch_dtype       = torch.float16,
        trust_remote_code = True,
    )
    base_model.config.use_cache = False
    print(f"✅ Base model loaded")

    results = {}
    for name in LORA_NAMES:
        status = "TRAINED" if is_trained(name) else "PENDING"
        print(f"\n  {name}: {status}")
        results[name] = train_one(name, base_model, tokenizer)

    print("\n" + "═"*50)
    print("  TRAINING SUMMARY")
    print("═"*50)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")


if __name__ == "__main__":
    main()
