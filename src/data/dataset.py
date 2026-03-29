"""Build train/val/test splits and HuggingFace Dataset objects."""
import json, random
from pathlib import Path
from datasets import Dataset
from src.config import cfg

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1


def build_splits(data: list, name: str) -> tuple[list, list, list]:
    """Shuffle and split data into train/val/test."""
    random.shuffle(data)
    n       = len(data)
    n_val   = max(1, int(n * VAL_RATIO))
    n_test  = max(1, int(n * TEST_RATIO))
    n_train = n - n_val - n_test
    train, val, test = data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]

    for split, split_data in [("train", train), ("val", val), ("test", test)]:
        path = cfg.LORA_DATA_DIR / f"{name}_{split}.json"
        path.write_text(json.dumps(split_data, ensure_ascii=False, indent=2))

    print(f"{name}: {len(train)} train | {len(val)} val | {len(test)} test")
    return train, val, test


def format_prompt(example: dict) -> dict:
    return {
        "text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    }


def load_json_raw(name: str, split: str) -> list:
    path = cfg.LORA_DATA_DIR / f"{name}_{split}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_json_dataset(name: str, split: str) -> Dataset | None:
    raw = load_json_raw(name, split)
    if not raw:
        return None
    clean = [
        {
            "instruction": str(r.get("instruction", "")),
            "input":       str(r.get("input", "")),
            "output":      str(r.get("output", "")),
        }
        for r in raw
    ]
    return Dataset.from_list(clean).map(format_prompt)


def build_lora1_record(cv: dict) -> dict:
    snippet = cv["text"][:600].replace("\n", " ").strip()
    return {
        "instruction": cfg.LORA_INSTRUCTIONS["lora1_classification"],
        "input":       snippet,
        "output":      cv["position"],
    }
