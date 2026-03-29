"""Evaluation metrics for classification and generation LoRAs."""
import json
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer as rouge_lib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
)
import torch
from peft import PeftModel
from src.config import cfg
from src.data.dataset import load_json_raw
from src.agent.inference import generate_prediction


def compute_classification_metrics(preds: list, labels: list, name: str) -> dict:
    acc       = accuracy_score(labels, preds)
    f1_macro  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weight = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted",    zero_division=0)
    report    = classification_report(labels, preds, zero_division=0)

    metrics = {
        "accuracy"          : round(acc,       4),
        "f1_macro"          : round(f1_macro,  4),
        "f1_weighted"       : round(f1_weight, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted"   : round(recall,    4),
    }
    print(f"\n  📊 {name} — Classification Metrics:")
    print(f"     Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"     F1 macro  : {f1_macro:.4f}")
    print(f"     F1 weight : {f1_weight:.4f}")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"\n  Per-class report:\n{report}")
    return metrics


def compute_generation_metrics(preds: list, labels: list, name: str) -> dict:
    scorer   = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    exact = 0
    for pred, label in zip(preds, labels):
        s = scorer.score(str(label), str(pred))
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
        if str(pred).strip().lower() == str(label).strip().lower():
            exact += 1

    metrics = {
        "rouge1"     : round(float(np.mean(r1)), 4),
        "rouge2"     : round(float(np.mean(r2)), 4),
        "rougeL"     : round(float(np.mean(rL)), 4),
        "exact_match": round(exact / len(preds), 4),
    }
    print(f"\n  📊 {name} — Generation Metrics:")
    print(f"     ROUGE-1     : {metrics['rouge1']:.4f}")
    print(f"     ROUGE-2     : {metrics['rouge2']:.4f}")
    print(f"     ROUGE-L     : {metrics['rougeL']:.4f}")
    print(f"     Exact Match : {metrics['exact_match']:.4f}  ({exact}/{len(preds)})")
    return metrics


def evaluate_adapter(name: str, adapter_path: str, base_model, tokenizer) -> dict:
    print(f"\n  🔍 Evaluating {name}...")
    eval_model = PeftModel.from_pretrained(base_model, adapter_path)
    eval_model = eval_model.to(cfg.DEVICE)
    eval_model.eval()
    all_metrics = {}

    for split in ["val", "test"]:
        raw_data = load_json_raw(name, split)
        if not raw_data:
            print(f"     ⚠️  No {split} data found")
            continue

        print(f"     Running {split} inference on {len(raw_data)} samples...")
        preds, labels = [], []

        for sample in tqdm(raw_data, desc=f"     {split}", leave=False):
            pred = generate_prediction(
                eval_model, tokenizer,
                str(sample.get("instruction", "")),
                str(sample.get("input", "")),
                max_new_tokens=32 if name in cfg.CLASSIFICATION_LORAS else 256,
            )
            preds.append(pred)
            labels.append(str(sample.get("output", "")))

        if name in cfg.CLASSIFICATION_LORAS:
            known   = list(set(labels))
            normed  = [
                next((l for l in known if l.lower() in p.strip().lower()
                      or p.strip().lower() in l.lower()), p.strip())
                for p in preds
            ]
            split_metrics = compute_classification_metrics(normed, labels, f"{name}/{split}")
        else:
            split_metrics = compute_generation_metrics(preds, labels, f"{name}/{split}")

        all_metrics[split] = split_metrics

    metrics_path = cfg.METRICS_DIR / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"     💾 Metrics saved → {metrics_path}")

    del eval_model
    torch.cuda.empty_cache()
    return all_metrics
