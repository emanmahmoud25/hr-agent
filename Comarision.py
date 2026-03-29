# """
# Evaluate Groq vs LoRA on test data.
# Usage: python scripts/evaluate_groq_vs_lora.py
# Results saved to: metrics/groq_vs_lora_metrics.json
# """
# import json
# import time
# import numpy as np
# from pathlib import Path
# from groq import Groq
# from sklearn.metrics import accuracy_score, f1_score
# from rouge_score import rouge_scorer
# from src.config import cfg

# # ── Setup ─────────────────────────────────────────────────────
# groq_client = Groq(api_key=cfg.GROQ_API_KEY)
# ROUGE       = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
# SLEEP       = 3   # seconds between Groq calls to avoid rate limit

# GROQ_EVAL_INSTRUCTIONS = {
#     "lora1_classification": """Classify this CV into EXACTLY ONE of these categories.
# Reply with the label only — no explanation:
# INFORMATION-TECHNOLOGY, ENGINEERING, FITNESS, FINANCE, AVIATION, CHEF, ADVOCATE,
# ACCOUNTANT, BANKING, CONSULTANT, SALES, PUBLIC-RELATIONS, HEALTHCARE,
# BUSINESS-DEVELOPMENT, HR, ARTS, CONSTRUCTION, DESIGNER, DIGITAL-MEDIA,
# TEACHER, APPAREL, AGRICULTURE, AUTOMOBILE, BPO""",

#     "lora2_skills"        : "Extract and categorize all skills from this resume.",
#     "lora3_interview"     : "Generate interview questions and ideal answers for this candidate.",
#     "lora4_improvement"   : "Review this resume and provide specific improvement suggestions.",
# }

# LORA_NAMES = [
#     "lora1_classification",
#     "lora2_skills",
#     "lora3_interview",
#     "lora4_improvement",
# ]


# # ── Load test data ────────────────────────────────────────────
# def load_test(name: str) -> list:
#     path = cfg.LORA_DATA_DIR / f"{name}_test.json"
#     if not path.exists():
#         print(f"  ⚠️  Not found: {path}")
#         return []
#     data = json.loads(path.read_text())
#     print(f"  ✅ {name}_test.json — {len(data)} samples")
#     return data


# # ── Groq prediction ───────────────────────────────────────────
# def groq_predict(lora_name: str, input_text: str) -> str:
#     for attempt in range(3):
#         try:
#             resp = groq_client.chat.completions.create(
#                 model    = "llama-3.3-70b-versatile",
#                 messages = [
#                     {"role": "system", "content": GROQ_EVAL_INSTRUCTIONS[lora_name]},
#                     {"role": "user",   "content": input_text[:2000]},
#                 ],
#                 temperature = 0.0,
#                 max_tokens  = 256,
#             )
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             wait = 30 * (attempt + 1)
#             print(f"\n  ⚠️  Groq error (attempt {attempt+1}): {e} — waiting {wait}s")
#             time.sleep(wait)
#     return "ERROR"


# # ── LoRA prediction (from saved adapter) ─────────────────────
# def lora_predict(lora_name: str, input_text: str,
#                  base_model, tokenizer) -> str:
#     from peft import PeftModel
#     import torch

#     adapter_path = cfg.ADAPTER_DIR / lora_name
#     if not adapter_path.exists():
#         return "ADAPTER_NOT_FOUND"

#     model = PeftModel.from_pretrained(base_model, str(adapter_path))
#     model = model.to(cfg.DEVICE)
#     model.eval()

#     from src.agent.inference import generate_prediction, classify_with_fallback

#     if lora_name == "lora1_classification":
#         # use keyword fallback
#         prompt = input_text
#         result = classify_with_fallback(prompt)
#     else:
#         result = generate_prediction(
#             model, tokenizer,
#             cfg.LORA_INSTRUCTIONS[lora_name],
#             input_text, max_new_tokens=256,
#         )

#     model.cpu()
#     del model
#     if cfg.DEVICE == "cuda":
#         torch.cuda.empty_cache()

#     return result


# # ── Metrics ───────────────────────────────────────────────────
# def norm_label(p: str) -> str:
#     first = p.strip().split()[0].upper().rstrip(".,;:") if p.strip() else ""
#     return first if first in cfg.VALID_LABELS else "UNKNOWN"


# def classification_metrics(preds, labels) -> dict:
#     acc = accuracy_score(labels, preds)
#     f1  = f1_score(labels, preds, average="weighted", zero_division=0)
#     return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


# def generation_metrics(preds, labels) -> dict:
#     r1, r2, rL = [], [], []
#     for p, l in zip(preds, labels):
#         s = ROUGE.score(str(l), str(p))
#         r1.append(s["rouge1"].fmeasure)
#         r2.append(s["rouge2"].fmeasure)
#         rL.append(s["rougeL"].fmeasure)
#     return {
#         "rouge1": round(float(np.mean(r1)), 4),
#         "rouge2": round(float(np.mean(r2)), 4),
#         "rougeL": round(float(np.mean(rL)), 4),
#     }


# # ── Main ──────────────────────────────────────────────────────
# def main():
#     print("═" * 55)
#     print("  Groq vs LoRA — Evaluation")
#     print("═" * 55)

#     # Load base model for LoRA evaluation
#     has_loras = any((cfg.ADAPTER_DIR / n).exists() for n in LORA_NAMES)
#     base_model, tokenizer = None, None

#     if has_loras:
#         print("\n⏳ Loading base model for LoRA evaluation...")
#         import torch
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#         tokenizer  = AutoTokenizer.from_pretrained(cfg.QWEN_MODEL, trust_remote_code=True)
#         tokenizer.pad_token = tokenizer.eos_token
#         base_model = AutoModelForCausalLM.from_pretrained(
#             cfg.QWEN_MODEL,
#             device_map        = cfg.DEVICE,
#             torch_dtype       = torch.float32,
#             low_cpu_mem_usage = True,
#             trust_remote_code = True,
#         )
#         base_model.config.use_cache = False
#         print(f"✅ Base model loaded on {cfg.DEVICE}")
#     else:
#         print("⚠️  No LoRA adapters found — will evaluate Groq only")

#     # Load test data
#     print("\n── Loading test data ──")
#     all_test_data = {n: load_test(n) for n in LORA_NAMES}

#     results = {}

#     for lora_name, samples in all_test_data.items():
#         if not samples:
#             print(f"\n⏭  Skipping {lora_name} — no test data")
#             continue

#         print(f"\n{'═'*55}")
#         print(f"  {lora_name}  ({len(samples)} samples)")
#         print(f"{'═'*55}")

#         groq_preds, lora_preds, labels = [], [], []

#         for i, sample in enumerate(samples):
#             input_text = str(sample.get("input", ""))
#             label      = str(sample.get("output", ""))

#             # Groq
#             g = groq_predict(lora_name, input_text)
#             groq_preds.append(g)

#             # LoRA
#             if base_model:
#                 l = lora_predict(lora_name, input_text, base_model, tokenizer)
#                 lora_preds.append(l)

#             labels.append(label)
#             print(f"  [{i+1}/{len(samples)}]  Groq: {g[:40]:<40}", end="\r")
#             time.sleep(SLEEP)

#         print()

#         # Compute metrics
#         if lora_name == "lora1_classification":
#             g_norm = [norm_label(p) for p in groq_preds]
#             l_norm = [norm_label(p) for p in lora_preds] if lora_preds else []
#             t_norm = [norm_label(l) for l in labels]
#             groq_m = classification_metrics(g_norm, t_norm)
#             lora_m = classification_metrics(l_norm, t_norm) if l_norm else {}
#         else:
#             groq_m = generation_metrics(groq_preds, labels)
#             lora_m = generation_metrics(lora_preds, labels) if lora_preds else {}

#         results[lora_name] = {"groq": groq_m, "lora": lora_m}

#         # Print comparison table
#         print(f"\n  {'Metric':<15} {'Groq':>10} {'LoRA':>10} {'Winner':>10} {'Diff':>8}")
#         print(f"  {'─'*55}")
#         for metric in groq_m:
#             g = groq_m[metric]
#             l = lora_m.get(metric, 0)
#             winner = "LoRA ✅" if l > g else "Groq ✅" if g > l else "Tie   "
#             diff   = round(l - g, 4)
#             sign   = "+" if diff >= 0 else ""
#             print(f"  {metric:<15} {g:>10.4f} {l:>10.4f} {winner:>10} {sign}{diff:>6}")

#     # ── Final summary ─────────────────────────────────────────
#     print(f"\n\n{'═'*55}")
#     print("  FINAL SUMMARY")
#     print(f"{'═'*55}")
#     for lora_name, r in results.items():
#         print(f"\n  {lora_name}")
#         for metric in r["groq"]:
#             g = r["groq"][metric]
#             l = r["lora"].get(metric, 0)
#             winner = "LoRA ✅" if l > g else "Groq ✅" if g > l else "Tie  "
#             diff   = round(l - g, 4)
#             sign   = "+" if diff >= 0 else ""
#             print(f"    {metric:<15} Groq={g:.4f}  LoRA={l:.4f}  {winner}  ({sign}{diff})")

#     # Save results
#     out = cfg.METRICS_DIR / "groq_vs_lora_metrics.json"
#     out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
#     print(f"\n💾 Results saved → {out}")


# if __name__ == "__main__":
#     main()

"""
Evaluate Groq on test data (no LoRA / no model loading needed).
Usage: python scripts/evaluate_groq_vs_lora.py
Results saved to: metrics/groq_eval_metrics.json
"""
import json
import time
import numpy as np
from groq import Groq
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from src.config import cfg

groq_client = Groq(api_key=cfg.GROQ_API_KEY)
ROUGE       = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
SLEEP       = 3

GROQ_INSTRUCTIONS = {
    "lora1_classification": """Classify this CV into EXACTLY ONE of these categories.
Reply with the label only — no explanation:
INFORMATION-TECHNOLOGY, ENGINEERING, FITNESS, FINANCE, AVIATION, CHEF, ADVOCATE,
ACCOUNTANT, BANKING, CONSULTANT, SALES, PUBLIC-RELATIONS, HEALTHCARE,
BUSINESS-DEVELOPMENT, HR, ARTS, CONSTRUCTION, DESIGNER, DIGITAL-MEDIA,
TEACHER, APPAREL, AGRICULTURE, AUTOMOBILE, BPO""",

    "lora2_skills"        : "Extract and categorize all skills from this resume.",
    "lora3_interview"     : "Generate interview questions and ideal answers for this candidate.",
    "lora4_improvement"   : "Review this resume and provide specific improvement suggestions.",
}

LORA_NAMES = [
    "lora1_classification",
    "lora2_skills",
    "lora3_interview",
    "lora4_improvement",
]


def load_test(name: str) -> list:
    path = cfg.LORA_DATA_DIR / f"{name}_test.json"
    if not path.exists():
        print(f"  ⚠️  Not found: {path}")
        return []
    data = json.loads(path.read_text())
    print(f"  ✅ {name}_test.json — {len(data)} samples")
    return data


def groq_predict(lora_name: str, input_text: str) -> str:
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model    = "llama3-8b-8192",
                messages = [
                    {"role": "system", "content": GROQ_INSTRUCTIONS[lora_name]},
                    {"role": "user",   "content": input_text[:2000]},
                ],
                temperature = 0.0,
                max_tokens  = 256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"\n  ⚠️  Groq error (attempt {attempt+1}): {e} — waiting {wait}s")
            time.sleep(wait)
    return "ERROR"


def norm_label(p: str) -> str:
    first = p.strip().split()[0].upper().rstrip(".,;:") if p.strip() else ""
    return first if first in cfg.VALID_LABELS else "UNKNOWN"


def classification_metrics(preds, labels) -> dict:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def generation_metrics(preds, labels) -> dict:
    r1, r2, rL = [], [], []
    for p, l in zip(preds, labels):
        s = ROUGE.score(str(l), str(p))
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(float(np.mean(r1)), 4),
        "rouge2": round(float(np.mean(r2)), 4),
        "rougeL": round(float(np.mean(rL)), 4),
    }


def main():
    print("═" * 55)
    print("  Groq Evaluation on Test Data")
    print("═" * 55)

    print("\n── Loading test data ──")
    all_test_data = {n: load_test(n) for n in LORA_NAMES}

    results = {}

    for lora_name, samples in all_test_data.items():
        if not samples:
            print(f"\n⏭  Skipping {lora_name} — no test data")
            continue

        print(f"\n{'═'*55}")
        print(f"  {lora_name}  ({len(samples)} samples)")
        print(f"{'═'*55}")

        preds, labels = [], []

        for i, sample in enumerate(samples):
            input_text = str(sample.get("input", ""))
            label      = str(sample.get("output", ""))

            pred = groq_predict(lora_name, input_text)
            preds.append(pred)
            labels.append(label)

            print(f"  [{i+1}/{len(samples)}]  {pred[:50]}", end="\r")
            time.sleep(SLEEP)

        print()

        if lora_name == "lora1_classification":
            p_norm  = [norm_label(p) for p in preds]
            l_norm  = [norm_label(l) for l in labels]
            metrics = classification_metrics(p_norm, l_norm)
        else:
            metrics = generation_metrics(preds, labels)

        results[lora_name] = metrics

        print(f"\n  {'Metric':<15} {'Score':>10}")
        print(f"  {'─'*27}")
        for metric, score in metrics.items():
            print(f"  {metric:<15} {score:>10.4f}")

    # Final summary
    print(f"\n\n{'═'*55}")
    print("  FINAL SUMMARY — Groq on Test Data")
    print(f"{'═'*55}")
    for lora_name, metrics in results.items():
        print(f"\n  {lora_name}")
        for metric, score in metrics.items():
            bar = "█" * int(score * 20)
            print(f"    {metric:<15} {score:.4f}  {bar}")

    out = cfg.METRICS_DIR / "groq_eval_metrics.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n💾 Saved → {out}")


if __name__ == "__main__":
    main()