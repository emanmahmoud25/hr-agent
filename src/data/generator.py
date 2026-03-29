"""Groq API data generation for LoRA 2/3/4."""
import json, time
from pathlib import Path
from typing import Optional
from groq import Groq
from src.config import cfg

GROQ_MODEL          = "llama3-70b-8192"
SLEEP_BETWEEN_CVS   = 62
CHECKPOINT_EVERY    = 5

_client: Optional[Groq] = None

def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.GROQ_API_KEY)
    return _client


def call_groq_safe(system_prompt: str, user_prompt: str) -> Optional[str]:
    delays = [30, 60, 90, 120]
    for attempt, wait in enumerate(delays):
        try:
            resp = get_client().chat.completions.create(
                model          = GROQ_MODEL,
                messages       = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature    = 0.7,
                max_tokens     = 1024,
                response_format= {"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                print(f"Rate limit — waiting {wait}s (attempt {attempt+1}/{len(delays)})...")
                time.sleep(wait)
            else:
                print(f"API error: {err}")
                time.sleep(10)
    print("All retries failed for this CV, skipping.")
    return None


def safe_run(name: str, cv_list: list, generate_fn) -> list:
    ckpt_path = cfg.CHECKPOINT_DIR / f"{name}.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
        done = {r["_src"] for r in data if "_src" in r}
        print(f"Resuming {name}: {len(data)} done, {len(cv_list)-len(done)} remaining")
    else:
        data, done = [], set()
        print(f"Starting {name}: {len(cv_list)} CVs to process")

    remaining = [cv for cv in cv_list if cv["filename"] not in done]
    errors    = 0

    for i, cv in enumerate(remaining):
        rec = generate_fn(cv)
        if rec:
            rec["_src"] = cv["filename"]
            data.append(rec)
        else:
            errors += 1
        print(f"  [{i+1}/{len(remaining)}] {cv['filename'][:40]} — saved: {len(data)}")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"  Checkpoint saved ({len(data)} records, {errors} errors)")
        if i < len(remaining) - 1:
            time.sleep(SLEEP_BETWEEN_CVS)

    ckpt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"{name} done — {len(data)} records | {errors} errors")
    return data
