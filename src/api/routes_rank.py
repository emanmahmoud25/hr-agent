"""
CV Ranking Router  —  POST /cv/rank
بيقبل: عدة CVs (PDF/TXT) + JD اختياري
بيرجع: ranked list بـ score من 3 معايير
"""
import os, json, time, re, logging
import httpx
import fitz                          # pymupdf
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional

logger = logging.getLogger("routes_rank")
router = APIRouter()

# ── Text extraction ───────────────────────────────────────
def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        doc  = fitz.open(stream=data, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text[:3000].strip()
    elif filename.lower().endswith(".txt"):
        return data.decode("utf-8", errors="ignore")[:3000].strip()
    raise HTTPException(400, f"Unsupported file type: {filename}")


# ── Groq scoring ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

RANK_SYSTEM = """You are an expert HR evaluator. Analyze the given CV and score it on 3 criteria.

Return ONLY a valid JSON object — no explanation, no markdown, no extra text:
{"jd_match": <0-100>, "quality": <0-100>, "skills": <0-100>, "summary": "<one sentence about candidate>"}

Scoring guide:
- jd_match: How well this CV matches the provided Job Description. If no JD given, base it on general employability (do NOT default to 50).
- quality: CV structure, clarity, depth of experience, achievements mentioned.
- skills: Breadth and depth of technical + soft skills relative to the role.

Be strict and differentiate between candidates. Scores should vary meaningfully."""


async def score_with_groq(cv_text: str, jd_text: str = "") -> dict:
    if not GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY not set in .env")

    user_msg = ""
    if jd_text:
        user_msg += f"JOB DESCRIPTION:\n{jd_text[:800]}\n\n"
    user_msg += f"CV TO EVALUATE:\n{cv_text[:1500]}"

    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type" : "application/json",
            },
            json={
                "model"      : GROQ_MODEL,
                "temperature": 0.1,
                "max_tokens" : 256,
                "messages"   : [
                    {"role": "system", "content": RANK_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            },
        )

    if resp.status_code != 200:
        logger.error(f"Groq error {resp.status_code}: {resp.text[:300]}")
        raise HTTPException(502, f"Groq API error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    logger.info(f"Groq raw response: {raw[:200]}")

    # ── Parse JSON robustly ──────────────────────────────
    # strip markdown fences if any
    raw = re.sub(r"```json|```", "", raw).strip()
    # extract first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.error(f"No JSON found in Groq response: {raw}")
        raise ValueError(f"Could not parse Groq response: {raw[:100]}")

    parsed = json.loads(match.group(0))

    # validate and clamp values
    result = {
        "jd_match": max(0, min(100, int(parsed.get("jd_match", 50)))),
        "quality" : max(0, min(100, int(parsed.get("quality",  50)))),
        "skills"  : max(0, min(100, int(parsed.get("skills",   50)))),
        "summary" : str(parsed.get("summary", ""))[:200],
    }
    return result


# ── Weighted rank score ───────────────────────────────────
def compute_rank_score(scores: dict, has_jd: bool) -> float:
    if has_jd:
        return round(
            scores["jd_match"] * 0.40 +
            scores["quality"]  * 0.35 +
            scores["skills"]   * 0.25, 1
        )
    else:
        return round(
            scores["quality"] * 0.55 +
            scores["skills"]  * 0.45, 1
        )


# ════════════════════════════════════════════════════════
# ENDPOINT
# ════════════════════════════════════════════════════════
@router.post("/cv/rank")
async def rank_cvs(
    files : List[UploadFile] = File(...),
    jd    : Optional[str]    = Form(None),
):
    if len(files) < 2:
        raise HTTPException(400, "Upload at least 2 CVs to rank")

    has_jd  = bool(jd and jd.strip())
    jd_text = jd.strip() if has_jd else ""
    results = []
    t0      = time.time()

    for f in files:
        content = await f.read()

        # extract text
        try:
            cv_text = extract_text(f.filename, content)
        except HTTPException as e:
            results.append({"filename": f.filename, "error": e.detail, "score": 0, "rank": 0})
            continue

        if len(cv_text) < 30:
            results.append({"filename": f.filename, "error": "Could not extract text", "score": 0, "rank": 0})
            continue

        # score via Groq
        try:
            scores = await score_with_groq(cv_text, jd_text)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Scoring failed for {f.filename}: {e}")
            results.append({
                "filename": f.filename,
                "error"   : str(e)[:100],
                "jd_match": 0, "quality": 0, "skills": 0,
                "summary" : "Scoring failed",
                "score"   : 0, "rank": 0,
            })
            continue

        rank_score = compute_rank_score(scores, has_jd)
        results.append({
            "filename": f.filename,
            "jd_match": scores["jd_match"],
            "quality" : scores["quality"],
            "skills"  : scores["skills"],
            "summary" : scores["summary"],
            "score"   : rank_score,
        })

    # sort + assign rank
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "mode"     : "groq",
        "has_jd"   : has_jd,
        "count"    : len(results),
        "elapsed_s": round(time.time() - t0, 2),
        "results"  : results,
    }