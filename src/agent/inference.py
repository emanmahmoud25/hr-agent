# """Core inference functions — used by agent and evaluator."""
# import torch
# from src.config import cfg


# def generate_prediction(model, tokenizer, instruction, input_text, max_new_tokens=128):
#     input_text = input_text[:1500].strip()

#     prompt = (
#         f"### Instruction:\n{instruction}\n\n"
#         f"### Input:\n{input_text}\n\n"
#         f"### Response:\n"
#     )

#     inputs = tokenizer(
#         prompt,
#         return_tensors='pt',
#         truncation=True,
#         max_length=MAX_SEQ_LENGTH
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             repetition_penalty=1.3,
#             no_repeat_ngram_size=4,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded.split('### Response:')[-1].strip()


# def predict_classification(cv_text):
#     raw = generate_prediction(
#             agent.active_model,
#             tokenizer,
#             LORA_INSTRUCTIONS['lora1_classification'],
#             cv_text[:600],
#             max_new_tokens=10
#         )

#     first = raw.strip().split()[0].upper().rstrip(".,;:") if raw.strip() else ""

#     if first in VALID_LABELS:
#         return first

#     return "INFORMATION-TECHNOLOGY"

#     # Layer 3: keyword score scan
#     cv_upper = cv_text.upper()
#     scores   = {lbl: sum(1 for kw in kws if kw in cv_upper)
#                 for lbl, kws in cfg.KEYWORD_MAP.items()}
#     best     = max(scores, key=scores.get)
#     if scores[best] > 0:
#         return best

#     return "INFORMATION-TECHNOLOGY"
"""Full Groq inference — no local model needed."""
from src.config import cfg

GROQ_INSTRUCTIONS = {
    "lora1_classification": """
You are an expert HR assistant.
Analyze the CV and infer the most suitable industry or career field 
for this candidate.

Respond with only one concise category label (2–4 words max).
No explanation.
""",

    "lora2_skills": """You are a CV analyzer. Extract ONLY the skills explicitly
mentioned in this CV. Organize as:
Technical Skills: ...
Soft Skills: ...
Do NOT invent or add anything not written in the CV.""",

    "lora3_interview": """You are an HR interviewer. Generate 5 interview questions
with ideal answers based on this CV.
Format:
Q1: ...
A1: ...
Q2: ...
A2: ...""",

    "lora4_improvement": """You are a CV coach. Give 5 specific improvement suggestions
referencing actual content from this CV.
Do NOT invent experience or skills not in the CV.""",
}
def groq_call(lora_name: str, cv_text: str) -> str:
    from groq import Groq
    client = Groq(api_key=cfg.GROQ_API_KEY)
    resp   = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [
            {"role": "system", "content": GROQ_INSTRUCTIONS[lora_name]},
            {"role": "user",   "content": cv_text[:3000]},
        ],
        temperature = 0.3,
        max_tokens  = 512,
    )
    return resp.choices[0].message.content.strip()


def classify_with_fallback(cv_text: str) -> str:
    """Groq classification + keyword fallback if invalid label."""
    raw   = groq_call("lora1_classification", cv_text)
    first = raw.strip().split()[0].upper().rstrip(".,;:")

    # Exact match
    if first in cfg.VALID_LABELS:
        return first

    # Keyword fallback
    cv_upper = cv_text.upper()
    scores   = {lbl: sum(1 for kw in kws if kw in cv_upper)
                for lbl, kws in cfg.KEYWORD_MAP.items()}
    best     = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "INFORMATION-TECHNOLOGY"


# keep these for backward compatibility with evaluator
def generate_prediction(model, tokenizer, instruction, input_text, max_new_tokens=128):
    raise NotImplementedError("Full Groq mode — no local model inference")

def predict_classification(model, tokenizer, cv_text):
    raise NotImplementedError("Full Groq mode — use classify_with_fallback()")