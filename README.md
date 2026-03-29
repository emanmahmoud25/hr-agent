# 🤖 Multi-Task Resume Analyzer — HR Agent

> **Generative AI system using Qwen2.5-3B-Instruct + 4 task-specific LoRA adapters for intelligent CV 

## 📌 Project Overview

The **HR Agent** is a multi-task AI system that automates resume analysis using **Parameter-Efficient Fine-Tuning (PEFT/LoRA)** on a large language model. Instead of training one general model for everything, we train **4 specialized LoRA adapters**, each focused on a single HR task. At inference time, the system **dynamically switches** between adapters — loading only the needed one on top of the frozen base model — making it both memory-efficient and highly accurate per task.

### What problem does it solve?

HR teams spend significant time manually reading, categorizing, and evaluating hundreds of CVs. This system automates that process end-to-end:

- **No manual reading** — AI extracts and analyzes CVs in seconds
- **Task-specific accuracy** — each LoRA is specialized for its task
- **Scalable** — handles batch ranking of multiple CVs
- **Deployable** — ships as a FastAPI web service with a web UI

---

## 🧠 How It Works — Full Pipeline

```
Raw PDF CVs
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1 — Data Preparation                                   │
│  • Extract text from PDFs using PyMuPDF                      │
│  • Generate training data via Groq API (LLaMA-3-70B)         │
│  • Split into Train (80%) / Val (10%) / Test (10%)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 — Fine-Tuning (4 LoRA Adapters)                      │
│                                                              │
│  Base Model: Qwen2.5-3B-Instruct (frozen weights)            │
│                                                              │
│  LoRA 1 → Classification   (job category labels)            │
│  LoRA 2 → Skills           (structured skills extraction)   │
│  LoRA 3 → Interview        (Q&A generation)                 │
│  LoRA 4 → Improvement      (resume enhancement tips)        │
│                                                              │
│  Each adapter: r=16, alpha=32, ~1.2GB on disk                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 — Dynamic LoRA Agent                                 │
│                                                              │
│  Base model loaded once in VRAM                             │
│  ┌──────────────┐                                           │
│  │  base_model  │ ← stays in memory always                  │
│  └──────┬───────┘                                           │
│         │  switch on demand                                  │
│    ┌────┴────┬─────────┬──────────┐                         │
│  LoRA1    LoRA2     LoRA3      LoRA4                         │
│  load→    load→     load→      load→                         │
│  infer→   infer→    infer→     infer→                        │
│  unload   unload    unload     unload                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 — FastAPI Service                                    │
│  • Web UI for single CV upload                               │
│  • REST API for programmatic access                          │
│  • CV Ranking pipeline (multi-CV scoring)                    │
│  • Real-time LoRA switching with status monitoring           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source:** [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

| Property | Value |
|---|---|
| Total resumes | 2,400+ |
| Job categories | 25 fields (IT, Engineering, Finance, etc.) |
| Format | PDF + CSV |
| Languages | English |

**Data splits per LoRA:**

| Split | Ratio | Purpose |
|---|---|---|
| Train | 80% | Model learning |
| Validation | 10% | Monitor overfitting during training |
| Test | 10% | Final evaluation after training |

**Training data generation:**  
For LoRAs 2, 3, and 4, training samples were synthetically generated using **Groq API (LLaMA-3-70B)** with `temperature=0.7` — giving diverse, high-quality HR-specific outputs from raw CV text. LoRA 1 used the raw position labels directly.

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit NF4 (GPU) / float16 (CPU) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch size | 2 (grad accumulation ×8 = effective 16) |
| Epochs | 3 |
| Scheduler | Cosine with 5% warmup |
| Max seq length | 1024 tokens |
| Optimizer | AdamW |
| Experiment tracking | WandB |

**Learning rates per task:**

| LoRA | Task | Learning Rate |
|---|---|---|
| lora1 | Classification | 2e-4 |
| lora2 | Skills | 5e-5 |
| lora3 | Interview | 5e-5 |
| lora4 | Improvement | 5e-5 |

---

## 📈 Results

| LoRA | Task | Metric | Score |
|---|---|---|---|
| lora1_classification | Job Classification | Accuracy | **80.0%** |
| lora1_classification | Job Classification | F1 (weighted) | **0.78** |
| lora2_skills | Skills Extraction | ROUGE-1 | **0.329** |
| lora3_interview | Interview Q&A | ROUGE-1 | **0.491** |
| lora4_improvement | CV Improvement | ROUGE-1 | **0.456** |

> Exact Match = 0.0 for generation tasks is expected — free-text generation never matches word-for-word. ROUGE-1 > 0.33 on a 3B model is solid for this domain.

---

## 🗂️ Project Structure

```
HR_AGENT/
│
├── 📁 data/
│   ├── raw_cvs/                    # Original PDF/TXT resumes
│   ├── extracted_cvs/              # Cleaned plain text files
│   └── lora_datasets/              # Train/Val/Test JSON splits
│       ├── lora1_classification_train.json
│       ├── lora1_classification_val.json
│       ├── lora1_classification_test.json
│       └── ...
│
├── 📁 adapters/                    # Trained LoRA weights (copy from Drive/Colab)
│   ├── lora1_classification/
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   ├── lora2_skills/
│   ├── lora3_interview/
│   └── lora4_improvement/
│
├── 📁 metrics/                     # Evaluation results per LoRA
│   ├── lora1_classification_metrics.json
│   └── ...
│
├── 📁 checkpoints/                 # Groq generation checkpoints (resume support)
│
├── 📁 src/
│   ├── config.py                   # Central config loaded from .env
│   │
│   ├── 📁 data/                    # Data pipeline
│   │   ├── __init__.py
│   │   ├── extractor.py            # PDF/TXT → clean text (PyMuPDF)
│   │   ├── generator.py            # Groq API → training samples
│   │   └── dataset.py              # Train/val/test splits + format_prompt
│   │
│   ├── 📁 training/                # Fine-tuning pipeline
│   │   ├── __init__.py
│   │   ├── config.py               # LoRA hyperparameters per task
│   │   ├── trainer.py              # train_one() — SFTTrainer wrapper
│   │   ├── callbacks.py            # ProgressBarCallback (tqdm %)
│   │   └── evaluate.py             # Accuracy/F1/ROUGE metrics
│   │
│   ├── 📁 agent/                   # HR Agent core
│   │   ├── __init__.py
│   │   ├── dynamic_lora.py         # DynamicLoRAAgent — load/switch/unload
│   │   └── inference.py            # generate_prediction()
│   │
│   └── 📁 api/                     # FastAPI service
│       ├── __init__.py
│       ├── app.py                  # FastAPI app + lifespan + middleware
│       ├── routes.py               # /cv/upload, /cv/upload/full, /status
│       ├── routes_rank.py          # /cv/rank — multi-CV ranking pipeline
│       ├── models.py               # Pydantic request/response schemas
│       └── 📁 ui/
│           └── index.html          # Web UI (vanilla JS, no framework)
│
├── 📁 scripts/                     # Standalone run scripts
│   ├── extract_cvs.py              # Run PDF extraction pipeline
│   ├── train_all.py                # Train all 4 LoRAs sequentially
│   └── evaluate_all.py             # Evaluate all adapters on test set
│
├── 📁 tests/
│   ├── test_agent.py               # Unit tests for DynamicLoRAAgent
│   └── test_api.py                 # API endpoint tests
│
├── .env                            # API keys — NEVER commit
├── .env.example                    # Template for .env
├── .gitignore
├── main.py                         # Entry point → starts FastAPI
├── main_proxy.py                   # Colab proxy version (eval_js)
├── Comparison.py                   # CV ranking/comparison utilities
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/hr-agent.git
cd hr-agent

python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
WANDB_API_KEY=...
WANDB_PROJECT=cv-lora-training
QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
ADAPTER_DIR=./adapters
DATA_DIR=./data
METRICS_DIR=./metrics
```

### 4. Add Trained Adapters

Copy your trained LoRA adapters (from Google Drive / Colab) into `./adapters/`:

```
adapters/
├── lora1_classification/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── lora2_skills/
├── lora3_interview/
└── lora4_improvement/
```

### 5. Run the API

```bash
python main.py
```

| URL | Description |
|---|---|
| http://localhost:8000 | Web UI |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/status | Agent status |

---

## 🔌 API Reference

### Single Task Analysis

```http
POST /cv/upload?task=classify
Content-Type: multipart/form-data

file: <CV PDF or TXT>
```

**Tasks:** `classify` · `skills` · `interview` · `improve`

**Response:**
```json
{
  "task": "classify",
  "result": "INFORMATION-TECHNOLOGY",
  "lora_used": "lora1_classification",
  "elapsed_s": 4.2
}
```

### Full Pipeline (All 4 Tasks)

```http
POST /cv/upload/full
Content-Type: multipart/form-data

file: <CV PDF or TXT>
```

**Response:**
```json
{
  "classify":  "INFORMATION-TECHNOLOGY",
  "skills":    "Python, Machine Learning, SQL, Docker...",
  "interview": "Q1: Describe your experience with...",
  "improve":   "1. Add quantified achievements...",
  "elapsed_s": 18.5
}
```

### CV Ranking

```http
POST /cv/rank
Content-Type: multipart/form-data

files: [cv1.pdf, cv2.pdf, cv3.pdf]
job_description: "Senior ML Engineer with 5+ years..."
```

### Agent Status

```http
GET /status
```

```json
{
  "active_lora": "lora2_skills",
  "switch_count": 3,
  "call_count": 12,
  "device": "cuda:0"
}
```

---

## 🏃 Training from Scratch

If you want to retrain the adapters on your own data:

```bash
# Step 1 — Extract text from raw PDFs
python scripts/extract_cvs.py

# Step 2 — Generate training data via Groq API
#           (LoRAs 2, 3, 4 — classification uses labels directly)
# Edit scripts/generate_data.py with your Groq key then run it

# Step 3 — Train all 4 LoRAs (requires GPU)
python scripts/train_all.py

# Step 4 — Evaluate all adapters
python scripts/evaluate_all.py
```

> **Tip:** Training was done on Google Colab T4 GPU (~28 min per LoRA). For CPU training expect 6–10 hours per LoRA.

---

## 💻 System Requirements

| Scenario | Requirement |
|---|---|
| Inference only (run API) | 6GB+ VRAM (GTX 1060 / RTX 3060) |
| Training new LoRAs | 16GB+ VRAM (RTX 3090 / A100) |
| CPU-only inference | Works but slow (~2–5 min per CV) |
| Disk space | ~11GB (base model ~6GB + adapters ~5GB) |
| Python | 3.10+ |

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `transformers` | 4.47.0 | Qwen model + tokenizer |
| `peft` | ≥ 0.7.0 | LoRA fine-tuning |
| `trl` | 0.12.0 | SFTTrainer |
| `torch` | ≥ 2.0 | Model training & inference |
| `fastapi` | ≥ 0.104 | REST API server |
| `pymupdf` | latest | PDF text extraction |
| `groq` | ≥ 0.4.0 | Training data generation |
| `scikit-learn` | latest | Accuracy, F1, precision, recall |
| `rouge-score` | latest | ROUGE metrics for generation |
| `wandb` | latest | Training experiment tracking |
| `bitsandbytes` | ≥ 0.43 | 4-bit quantization (GPU) |

---

## 🔬 Technical Highlights

- **Dynamic LoRA Switching** — base model loaded once; adapters load/unload on demand → saves VRAM vs. loading 4 separate models
- **Parameter-Efficient Fine-Tuning** — only ~1% of weights trained per adapter (r=16 LoRA vs full fine-tuning)
- **Greedy Decoding** (`do_sample=False`) — deterministic outputs; same CV always gives same result
- **Checkpoint Resume** — training automatically resumes from last checkpoint if interrupted
- **Groq Data Generation** — `temperature=0.7` for diverse but coherent training samples
- **Normalization** — classification predictions normalized against known label set for robustness

---

## 📁 .gitignore (recommended)

```
adapters/
data/raw_cvs/
data/extracted_cvs/
checkpoints/
metrics/
.env
__pycache__/
*.pyc
venv/
*.egg-info/
.DS_Store
wandb/
```

---

## 🙏 Acknowledgements

- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anbhawal
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba Cloud
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
- [Groq](https://groq.com) for fast LLaMA-3 inference used in data generation

