"""
Central configuration — loaded once from .env
Import anywhere: from src.config import cfg
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── Paths ────────────────────────────────────────────────
    ROOT_DIR      = Path(__file__).parent.parent
    ADAPTER_DIR   = ROOT_DIR / os.getenv("ADAPTER_DIR",   "adapters")
    DATA_DIR      = ROOT_DIR / os.getenv("DATA_DIR",      "data")
    METRICS_DIR   = ROOT_DIR / os.getenv("METRICS_DIR",   "metrics")
    CHECKPOINT_DIR= ROOT_DIR / os.getenv("CHECKPOINT_DIR","checkpoints")
    LORA_DATA_DIR = DATA_DIR / "lora_datasets"
    EXTRACTED_DIR = DATA_DIR / "extracted_cvs"
    RAW_CV_DIR    = DATA_DIR / "raw_cvs"

    # ── Model ────────────────────────────────────────────────
    QWEN_MODEL     = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    DEVICE         = os.getenv("DEVICE", "cuda")
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 1024))

    # ── API Keys ─────────────────────────────────────────────
    GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
    HF_TOKEN      = os.getenv("HF_TOKEN", "")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "cv-lora-training")

    # ── LoRA ─────────────────────────────────────────────────
    CLASSIFICATION_LORAS = {"lora1_classification"}

    LORA_INSTRUCTIONS = {
        "lora1_classification": "Classify this resume into the appropriate job category based on the candidate's experience and skills.",
        "lora2_skills":         "Extract and categorize all skills from this resume.",
        "lora3_interview":      "Generate interview questions and ideal answers for this candidate.",
        "lora4_improvement":    "Review this resume and provide specific improvement suggestions.",
    }

    TASK_MAP = {
        "classify" : "lora1_classification",
        "skills"   : "lora2_skills",
        "interview": "lora3_interview",
        "improve"  : "lora4_improvement",
    }

    VALID_LABELS = [
        "INFORMATION-TECHNOLOGY", "ENGINEERING", "FITNESS", "FINANCE",
        "AVIATION", "CHEF", "ADVOCATE", "ACCOUNTANT", "BANKING",
        "CONSULTANT", "SALES", "PUBLIC-RELATIONS", "HEALTHCARE",
        "BUSINESS-DEVELOPMENT", "HR", "ARTS", "CONSTRUCTION", "DESIGNER",
        "DIGITAL-MEDIA", "TEACHER", "APPAREL", "AGRICULTURE",
        "AUTOMOBILE", "BPO",
    ]

    KEYWORD_MAP = {
        "INFORMATION-TECHNOLOGY": ["DATA SCIENTIST","MACHINE LEARNING","AI ENGINEER",
            "SOFTWARE","DEVELOPER","PYTHON","NLP","DEEP LEARNING",
            "COMPUTER VISION","DATA ENGINEER","DEVOPS","CLOUD","BACKEND","FRONTEND"],
        "ENGINEERING":    ["MECHANICAL","CIVIL","ELECTRICAL","STRUCTURAL","CHEMICAL","INDUSTRIAL"],
        "FINANCE":        ["FINANCIAL","INVESTMENT","EQUITY","TRADING","PORTFOLIO","CFA"],
        "HEALTHCARE":     ["DOCTOR","NURSE","MEDICAL","CLINICAL","PHARMACY","HOSPITAL"],
        "BANKING":        ["BANK","CREDIT","LOAN","TELLER","MORTGAGE"],
        "SALES":          ["SALES","REVENUE","ACCOUNT EXECUTIVE"],
        "HR":             ["HUMAN RESOURCES","RECRUITMENT","TALENT","PAYROLL"],
        "DESIGNER":       ["GRAPHIC DESIGN","UI/UX","ADOBE","FIGMA","ILLUSTRATOR"],
        "DIGITAL-MEDIA":  ["SOCIAL MEDIA","CONTENT","SEO","MARKETING","DIGITAL"],
        "TEACHER":        ["TEACHER","PROFESSOR","INSTRUCTOR","TUTOR"],
        "ACCOUNTANT":     ["ACCOUNTANT","AUDIT","TAX","BOOKKEEPING","CPA"],
        "CONSULTANT":     ["CONSULTANT","ADVISORY","STRATEGY","MANAGEMENT CONSULTING"],
        "CHEF":           ["CHEF","COOK","CULINARY","KITCHEN","RESTAURANT"],
        "FITNESS":        ["FITNESS","TRAINER","GYM","NUTRITION","WELLNESS"],
        "AVIATION":       ["PILOT","AVIATION","AIRCRAFT","AIRLINE","FLIGHT"],
        "CONSTRUCTION":   ["CONSTRUCTION","ARCHITECTURE","SITE ENGINEER","CONTRACTOR"],
        "AGRICULTURE":    ["AGRICULTURE","FARMING","CROP","SOIL","IRRIGATION"],
        "AUTOMOBILE":     ["AUTOMOBILE","AUTOMOTIVE","VEHICLE","MECHANIC","CAR"],
        "ADVOCATE":       ["LAWYER","ADVOCATE","LEGAL","ATTORNEY","LAW"],
        "APPAREL":        ["FASHION","APPAREL","TEXTILE","CLOTHING","GARMENT"],
        "PUBLIC-RELATIONS":     ["PUBLIC RELATIONS","PR","COMMUNICATIONS","MEDIA RELATIONS"],
        "BUSINESS-DEVELOPMENT": ["BUSINESS DEVELOPMENT","PARTNERSHIPS","B2B","GROWTH"],
        "BPO":            ["BPO","CALL CENTER","CUSTOMER SERVICE","OUTSOURCING"],
    }

    # ── Training ─────────────────────────────────────────────
    LORA_TRAIN_CONFIGS = {
        "lora1_classification": {"num_train_epochs": 3, "learning_rate": 2e-4},
        "lora2_skills":         {"num_train_epochs": 3, "learning_rate": 5e-5},
        "lora3_interview":      {"num_train_epochs": 3, "learning_rate": 5e-5},
        "lora4_improvement":    {"num_train_epochs": 3, "learning_rate": 5e-5},
    }

    # ── Server ───────────────────────────────────────────────
    PORT = int(os.getenv("PORT", 8000))

    def ensure_dirs(self):
        for d in [self.ADAPTER_DIR, self.DATA_DIR, self.METRICS_DIR,
                  self.CHECKPOINT_DIR, self.LORA_DATA_DIR,
                  self.EXTRACTED_DIR, self.RAW_CV_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()
