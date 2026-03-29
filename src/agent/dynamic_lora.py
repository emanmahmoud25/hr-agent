"""DynamicLoRAAgent — base model stays in VRAM, LoRAs load/unload on demand."""
import time
import torch
from peft import PeftModel
from src.config import cfg
from src.agent.inference import generate_prediction, predict_classification


class DynamicLoRAAgent:
    def __init__(self, base_model, tokenizer):
        self.base_model    = base_model
        self.tokenizer     = tokenizer
        self.current_lora  = None
        self.active_model  = None
        self._switch_count = 0
        self._call_count   = 0
        print(f"🤖 HR Agent ready  (device: {cfg.DEVICE})")
        print(f"   LoRAs: {list(cfg.LORA_INSTRUCTIONS.keys())}")

    # ── Internal load/unload ──────────────────────────────────
    def _get_clean_base(self):
        """Always return the unwrapped base model (no stacked adapters)."""
        model = self.base_model
        while hasattr(model, "base_model"):
            model = model.base_model
        return self.base_model  # return original ref so device_map stays

    def _load(self, name: str):
        path = cfg.ADAPTER_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {path}")
        t0 = time.time()
        self.active_model = PeftModel.from_pretrained(
            self._get_clean_base(), str(path), device_map=cfg.DEVICE
        )
        self.active_model.eval()
        self.current_lora = name
        dev = next(self.active_model.parameters()).device
        print(f"  ⚡ Loaded {name} on {dev} in {time.time()-t0:.1f}s")

    def _unload(self):
        if self.active_model is not None:
            self.active_model.cpu()
            del self.active_model
            self.active_model = None
            if cfg.DEVICE == "cuda":
                torch.cuda.empty_cache()
            self.current_lora = None

    def switch_to(self, name: str):
        if self.current_lora == name:
            return
        if self.current_lora:
            print(f"  🔄 Switching: {self.current_lora} → {name}")
            self._unload()
            self._switch_count += 1
        else:
            print(f"  📥 Loading: {name}")
        self._load(name)

    # ── Public API ────────────────────────────────────────────

    # def run(self, task: str, cv_text: str, max_new_tokens: int = None) -> str:
    #     lora_name = cfg.TASK_MAP.get(task)
    #     if not lora_name:
    #         raise ValueError(f"Unknown task: '{task}'. Use: {list(cfg.TASK_MAP)}")
    #     self.switch_to(lora_name)
    #
    #     if lora_name in cfg.CLASSIFICATION_LORAS:
    #         result = predict_classification(self.active_model, self.tokenizer, cv_text)
    #     else:
    #         tokens = max_new_tokens or 256
    #         result = generate_prediction(
    #             self.active_model, self.tokenizer,
    #             cfg.LORA_INSTRUCTIONS[lora_name], cv_text, tokens,
    #         )
    #
    #     self._call_count += 1
    #     return result

    def run(self, task: str, cv_text: str, max_new_tokens: int = None) -> str:
        from src.agent.inference import groq_call, classify_with_fallback

        lora_name = cfg.TASK_MAP.get(task)
        if not lora_name:
            raise ValueError(f"Unknown task: '{task}'. Use: {list(cfg.TASK_MAP)}")

        if lora_name == "lora1_classification":
            result = classify_with_fallback(cv_text)
        else:
            result = groq_call(lora_name, cv_text)

        self._call_count += 1
        return result

    def process_cv_full(self, cv_text: str) -> dict:
        results = {}
        for task in ["classify", "skills", "interview", "improve"]:
            t0 = time.time()
            results[task] = self.run(task, cv_text)
            print(f"  ✅ {task} done in {time.time()-t0:.1f}s")
        return results

    def unload_all(self):
        self._unload()
        print("🧹 LoRAs unloaded — base model still in memory")

    @property
    def status(self) -> dict:
        return {
            "device": cfg.DEVICE,
            "active_lora": self.current_lora,
            "switch_count": self._switch_count,
            "call_count": self._call_count,
            "adapters": {
                n: (cfg.ADAPTER_DIR / n).exists()
                for n in cfg.LORA_INSTRUCTIONS
            },
        }