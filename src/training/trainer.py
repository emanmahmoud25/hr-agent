"""Train a single LoRA adapter."""
import shutil
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb

from src.config import cfg
from src.data.dataset import load_json_dataset
from src.training.callbacks import ProgressBarCallback
from src.training.evaluate import evaluate_adapter


def is_trained(name: str) -> bool:
    for base in [Path("./adapters"), cfg.ADAPTER_DIR]:
        p = base / name
        if (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists():
            return True
    return False


def get_last_checkpoint(name: str) -> str | None:
    for base in [Path("./adapters"), cfg.ADAPTER_DIR]:
        p = base / name
        if p.exists():
            ckpts = sorted(p.glob("checkpoint-*"),
                           key=lambda x: int(x.name.split("-")[-1]))
            if ckpts:
                return str(ckpts[-1])
    return None


def train_one(name: str, base_model, tokenizer) -> bool:
    if is_trained(name):
        print(f"\n⏭  SKIP {name} — already trained")
        return True

    train_ds = load_json_dataset(name, "train")
    val_ds   = load_json_dataset(name, "val")

    if train_ds is None or len(train_ds) == 0:
        print(f"\n⚠️  No training data for {name} — skipping")
        return False

    print(f'\n{"─"*60}')
    print(f"  Training: {name}  ({len(train_ds)} samples)")
    print(f'{"─"*60}')

    lora_config = LoraConfig(
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
        target_modules = ["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    train_cfg   = cfg.LORA_TRAIN_CONFIGS.get(name, {})
    out_dir     = str(Path("./adapters") / name)
    resume      = get_last_checkpoint(name)
    batch_size  = 2
    grad_accum  = 8
    epochs      = train_cfg.get("num_train_epochs", 3)
    total_steps = max(1, len(train_ds) // (batch_size * grad_accum)) * epochs

    wandb.init(project=cfg.WANDB_PROJECT, name=name, reinit=True)
    cb = ProgressBarCallback(total_steps=total_steps, name=name)

    use_fp16 = cfg.DEVICE == "cuda"

    trainer = SFTTrainer(
        model              = peft_model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        dataset_text_field = "text",
        max_seq_length     = cfg.MAX_SEQ_LENGTH,
        callbacks          = [cb],
        args               = TrainingArguments(
            output_dir                  = out_dir,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate               = train_cfg.get("learning_rate", 5e-5),
            lr_scheduler_type           = "cosine",
            warmup_ratio                = 0.05,
            fp16                        = use_fp16,
            logging_steps               = 10,
            save_steps                  = 50,
            save_total_limit            = 3,
            evaluation_strategy         = "steps" if val_ds else "no",
            eval_steps                  = 50       if val_ds else None,
            load_best_model_at_end      = False,
            report_to                   = "wandb",
            run_name                    = name,
        ),
    )

    ok = False
    try:
        trainer.train(resume_from_checkpoint=resume)
        ok = True
    except Exception as e:
        print(f"\n  ❌ Training error: {e}")

    wandb.finish()

    if ok:
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        src = Path("./adapters") / name
        dst = cfg.ADAPTER_DIR   / name
        print(f"\n  💾 Saving {name} → {dst}...")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        total_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"  ✅ Saved — {total_mb:.1f} MB → {dst}")

        wandb.init(project=cfg.WANDB_PROJECT, name=f"{name}_eval", reinit=True)
        evaluate_adapter(name, out_dir, base_model, tokenizer)
        wandb.finish()

    return ok
