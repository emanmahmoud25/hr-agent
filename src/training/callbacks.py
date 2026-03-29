"""Training progress callback for tqdm display."""
from tqdm import tqdm
from transformers import TrainerCallback


class ProgressBarCallback(TrainerCallback):
    def __init__(self, total_steps: int, name: str):
        self.bar = tqdm(
            total      = 100,
            desc       = f"  {name}",
            unit       = "%",
            bar_format = "{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}] {postfix}",
            leave      = True,
            position   = 1,
        )
        self.total_steps = total_steps
        self.last_pct    = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step and self.total_steps:
            new_pct = int(100 * state.global_step / self.total_steps)
            if new_pct > self.last_pct:
                self.bar.update(new_pct - self.last_pct)
                self.last_pct = new_pct
            if logs and "loss" in logs:
                self.bar.set_postfix({
                    "loss": f"{logs['loss']:.4f}",
                    "step": f"{state.global_step}/{self.total_steps}",
                })

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar.n < 100:
            self.bar.update(100 - self.bar.n)
        self.bar.set_postfix({"status": "✅ done"})
        self.bar.close()
