import os, json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from transformers import TrainerCallback

import matplotlib
matplotlib.use("Agg")  # ✅ 服务器/无显示环境防报错
import matplotlib.pyplot as plt


def _unwrap_model(model):
    return getattr(model, "module", model)


@dataclass
class MetricBuffer:
    steps: List[int] = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)
    lm_loss: List[float] = field(default_factory=list)
    prune_loss: List[float] = field(default_factory=list)
    prune_ratio: List[float] = field(default_factory=list)

    def append(self, step: int, total: float, lm: float, prune: float, ratio: float):
        self.steps.append(step)
        self.total_loss.append(total)
        self.lm_loss.append(lm)
        self.prune_loss.append(prune)
        self.prune_ratio.append(ratio)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": self.steps[i],
                "total_loss": self.total_loss[i],
                "lm_loss": self.lm_loss[i],
                "prune_loss_mean": self.prune_loss[i],
                "prune_ratio_mean": self.prune_ratio[i],
            }
            for i in range(len(self.steps))
        ]


class LossPlotCallback(TrainerCallback):
    """
    每隔 plot_every_steps:
    - 追加写 metrics.jsonl
    - 保存曲线图（latest 覆盖 + step 版本不覆盖）
    """
    def __init__(self, out_dir: str, plot_every_steps: int = 50, keep_step_png: bool = True):
        self.out_dir = out_dir
        self.plot_every_steps = int(plot_every_steps)
        self.keep_step_png = keep_step_png

        self.buf = MetricBuffer()
        os.makedirs(self.out_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.out_dir, "metrics.jsonl")
        self.loss_latest_path = os.path.join(self.out_dir, "loss_curves_latest.png")
        self.ratio_latest_path = os.path.join(self.out_dir, "prune_ratio_latest.png")

        self._last_flushed_idx = 0

    def on_step_end(self, args, state, control, **kwargs):
        # ✅ 多卡：只在 rank0 写
        if hasattr(state, "is_world_process_zero") and (not state.is_world_process_zero):
            return

        model = _unwrap_model(kwargs.get("model"))

        # ✅ 拼写修正：newest_*
        needed = ["newest_total_loss", "newest_lm_loss", "newest_prune_loss", "newest_prune_ratio"]
        if not all(hasattr(model, k) for k in needed):
            # 建议：只偶尔打印，避免刷屏
            if state.global_step % max(self.plot_every_steps, 50) == 0:
                missing = [k for k in needed if not hasattr(model, k)]
                print(f"[LossPlotCallback] missing attrs: {missing}")
            return

        step = int(state.global_step)
        total = float(model.newest_total_loss.detach().float().cpu().item())
        lm = float(model.newest_lm_loss.detach().float().cpu().item())
        prune = float(model.newest_prune_loss.detach().float().cpu().item())
        ratio = float(model.newest_prune_ratio.detach().float().cpu().item())

        self.buf.append(step, total, lm, prune, ratio)

        if step > 0 and (step % self.plot_every_steps == 0):
            self._flush_jsonl()
            self._plot(step)

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and (not state.is_world_process_zero):
            return
        self._flush_jsonl()
        self._plot(int(state.global_step))

    def _flush_jsonl(self):
        rows = self.buf.to_rows()
        if self._last_flushed_idx >= len(rows):
            return
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for r in rows[self._last_flushed_idx:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._last_flushed_idx = len(rows)

    def _plot(self, step: int):
        if len(self.buf.steps) < 2:
            return

        # -------- loss curves --------
        plt.figure()
        plt.plot(self.buf.steps, self.buf.total_loss, label="total_loss")
        plt.plot(self.buf.steps, self.buf.lm_loss, label="lm_loss")
        plt.plot(self.buf.steps, self.buf.prune_loss, label="prune_loss_mean")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()

        # ✅ latest 覆盖
        plt.savefig(self.loss_latest_path, dpi=150)

        # ✅ step 版本不覆盖
        if self.keep_step_png:
            step_path = os.path.join(self.out_dir, f"loss_curves_step{step:07d}.png")
            plt.savefig(step_path, dpi=150)

        plt.close()

        # -------- prune ratio --------
        plt.figure()
        plt.plot(self.buf.steps, self.buf.prune_ratio, label="prune_ratio_mean")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("prune_ratio")
        plt.tight_layout()

        plt.savefig(self.ratio_latest_path, dpi=150)
        if self.keep_step_png:
            step_path = os.path.join(self.out_dir, f"prune_ratio_step{step:07d}.png")
            plt.savefig(step_path, dpi=150)

        plt.close()