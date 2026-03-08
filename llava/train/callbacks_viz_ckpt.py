import os, json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from transformers import TrainerCallback

import matplotlib
matplotlib.use("Agg")  # ✅ 服务器/无显示环境防报错
import matplotlib.pyplot as plt
import logging


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
        
#----------------------------------------------------------------------------------------------------------
        
#官方实现maybe_zero3
def maybe_zero_3(param, ignore_status=False, name=None):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except Exception:
        return param.detach().cpu().clone()
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def freeze_all_but_actor(model):
    # 先冻结全部参数
    model.requires_grad_(False)
    # model是LLaVAllamaForCausalLM
    # model.model是LlavaLlamaModel,他继承来自llamamodel
    #llamamodel中定义了actor,所以model.model.actor就是actor
    actor = model.model.actor
    # 只解冻 actor
    for p in actor.parameters():
        p.requires_grad = True

    # （可选）打印一下确认
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Actor-only] trainable params: {n_train}/{n_total} ({100*n_train/n_total:.4f}%)")
    return actor

def get_actor_state_maybe_zero_3(model):
    named_params = model.model.actor.named_parameters()
    # 复用已有的 maybe_zero_3
    state = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in named_params}
    return state

def save_actor_only(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    actor_state = get_actor_state_maybe_zero_3(model)
    # 推荐文件名
    torch.save(actor_state, os.path.join(output_dir, "actor.bin"))
        
class SaveActorCallback(TrainerCallback):
    def __init__(self, out_dir: str, save_every_steps: int = 200):
        self.out_dir = out_dir
        self.save_every_steps = save_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        # 只在真正到达保存步数时执行
        if state.global_step <= 0:
            return control
        if state.global_step % self.save_every_steps != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        # 只在 rank0 保存，避免多卡重复写
        if args.local_rank not in [-1, 0]:
            return control

        ckpt_dir = os.path.join(self.out_dir, f"actor_checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        save_actor_only(model, ckpt_dir)

        # 顺便把一些配置也一起存一下
        with open(os.path.join(ckpt_dir, "actor_config.json"), "w") as f:
            json.dump({
                "global_step": int(state.global_step),
                "SYS_PROMPT_LEN": os.environ.get("SYS_PROMPT_LEN"),
                "IMG_TOKEN_LEN": os.environ.get("IMG_TOKEN_LEN"),
                "PRUNE_LAYER_INDEX": os.environ.get("PRUNE_LAYER_INDEX"),
                "MODEL_DIM": os.environ.get("MODEL_DIM"),
                "NUM_HEADS": os.environ.get("NUM_HEADS"),
                "DROPOUT": os.environ.get("DROPOUT"),
                "TOP_K": os.environ.get("TOP_K"),
                "TAU": os.environ.get("TAU"),
                "USE_LAYERNORM": os.environ.get("USE_LAYERNORM"),
                "HARD_MODE": os.environ.get("HARD_MODE"),
                "TARGET_PRUNE_RATIO": os.environ.get("TARGET_PRUNE_RATIO"),
                "LAMBDA_PRUNE": os.environ.get("LAMBDA_PRUNE"),
            }, f, indent=2)

        print(f"[Actor Save] saved actor checkpoint to {ckpt_dir}")

        return control