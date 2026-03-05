# 修复 PyTorch 2.6+ 的 weights_only 问题
# Monkey-patch torch.load 以默认使用 weights_only=False
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
