"""Checkpoint saving helper."""

import os
import torch


def save_model_w_condition(model,
                           model_dir: str,
                           model_name: str,
                           accu: float,
                           target_accu: float,
                           log=print) -> None:
    """정확도가 target을 넘으면 전체 모델을 `.pth`로 저장."""

    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model,
                   f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
