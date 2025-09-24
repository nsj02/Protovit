"""Common helper utilities used across training, push, and analysis scripts."""

import os
import torch
import numpy as np


def list_of_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """L2 거리 리스트 계산.

    - 입력: `X` shape [N, D], `Y` shape [M, D]
    - 출력: `tensor` shape [N, M] (각 행: X_i와 모든 Y_j 간의 제곱거리)
    """

    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)


def make_one_hot(target: torch.Tensor, target_one_hot: torch.Tensor) -> None:
    """정수 라벨 벡터를 원-핫 텐서에 in-place로 기록."""

    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path: str) -> None:
    """간단한 `mkdir -p` 래퍼."""

    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(text: str, file) -> None:
    """문자열을 stdout과 파일에 동시에 기록."""

    print(text)
    file.write(text + '\n')


def find_high_activation_crop(activation_map: np.ndarray, percentile: float = 95):
    """활성 맵에서 상위 percentile 영역을 bounding box로 반환."""

    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1
