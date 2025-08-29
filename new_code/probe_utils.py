# -*- coding: utf-8 -*-
"""
probe_utils.py

A collection of utility classes and helpers shared by different probe training
scripts. Moving these common pieces here keeps the main training scripts clean
and avoids code duplication.

Author: Assistant
Date: 2025-07-18
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = [
    "FocalLoss",
    "ClassBalancedLoss",
    "S1TextDataset",
    "EnhancedProbe",
    "MLPProbe",
]


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Parameters
    ----------
    alpha : float | torch.Tensor | None, optional
        Weighting factor for the rare class. If a tensor is provided it should
        contain per-class weights. If `None`, no weighting is applied.
    gamma : float, default 2.0
        Focusing parameter that down-weights easy examples and focuses learning
        on hard negatives.
    reduction : {"mean", "sum", "none"}, default "mean"
        Specifies the reduction to apply to the output.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Ensure alpha tensor is on the correct device
                if hasattr(self.alpha, "device") and self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on the effective number of samples.

    References
    ----------
    Cui, Yin, et al. "Class-balanced loss based on effective number of samples."
    CVPR 2019.
    """

    def __init__(self, samples_per_class: List[int], beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.register_buffer("weights", torch.FloatTensor(weights))  # type: ignore[arg-type]
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        labels_one_hot = F.one_hot(targets, num_classes=len(self.weights)).float()
        weights = self.weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        cb_loss = F.cross_entropy(inputs, targets, reduction="none")
        cb_loss = weights * cb_loss

        if self.gamma > 0:
            pt = torch.exp(-cb_loss)
            cb_loss = (1 - pt) ** self.gamma * cb_loss

        return cb_loss.mean()


class S1TextDataset(Dataset):
    """A minimal text-classification dataset wrapper for probe training."""

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.texts)

    def __getitem__(self, idx):  # type: ignore[override]
        return self.texts[idx], self.labels[idx] 


# ----------------------------------
# Simple Probe Architecture
# ----------------------------------


class EnhancedProbe(nn.Module):
    """A simple linear probe.

    It supports two common constructor signatures used across the codebase:

    1. EnhancedProbe(input_dim, output_dim)
    2. EnhancedProbe(input_dim, hidden_dim, output_dim)

    In the two-argument variant the model is just a single linear layer.
    In the three-argument variant a hidden layer can be added, but to keep
    things lightweight we map *input_dim → output_dim* directly, ignoring
    ``hidden_dim``.  This mirrors the behaviour of the training scripts where
    only a single linear projection is ultimately used.
    """

    def __init__(self, input_dim: int, hidden_or_output_dim: int, output_dim: Optional[int] = None, *, dropout_rate: float = 0.3):
        super().__init__()

        # Determine whether we are in 2-arg or 3-arg mode.
        if output_dim is None:
            # Signature: (input_dim, output_dim)
            output_dim = hidden_or_output_dim
        # else: we were given (input_dim, hidden_dim, output_dim) – we ignore hidden_dim.

        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


class MLPProbe(nn.Module):
    """An MLP probe with a single hidden layer with fixed dimensionality.

    This probe is a non-linear classifier, based on a multi-layer perceptron (MLP)
    with a single hidden layer of 512 units and ReLU activation, as described
    in related research literature.

    The architecture is:
    - Linear(input_dim, 512)
    - ReLU()
    - Linear(512, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = 512
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x) 