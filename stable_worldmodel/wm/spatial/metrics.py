"""Diagnostics for one-step spatial prediction.

Uniform patch MSE is the training loss only. It is not a planning cost:
G0 found essentially no correlation between that distance and Cube success.
"""

from __future__ import annotations

import torch


def prediction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on spatial tokens. The target is a label."""
    return (pred - target.detach()).pow(2).mean()


def moving_patch_errors(
    pred: torch.Tensor, target: torch.Tensor, current: torch.Tensor
) -> dict[str, float]:
    """Error on patches that actually change, versus patches that do not.

    ``pred``, ``target`` and ``current`` are ``(B, P, D)``. The moving set is
    the patches whose target-versus-current energy is above the per-sample
    median, so a static background cannot hide a small moving object.
    """
    if pred.shape != target.shape or pred.shape != current.shape:
        raise ValueError('pred, target and current shapes must match')
    delta = (target - current).pow(2).mean(dim=-1)
    move = delta > delta.median(dim=-1, keepdim=True).values
    err = (pred - target).pow(2).mean(dim=-1)
    persist = (current - target).pow(2).mean(dim=-1)

    def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.sum()) == 0:
            return float('nan')
        return float(values[mask].mean())

    return {
        'moving_mse': _mean(err, move),
        'static_mse': _mean(err, ~move),
        'moving_persistence': _mean(persist, move),
        'static_persistence': _mean(persist, ~move),
        'moving_fraction': float(move.float().mean()),
    }
