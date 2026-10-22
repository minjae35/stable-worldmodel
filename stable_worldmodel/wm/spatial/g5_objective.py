"""Label-free, environment-agnostic objectives for Stage-3 G5.

Only frozen visual features, predictions, and actions routed through the model
are used here.  State, masks, rewards, contacts, and environment-specific
definitions are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


G5_ARMS = (
    'A_uniform_control',
    'B_motion_balanced',
    'C_action_sensitive',
    'D_combined',
)


@dataclass(frozen=True)
class ObjectiveSpec:
    motion_balanced: bool
    action_sensitive: bool


ARM_SPECS = {
    'A_uniform_control': ObjectiveSpec(False, False),
    'B_motion_balanced': ObjectiveSpec(True, False),
    'C_action_sensitive': ObjectiveSpec(False, True),
    'D_combined': ObjectiveSpec(True, True),
}


def make_derangement_schedule(steps: int, batch_size: int, seed: int) -> np.ndarray:
    """Deterministic in-batch action permutations with no fixed points."""
    if steps < 1 or batch_size < 2:
        raise ValueError('derangement schedule requires steps >= 1 and batch >= 2')
    rng = np.random.default_rng(seed)
    output = np.empty((steps, batch_size), dtype=np.int64)
    for step in range(steps):
        order = rng.permutation(batch_size)
        permutation = np.empty(batch_size, dtype=np.int64)
        permutation[order] = np.roll(order, 1)
        if np.any(permutation == np.arange(batch_size)):
            raise RuntimeError('constructed action permutation has a fixed point')
        output[step] = permutation
    return output


def patch_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('prediction and target must both be (B,H,P,D)')
    return (prediction.float() - target.detach().float()).square().mean(dim=-1)


def local_motion_weights(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    grid: int = 16,
    tile: int = 4,
    blend: float = 0.5,
    relative_cap: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Locally normalized motion weights with unit mean in every spatial tile.

    Local normalization prevents a single large mover from taking the full
    image budget.  Clipping limits outliers, while ``blend`` retains a uniform
    component so static prediction cannot be discarded.
    """
    if current.shape != target.shape or current.ndim != 4:
        raise ValueError('current and target must both be (B,H,P,D)')
    if current.shape[2] != grid * grid or grid % tile:
        raise ValueError('patch count must match a grid divisible by tile size')
    if not 0.0 <= blend <= 1.0 or relative_cap < 1.0:
        raise ValueError('invalid motion weighting hyperparameters')
    motion = (target.detach().float() - current.detach().float()).square().mean(-1).sqrt()
    b, h, _ = motion.shape
    ntiles = grid // tile
    tiled = motion.reshape(b, h, ntiles, tile, ntiles, tile)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).reshape(b, h, ntiles, ntiles, tile * tile)
    relative = tiled / tiled.mean(-1, keepdim=True).clamp_min(eps)
    relative = relative.clamp(min=1.0 / relative_cap, max=relative_cap)
    # Bounded, centered log-ratio keeps the exact per-tile mean at one.  At
    # blend=0.5 weights are in [0,2], so no patch can dominate a tile.
    signal = torch.tanh(torch.log(relative))
    signal = signal - signal.mean(-1, keepdim=True)
    weights = 1.0 + blend * signal
    weights = weights.reshape(b, h, ntiles, ntiles, tile, tile)
    weights = weights.permute(0, 1, 2, 4, 3, 5).reshape(b, h, grid * grid)
    return weights.detach()


def weighted_patch_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    if values.shape != weights.shape:
        raise ValueError('patch values and weights must have identical shape')
    normalized = weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return (values * normalized).mean()


def detached_negative_ranking_loss(
    correct_prediction: torch.Tensor,
    shuffled_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    margin_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require factual patch error to beat a detached shuffled-action error.

    The negative branch is detached: optimization can improve the factual
    prediction but cannot satisfy the ranking term by deliberately corrupting
    shuffled-action output.  No counterfactual target is asserted.
    """
    if not 0.0 < margin_fraction < 1.0:
        raise ValueError('margin_fraction must be between zero and one')
    correct_error = patch_mse(correct_prediction, target)
    shuffled_error = patch_mse(shuffled_prediction, target).detach()
    hinge = torch.relu(correct_error - (1.0 - margin_fraction) * shuffled_error)
    loss = weighted_patch_mean(hinge, weights)
    active = (hinge > 0).float().mean().detach()
    return loss, active


def objective_components(
    arm: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    shuffled_prediction: torch.Tensor | None,
    *,
    motion_blend: float,
    relative_cap: float,
    margin_fraction: float,
    ranking_lambda: float,
    objective_scale: float,
) -> dict[str, torch.Tensor]:
    if arm not in ARM_SPECS:
        raise ValueError(f'unknown G5 arm {arm!r}')
    spec = ARM_SPECS[arm]
    uniform = patch_mse(prediction, target).mean()
    weights = local_motion_weights(
        current, target, blend=motion_blend, relative_cap=relative_cap
    )
    motion = weighted_patch_mean(patch_mse(prediction, target), weights)
    base = motion if spec.motion_balanced else uniform
    ranking = uniform.new_zeros(())
    active = uniform.new_zeros(())
    if spec.action_sensitive:
        if shuffled_prediction is None:
            raise ValueError('action-sensitive arms require a shuffled prediction')
        ranking, active = detached_negative_ranking_loss(
            prediction,
            shuffled_prediction,
            target,
            weights=weights if spec.motion_balanced else None,
            margin_fraction=margin_fraction,
        )
    raw = base + float(ranking_lambda) * ranking
    return {
        'objective': raw * float(objective_scale),
        'raw_objective': raw,
        'uniform_mse': uniform,
        'motion_balanced_mse': motion,
        'ranking_loss': ranking,
        'ranking_active_fraction': active,
        'weight_min': weights.min(),
        'weight_max': weights.max(),
        'weight_mean': weights.mean(),
    }
