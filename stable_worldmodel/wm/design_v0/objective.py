"""Training-time recursive rollout and multi-step MSE for Design v0."""

from __future__ import annotations

import torch
from torch import nn

from .core import DesignV0Core


def transition_actions(
    action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    history_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the H actions that leave observations ``t, t+1, ..., t+H-1``.

    Dataset clips store ``action[i]`` as the transition
    ``observation[i] → observation[i+1]``.  After K history frames the first
    prediction therefore consumes ``action[K-1]``.
    """
    if action.shape != action_mask.shape:
        raise ValueError(
            'action and action_mask must have the same shape, '
            f'got {tuple(action.shape)} and {tuple(action_mask.shape)}'
        )
    if action.ndim != 3:
        raise ValueError(
            f'action must have shape (B, T, A) or (B, H, A), got {tuple(action.shape)}'
        )

    time = action.shape[1]
    if time == horizon:
        return action, action_mask
    needed = history_size + horizon - 1
    if time < needed:
        raise ValueError(
            f'Need {needed} transition actions for history_size={history_size} '
            f'and horizon={horizon}, got {time}'
        )
    start = history_size - 1
    end = start + horizon
    return action[:, start:end], action_mask[:, start:end]


def recursive_rollout(
    core: DesignV0Core,
    history_latents: torch.Tensor,
    action: torch.Tensor,
    action_mask: torch.Tensor,
    env_id: torch.Tensor,
) -> torch.Tensor:
    """Roll predicted latents for H steps without injecting ground-truth futures.

    Args:
        history_latents: Initial ``[z_{t-K+1}, ..., z_t]`` with shape ``(B, K, D)``.
        action: Horizon-aligned actions ``[a_t, ..., a_{t+H-1}]`` of shape
            ``(B, H, A)``.
        action_mask: Boolean mask with the same shape as ``action``.
        env_id: Integer environment ids of shape ``(B,)``.

    Returns:
        Predicted latents ``[z_hat_{t+1}, ..., z_hat_{t+H}]`` of shape
        ``(B, H, D)``.
    """
    if history_latents.ndim != 3:
        raise ValueError(
            'history_latents must have shape (B, K, D), '
            f'got {tuple(history_latents.shape)}'
        )
    if history_latents.shape[1] != core.history_size:
        raise ValueError(
            f'history_latents must contain {core.history_size} steps, '
            f'got {history_latents.shape[1]}'
        )
    if action.ndim != 3 or action_mask.ndim != 3:
        raise ValueError('rollout actions must have shape (B, H, A)')
    if action.shape != action_mask.shape:
        raise ValueError(
            'action and action_mask must have the same shape, '
            f'got {tuple(action.shape)} and {tuple(action_mask.shape)}'
        )

    horizon = action.shape[1]
    if horizon <= 0:
        raise ValueError('horizon must be positive')

    history = history_latents
    predictions = []
    for step in range(horizon):
        predicted = core.predict_next(
            history,
            action[:, step],
            action_mask[:, step],
            env_id,
        )
        predictions.append(predicted)
        history = torch.cat([history[:, 1:], predicted.unsqueeze(1)], dim=1)
    return torch.stack(predictions, dim=1)


class DesignV0Objective(nn.Module):
    """Frozen-encoder multi-step MSE used to train Design v0 dynamics."""

    def __init__(self, core: DesignV0Core) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        env_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a ``K+H`` clip, roll out H predicted latents, and average MSE.

        Args:
            pixels: RGB clip ``(B, K+H, C, H, W)``.  The first K frames are
                history; the last H frames are frozen CLS targets.
            action: Either H horizon-aligned actions ``(B, H, A)`` or a clip of
                transition actions ``(B, T, A)`` with ``T >= K+H-1``.
            action_mask: Boolean mask matching ``action``.
            env_id: Integer environment ids of shape ``(B,)``.

        Returns:
            Dict with scalar ``loss``, predictions ``(B, H, D)``, detached
            targets ``(B, H, D)``, and per-horizon MSE ``(H,)``.
        """
        if pixels.ndim != 5:
            raise ValueError(
                f'pixels must have shape (B, T, C, H, W), got {tuple(pixels.shape)}'
            )

        history_size = self.core.history_size
        if pixels.shape[1] <= history_size:
            raise ValueError(
                f'Need more than {history_size} frames for a training clip, '
                f'got {pixels.shape[1]}'
            )

        latents = self.core.encode(pixels)
        horizon = latents.shape[1] - history_size
        history_latents = latents[:, :history_size]
        targets = latents[:, history_size:].detach()

        rollout_action, rollout_mask = transition_actions(
            action,
            action_mask,
            history_size=history_size,
            horizon=horizon,
        )
        predicted = recursive_rollout(
            self.core,
            history_latents,
            rollout_action,
            rollout_mask,
            env_id,
        )

        # Per-horizon MSE over batch and latent dim, then mean over H.
        per_horizon_mse = (predicted - targets).pow(2).mean(dim=(0, 2))
        loss = per_horizon_mse.mean()
        return {
            'loss': loss,
            'predicted': predicted,
            'target': targets,
            'per_horizon_mse': per_horizon_mse,
        }
