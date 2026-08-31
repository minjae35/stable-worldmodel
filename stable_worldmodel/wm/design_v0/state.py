"""Latent state construction for Design v0."""

from __future__ import annotations

import torch


def concatenate_latent_history(
    latents: torch.Tensor, history_size: int
) -> torch.Tensor:
    """Concatenate the latest K latents from oldest to newest.

    Args:
        latents: Visual latent sequence with shape ``(B, T, D)``.
        history_size: Number of recent latents K to include.

    Returns:
        Ordered state ``[z_{t-K+1}; ...; z_t]`` with shape ``(B, K * D)``.
    """
    if latents.ndim != 3:
        raise ValueError(
            f'latents must have shape (B, T, D), got {tuple(latents.shape)}'
        )
    if history_size <= 0:
        raise ValueError('history_size must be positive')
    if latents.shape[1] < history_size:
        raise ValueError(
            f'Need at least {history_size} latents, got {latents.shape[1]}'
        )

    recent = latents[:, -history_size:]
    return recent.reshape(latents.shape[0], -1)
