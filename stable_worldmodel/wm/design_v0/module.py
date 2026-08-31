"""Design v0 action and shared dynamics modules."""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    """Embed concatenated padded actions and boolean action masks."""

    def __init__(self, max_action_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if max_action_dim <= 0:
            raise ValueError('max_action_dim must be positive')
        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be positive')

        self.max_action_dim = int(max_action_dim)
        self.embedding_dim = int(embedding_dim)
        self.linear = nn.Linear(2 * self.max_action_dim, self.embedding_dim)
        self.activation = nn.GELU()

    def forward(
        self, action: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        if action.shape != action_mask.shape:
            raise ValueError(
                'action and action_mask must have the same shape, '
                f'got {tuple(action.shape)} and {tuple(action_mask.shape)}'
            )
        if action.ndim != 2 or action.shape[-1] != self.max_action_dim:
            raise ValueError(
                f'action must have shape (B, {self.max_action_dim}), '
                f'got {tuple(action.shape)}'
            )
        if action_mask.dtype is not torch.bool:
            raise TypeError('action_mask must have boolean dtype')

        action_with_mask = torch.cat(
            [action, action_mask.to(dtype=action.dtype)], dim=-1
        )
        return self.activation(self.linear(action_with_mask))


class SharedDynamics(nn.Module):
    """Two-layer MLP that predicts a visual-latent delta."""

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, latent_dim) <= 0:
            raise ValueError('Dynamics dimensions must be positive')

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))
