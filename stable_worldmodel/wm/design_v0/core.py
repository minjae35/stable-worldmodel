"""One-step Design v0 model core."""

from __future__ import annotations

import torch
from torch import nn

from .module import ActionEncoder, SharedDynamics
from .state import concatenate_latent_history
from .visual_encoder import FrozenVisualEncoder


class DesignV0Core(nn.Module):
    """Encode visual history and predict one residual next visual latent."""

    def __init__(
        self,
        visual_encoder: FrozenVisualEncoder,
        *,
        history_size: int,
        max_action_dim: int,
        action_embedding_dim: int,
        num_environments: int,
        environment_embedding_dim: int,
        dynamics_hidden_dim: int,
    ) -> None:
        super().__init__()
        if history_size <= 0:
            raise ValueError('history_size must be positive')
        if num_environments <= 0:
            raise ValueError('num_environments must be positive')
        if environment_embedding_dim <= 0:
            raise ValueError('environment_embedding_dim must be positive')

        self.visual_encoder = visual_encoder
        self.history_size = int(history_size)
        self.latent_dim = visual_encoder.latent_dim
        self.action_encoder = ActionEncoder(
            max_action_dim=max_action_dim,
            embedding_dim=action_embedding_dim,
        )
        self.environment_embedding = nn.Embedding(
            num_embeddings=num_environments,
            embedding_dim=environment_embedding_dim,
        )

        dynamics_input_dim = (
            self.history_size * self.latent_dim
            + action_embedding_dim
            + environment_embedding_dim
        )
        self.dynamics = SharedDynamics(
            input_dim=dynamics_input_dim,
            hidden_dim=dynamics_hidden_dim,
            latent_dim=self.latent_dim,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return one frozen CLS latent per visual observation."""
        return self.visual_encoder(pixels)

    def build_state(self, latents: torch.Tensor) -> torch.Tensor:
        """Build ``[z_{t-K+1}; ...; z_t]`` from the latest K latents."""
        return concatenate_latent_history(latents, self.history_size)

    def predict_next(
        self,
        latents: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        env_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``z_{t+1}`` from K latents and the action leaving ``z_t``."""
        batch_size = latents.shape[0]
        if env_id.ndim != 1 or env_id.shape[0] != batch_size:
            raise ValueError(
                f'env_id must have shape ({batch_size},), '
                f'got {tuple(env_id.shape)}'
            )
        if action.shape[0] != batch_size:
            raise ValueError('action batch size must match latents')

        state = self.build_state(latents)
        action_embedding = self.action_encoder(action, action_mask)
        environment_embedding = self.environment_embedding(env_id)
        dynamics_input = torch.cat(
            [state, action_embedding, environment_embedding], dim=-1
        )

        delta = self.dynamics(dynamics_input)
        current_latent = latents[:, -1]
        return current_latent + delta

    def forward(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        env_id: torch.Tensor,
    ) -> torch.Tensor:
        """Encode K visual observations and predict one next visual latent."""
        latents = self.encode(pixels)
        return self.predict_next(latents, action, action_mask, env_id)
