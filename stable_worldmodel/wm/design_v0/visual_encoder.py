"""Frozen pretrained visual encoder for Design v0."""

from __future__ import annotations

import torch
from torch import nn

from stable_worldmodel.wm.prejepa.module import create_backbone


class FrozenVisualEncoder(nn.Module):
    """Wrap a pretrained token-based backbone and return final-layer CLS tokens.

    The wrapper owns no learnable projection. The backbone remains in evaluation
    mode even when the surrounding Design v0 model is switched to training mode.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

        config = getattr(backbone, 'config', None)
        latent_dim = getattr(config, 'hidden_size', None)
        if latent_dim is None:
            raise ValueError(
                'The visual backbone must expose config.hidden_size'
            )
        self.latent_dim = int(latent_dim)

        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_pretrained(cls, name: str) -> FrozenVisualEncoder:
        """Load a pretrained backbone through the existing shared loader."""
        return cls(create_backbone(name))

    def train(self, mode: bool = True) -> FrozenVisualEncoder:
        """Keep the frozen backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, C, H, W)`` pixels as ``(B, T, D)`` CLS latents."""
        if pixels.ndim != 5:
            raise ValueError(
                'pixels must have shape (B, T, C, H, W), '
                f'got {tuple(pixels.shape)}'
            )

        batch_size, num_steps = pixels.shape[:2]
        flat_pixels = pixels.reshape(
            batch_size * num_steps, *pixels.shape[2:]
        )

        parameter = next(self.backbone.parameters(), None)
        if parameter is not None and flat_pixels.is_floating_point():
            flat_pixels = flat_pixels.to(dtype=parameter.dtype)

        with torch.no_grad():
            output = self.backbone(flat_pixels)

        hidden = getattr(output, 'last_hidden_state', None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[1] < 1:
            raise ValueError(
                'The visual backbone must return final-layer '
                'last_hidden_state with a CLS token'
            )

        cls = hidden[:, 0]
        if cls.shape[-1] != self.latent_dim:
            raise ValueError(
                f'CLS width {cls.shape[-1]} does not match '
                f'config.hidden_size={self.latent_dim}'
            )
        return cls.reshape(batch_size, num_steps, self.latent_dim)
