"""Shared spatial dynamics on frozen DINOv2 patch tokens.

Design, relative to a flat 6-layer width-256 predictor:

- The encoder is pretrained DINOv2-S/14 and stays frozen. The CLS token is
  dropped. 224x224 input yields 16x16 patch tokens of dimension 384.
- The dynamics are factorized. Each block does full attention over the
  patches of one frame, then causal attention over time at each patch.
  A flat sequence would also fit in memory at history 1, but a hand-rolled
  space-time mask is easy to get wrong. Factorizing makes "no future frame
  and no future action" a property of the mask, which the tests check at
  history 2. The G1 run itself is one-step (history 1).
- Depth 6, hidden 256 and 8 heads are the starting predictor. The MLP ratio
  is 4. Dropout is 0 so a small set can actually be fit.
- Each environment has its own action MLP. The embedding is added onto
  every patch of that frame. It is not AdaLN-zero: a zero-initialized
  modulation would ignore actions at step 0 and hide a wiring bug.
- An environment id embedding is added as well, so the shared trunk can
  tell the three domains apart.
- The training target is the next frame's frozen patch tokens. Uniform MSE
  is only that training loss. No planning cost is implemented.
"""

from __future__ import annotations

import torch
from torch import nn

ENV_NAMES = ('pusht', 'tworoom', 'cube')
DEFAULT_ACTION_DIMS = {'pusht': 10, 'tworoom': 10, 'cube': 25}


def causal_time_mask(length: int, device, dtype) -> torch.Tensor:
    """Additive mask of shape ``(length, length)``.

    Position ``t`` may attend to ``0..t`` and not to any later frame.
    A finite entry is 0. A future entry is ``-inf``.
    """
    if length < 1:
        raise ValueError('temporal length must be positive')
    mask = torch.zeros(length, length, device=device, dtype=dtype)
    future = torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask.masked_fill(future, float('-inf'))


class ActionAdapter(nn.Module):
    """Map one environment's blocked action to the shared hidden size."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class FactorizedSTBlock(nn.Module):
    """Spatial attention inside a frame, then causal attention across time."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_t = nn.LayerNorm(dim)
        self.temporal = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_m = nn.LayerNorm(dim)
        inner = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, H, P, C). Spatial attention never crosses frames.
        b, h, p, c = x.shape
        spatial = x.reshape(b * h, p, c)
        hidden = self.norm_s(spatial)
        spatial = spatial + self.spatial(
            hidden, hidden, hidden, need_weights=False
        )[0]
        spatial = spatial.reshape(b, h, p, c)

        # Temporal attention never looks past the current frame.
        temporal = spatial.permute(0, 2, 1, 3).reshape(b * p, h, c)
        hidden = self.norm_t(temporal)
        temporal = temporal + self.temporal(
            hidden,
            hidden,
            hidden,
            attn_mask=time_mask,
            need_weights=False,
        )[0]
        temporal = temporal.reshape(b, p, h, c).permute(0, 2, 1, 3)
        return temporal + self.mlp(self.norm_m(temporal))


class SpatialDynamics(nn.Module):
    """Predict the next DINOv2 patch tokens from context tokens and actions."""

    def __init__(
        self,
        token_dim: int = 384,
        hidden: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        grid: int = 16,
        action_dims: dict | None = None,
        env_names: tuple[str, ...] = ENV_NAMES,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f'hidden {hidden} must be divisible by heads {heads}')
        if depth < 1 or grid < 1:
            raise ValueError('depth and grid must be positive')
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.heads = int(heads)
        self.grid = int(grid)
        self.num_patches = self.grid * self.grid
        self.env_names = tuple(env_names)
        self.env_to_id = {name: i for i, name in enumerate(self.env_names)}
        self.action_dims = dict(DEFAULT_ACTION_DIMS if action_dims is None else action_dims)
        missing = [name for name in self.env_names if name not in self.action_dims]
        if missing:
            raise ValueError(f'missing action dims for {missing}')

        self.in_proj = nn.Linear(self.token_dim, self.hidden)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.hidden))
        self.env_embed = nn.Embedding(len(self.env_names), self.hidden)
        self.adapters = nn.ModuleDict(
            {
                name: ActionAdapter(self.action_dims[name], self.hidden)
                for name in self.env_names
            }
        )
        self.blocks = nn.ModuleList(
            [
                FactorizedSTBlock(self.hidden, self.heads, mlp_ratio, dropout)
                for _ in range(self.depth)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden)
        self.out_proj = nn.Linear(self.hidden, self.token_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.env_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed_actions(
        self, actions: torch.Tensor, env_id: torch.Tensor
    ) -> torch.Tensor:
        names = [self.env_names[int(i)] for i in env_id.unique().tolist()]
        widths = {self.action_dims[name] for name in names}
        if len(widths) != 1:
            raise ValueError(
                'one batch cannot mix environments whose action widths differ'
            )
        width = next(iter(widths))
        if actions.shape[-1] != width:
            raise ValueError(
                f'action dim {actions.shape[-1]} does not match adapter width {width}'
            )
        out = actions.new_empty(actions.shape[0], actions.shape[1], self.hidden)
        for index, name in enumerate(self.env_names):
            take = env_id == index
            if not bool(take.any()):
                continue
            out[take] = self.adapters[name](actions[take])
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        env_id: torch.Tensor,
        use_action: bool = True,
    ) -> torch.Tensor:
        """Predict the token grid of the frame after each context frame.

        ``tokens`` is ``(B, H, P, D)`` and contains only frames already
        observed. ``actions`` is ``(B, H, A)``; ``actions[:, t]`` is applied
        at frame ``t``. ``env_id`` is ``(B,)`` long. The return value has the
        same shape as ``tokens`` and is the prediction of frame ``t + 1``.
        """
        if tokens.ndim != 4:
            raise ValueError(f'tokens must be (B, H, P, D), got {tuple(tokens.shape)}')
        b, h, p, _ = tokens.shape
        if p != self.num_patches:
            raise ValueError(
                f'expected {self.num_patches} patches, got {p}'
            )
        if actions.ndim != 3 or actions.shape[0] != b or actions.shape[1] != h:
            raise ValueError(
                f'actions must be (B, H, A) aligned with tokens, got {tuple(actions.shape)}'
            )
        if env_id.shape != (b,):
            raise ValueError(f'env_id must be ({b},), got {tuple(env_id.shape)}')
        env_id = env_id.to(device=tokens.device, dtype=torch.long)
        actions = actions.to(dtype=tokens.dtype)
        x = self.in_proj(tokens) + self.pos
        if use_action:
            x = x + self._embed_actions(actions, env_id)[:, :, None, :]
        x = x + self.env_embed(env_id)[:, None, None, :]
        mask = causal_time_mask(h, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.out_proj(self.out_norm(x))

    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        env_id: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive feature rollout for a later planner.

        G1 trains history 1, so each step conditions only on the latest
        frame. ``actions[:, t]`` is applied to the frame predicted at ``t-1``
        (or to the last observed frame when ``t`` is 0). Changing a later
        action cannot change an earlier prediction. This does not define a
        planning cost.
        """
        if context.ndim != 4 or actions.ndim != 3:
            raise ValueError('context is (B, H, P, D) and actions is (B, T, A)')
        current = context[:, -1:]
        preds = []
        for t in range(actions.shape[1]):
            pred = self.forward(current, actions[:, t : t + 1], env_id)
            preds.append(pred)
            current = pred
        return torch.cat(preds, dim=1)


class FrozenDINOv2(nn.Module):
    """DINOv2-S/14 patch encoder. ``train()`` cannot unfreeze it."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.freeze()

    @classmethod
    def from_pretrained(cls, weights_dir: str) -> 'FrozenDINOv2':
        from transformers import Dinov2Model

        backbone = Dinov2Model.from_pretrained(
            weights_dir, local_files_only=True
        )
        return cls(backbone)

    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Patch tokens only. ``pixels`` is ``(B, 3, H, W)`` or ``(B, T, 3, H, W)``."""
        self.freeze()
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.ndim != 5:
            raise ValueError(f'pixels must be NCHW or NTCHW, got {tuple(pixels.shape)}')
        b, t, c, h, w = pixels.shape
        flat = pixels.reshape(b * t, c, h, w)
        hidden = self.backbone(pixel_values=flat).last_hidden_state
        patches = hidden[:, 1:]
        return patches.reshape(b, t, patches.shape[1], patches.shape[2])
