"""Shared RD-WM model for Experiment 1 variants A/B/C/D.

One class covers every variant (experiment-1.md Section 3):

- B (dense): shared encoder, per-env action adapter, env embedding, fully
  shared dynamics.
- C (rdwm): B plus per-env residual adapters on the outputs of dynamics
  blocks 5 and 6, with the up projection zero-initialized.
- D (pooled): B with the 64 projected latent tokens mean-pooled to 1 token.
- A (specialist): B built for a single env. Each env gets its own model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """``in -> hidden -> out`` with hidden LayerNorm + GELU, no output norm."""

    def __init__(self, dim_in: int, hidden: int, dim_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv = nn.Linear(dim, 3 * inner)
        self.proj = nn.Linear(inner, dim)
        self.dropout = dropout
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.proj_drop(self.proj(out))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, heads, dim // heads, 0.0)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    """Trainable ViT-Tiny. Returns the patch tokens without CLS."""

    def __init__(self, image_size, patch, width, depth, heads, mlp):
        super().__init__()
        self.grid = image_size // patch
        self.patch_embed = nn.Conv2d(3, width, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid * self.grid + 1, width)
        )
        self.blocks = nn.ModuleList(
            ViTBlock(width, heads, mlp) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 1:]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DynamicsBlock(nn.Module):
    """Transformer block with AdaLN conditioning; last modulation linear is zero."""

    def __init__(self, dim, heads, head_dim, ffn, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, dim),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def zero_adaln(self) -> None:
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, cond, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(cond).chunk(
            6, dim=-1
        )
        x = x + gate1 * self.attn(modulate(self.norm1(x), shift1, scale1), mask)
        return x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))


class ResidualAdapter(nn.Module):
    """``h + Linear(r, d)(GELU(Linear(d, r)(LN(h))))`` with a zero up projection."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)

    def zero_up(self) -> None:
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(self.norm(h))))


def frame_causal_mask(frames: int, tokens: int, device) -> torch.Tensor:
    """Boolean mask, True = may attend. Full attention inside a frame,
    no attention to later frames."""
    frame_id = torch.arange(frames, device=device).repeat_interleave(tokens)
    return frame_id[None, :] <= frame_id[:, None]


class Dynamics(nn.Module):
    def __init__(self, cfg: dict, tokens: int, envs: list[str], adapters: bool):
        super().__init__()
        dim = cfg['dim']
        self.tokens = tokens
        self.max_frames = cfg['max_frames']
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, tokens, dim))
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, self.max_frames, 1, dim)
        )
        self.blocks = nn.ModuleList(
            DynamicsBlock(
                dim,
                cfg['dyn_heads'],
                cfg['dyn_head_dim'],
                cfg['dyn_ffn'],
                cfg['dyn_dropout'],
            )
            for _ in range(cfg['dyn_depth'])
        )
        self.adapter_blocks = (
            [int(b) - 1 for b in cfg['adapter_blocks']] if adapters else []
        )
        self.adapters = nn.ModuleDict(
            {
                str(block): nn.ModuleDict(
                    {env: ResidualAdapter(dim, cfg['adapter_rank']) for env in envs}
                )
                for block in self.adapter_blocks
            }
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, z, cond, env: str):
        """z: [B, T, N, D] latents, cond: [B, T, D] per-frame conditioning."""
        b, t, n, d = z.shape
        if t > self.max_frames:
            raise ValueError(f'{t} frames > max_frames {self.max_frames}')
        x = z + self.spatial_pos + self.temporal_pos[:, :t]
        x = x.reshape(b, t * n, d)
        c = cond[:, :, None, :].expand(b, t, n, d).reshape(b, t * n, d)
        mask = frame_causal_mask(t, n, z.device)
        for index, block in enumerate(self.blocks):
            x = block(x, c, mask)
            if index in self.adapter_blocks:
                x = self.adapters[str(index)][env](x)
        return self.norm(x).reshape(b, t, n, d)


class ActionNorm(nn.Module):
    """Train-split action mean/std. Normalization happens here and only here."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(action_dim))
        self.register_buffer('std', torch.ones(action_dim))
        self.register_buffer('set', torch.zeros((), dtype=torch.bool))

    def load(self, mean, std) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        self.std.copy_(torch.as_tensor(std, dtype=self.std.dtype))
        self.set.fill_(True)

    def forward(self, actions):
        return (actions - self.mean) / self.std


class RDWM(nn.Module):
    def __init__(
        self,
        cfg: dict,
        variant: str,
        envs: list[str],
        action_dims: dict[str, int],
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        mcfg = cfg['model']
        vcfg = cfg['variants'][variant]
        self.variant = variant
        self.envs = list(envs)
        self.pooled = bool(vcfg['pooled'])
        self.action_block = mcfg['action_block']
        self.action_dims = {env: int(action_dims[env]) for env in envs}
        dim = mcfg['dim']

        # Default is the trainable ViT-Tiny. A frozen encoder is only for the
        # PushT prototype; Experiment 1 variants keep the ViT path.
        frozen_snapshot = None
        if encoder is None:
            self.encoder = ViTTiny(
                mcfg['image_size'],
                mcfg['patch_size'],
                mcfg['encoder_width'],
                mcfg['encoder_depth'],
                mcfg['encoder_heads'],
                mcfg['encoder_mlp'],
            )
            enc_width = mcfg['encoder_width']
            self.visual_encoder = 'vit'
        else:
            if not getattr(encoder, 'frozen', False):
                raise ValueError('a replacement encoder has to be frozen')
            self.encoder = encoder
            enc_width = int(encoder.width)
            self.visual_encoder = 'frozen_dinov2_small'
            frozen_snapshot = {
                k: v.detach().clone() for k, v in encoder.state_dict().items()
            }
        self.pool = mcfg['pool']
        grid = self.encoder.grid // self.pool
        self.spatial_tokens = grid * grid
        self.projector = MLP(enc_width, mcfg['projector_hidden'], dim)
        self.action_norm = nn.ModuleDict(
            {env: ActionNorm(self.action_dims[env]) for env in envs}
        )
        # Default is AdaLN. ``concat`` is the PushT ablation that feeds action
        # by concatenation, the way PreJEPA does, and leaves env on AdaLN.
        self.action_conditioning = str(mcfg.get('action_conditioning', 'adaln'))
        self.action_emb_dim = int(mcfg.get('action_emb_dim', 10))
        if self.action_conditioning == 'adaln':
            self.action_adapter = nn.ModuleDict(
                {
                    env: nn.Sequential(
                        nn.Linear(
                            self.action_block * self.action_dims[env],
                            mcfg['action_hidden'],
                        ),
                        nn.GELU(),
                        nn.Linear(mcfg['action_hidden'], dim),
                    )
                    for env in envs
                }
            )
        elif self.action_conditioning == 'concat':
            self.action_embed = nn.ModuleDict(
                {
                    env: nn.Linear(
                        self.action_block * self.action_dims[env],
                        self.action_emb_dim,
                    )
                    for env in envs
                }
            )
            self.input_proj = nn.Linear(dim + self.action_emb_dim, dim)
        else:
            raise ValueError(f'unknown action conditioning {self.action_conditioning}')
        self.env_embed = nn.ParameterDict(
            {env: nn.Parameter(torch.zeros(dim)) for env in envs}
        )
        self.dynamics = Dynamics(
            mcfg, 1 if self.pooled else self.spatial_tokens, envs, vcfg['adapters']
        )
        self.head = MLP(dim, mcfg['head_hidden'], dim)

        self.apply(_init_weights)
        for param in self.env_embed.values():
            nn.init.trunc_normal_(param, std=0.02)
        for block in self.dynamics.blocks:
            block.zero_adaln()
        for per_env in self.dynamics.adapters.values():
            for adapter in per_env.values():
                adapter.zero_up()
        if frozen_snapshot is not None:
            # apply() reinitializes every Linear. Put the pretrained encoder back.
            self.encoder.load_state_dict(frozen_snapshot)
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.encoder, 'frozen', False):
            self.encoder.eval()
        return self

    @property
    def tokens(self) -> int:
        return 1 if self.pooled else self.spatial_tokens

    def encode(self, pixels):
        """pixels: [B, T, 3, 224, 224] normalized -> latents [B, T, N, D]."""
        b, t = pixels.shape[:2]
        tokens = self.encoder(pixels.flatten(0, 1))
        width = tokens.shape[-1]
        grid = self.encoder.grid
        tokens = tokens.transpose(1, 2).reshape(-1, width, grid, grid)
        tokens = F.avg_pool2d(tokens, self.pool)
        tokens = tokens.flatten(2).transpose(1, 2)
        z = self.projector(tokens)
        if self.pooled:
            z = z.mean(dim=1, keepdim=True)
        return z.reshape(b, t, z.shape[1], z.shape[2])

    def _action_features(self, actions, env: str) -> torch.Tensor:
        """Native [B, T, block, d] -> normalized flattened block [B, T, block*d]."""
        flat = self.action_norm[env](actions.float()).flatten(-2)
        return flat.to(self.env_embed[env].dtype)

    def condition(self, actions, env: str):
        """actions: [B, T, 5, d_env] native -> [B, T, D] AdaLN conditioning.

        AdaLN mode adds the action adapter to the env embedding. Concat mode
        keeps only the env embedding here; the action enters by concatenation.
        """
        env_e = self.env_embed[env]
        if self.action_conditioning == 'concat':
            return env_e.view(1, 1, -1).expand(actions.shape[0], actions.shape[1], -1)
        action = self.action_adapter[env](self._action_features(actions, env))
        return action + env_e

    def _mix_action(self, z, actions, env: str) -> torch.Tensor:
        """Concatenate a 10-d action embedding onto each token and project back."""
        emb = self.action_embed[env](self._action_features(actions, env))
        tiled = emb[:, :, None, :].expand(-1, -1, z.shape[2], -1)
        return self.input_proj(torch.cat([z, tiled.to(z.dtype)], dim=-1))

    def predict_next(self, z, actions, env: str):
        """z: [B, T, N, D] window (T <= 3), actions: [B, T, 5, d] blocks
        leaving each window frame. Returns the next latent [B, N, D]."""
        if z.shape[1] != actions.shape[1]:
            raise ValueError(
                f'{z.shape[1]} frames but {actions.shape[1]} action blocks'
            )
        if self.action_conditioning == 'concat':
            z = self._mix_action(z, actions, env)
        h = self.dynamics(z, self.condition(actions, env), env)
        return self.head(h[:, -1])

    def rollout(self, z_hist, a_hist, a_future, env: str):
        """Recursive open-loop rollout used by training and CEM.

        z_hist: [B, H, N, D] observed latents (H >= 1).
        a_hist: [B, H-1, 5, d] executed blocks between those frames.
        a_future: [B, K, 5, d] blocks starting at the last observed frame.
        Returns predicted latents [B, K, N, D]. Predictions are fed back
        without detaching; the window keeps the last ``max_frames``.
        """
        frames = list(z_hist.unbind(dim=1))
        blocks = list(a_hist.unbind(dim=1)) + list(a_future.unbind(dim=1))
        limit = self.dynamics.max_frames
        preds = []
        for _ in range(a_future.shape[1]):
            n = len(frames)
            lo = max(0, n - limit)
            window = torch.stack(frames[lo:], dim=1)
            acts = torch.stack(blocks[lo:n], dim=1)
            nxt = self.predict_next(window, acts, env)
            preds.append(nxt)
            frames.append(nxt)
        return torch.stack(preds, dim=1)
