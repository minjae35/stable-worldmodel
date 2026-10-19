"""Evaluation adapter for a gaoyuezhou/dino_wm PushT checkpoint.

The checkpoint is a ``VWorldModel`` (predictor + action encoder + proprio
encoder). The frozen DINOv2 encoder is not stored in the file. This module
reimplements that forward pass and the ``Dynamics`` surface used by
``ShootingCostEvaluator``. It loads the checkpoint weights as they are.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from einops import repeat
from torch import nn
from torchvision import transforms

from stable_worldmodel.wm.prejepa.module import Embedder


CKPT_DIR = Path('/workspace/model-checkpoints/point-cloud-pusht/dino-wm')
VISUAL_DIM = 384
PATCHES = 196
NUM_HIST = 3
PROPRIO_RAW_DIM = 1
PROPRIO_EMB_DIM = 10
ACTION_EMB_DIM = 10
LATENT_DIM = VISUAL_DIM + PROPRIO_EMB_DIM + ACTION_EMB_DIM  # 404


def _mask(n_patches: int, n_frames: int) -> torch.Tensor:
    zeros = torch.zeros(n_patches, n_patches)
    ones = torch.ones(n_patches, n_patches)
    rows = []
    for i in range(n_frames):
        rows.append(
            torch.cat([ones] * (i + 1) + [zeros] * (n_frames - i - 1), dim=1)
        )
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class _Attention(nn.Module):
    """Same block as ``models.vit.Attention`` in gaoyuezhou/dino_wm."""

    def __init__(self, dim, heads, dim_head, dropout, n_patches, n_frames):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )
        self.register_buffer('bias', _mask(n_patches, n_frames), persistent=False)

    def forward(self, x):
        _, tokens, _ = x.shape
        x = self.norm(x)
        q, k, v = map(
            lambda t: t.reshape(t.shape[0], tokens, self.heads, -1).transpose(1, 2),
            self.to_qkv(x).chunk(3, dim=-1),
        )
        # Same causal mask as the checkpoint. SDPA applies that mask without
        # materializing a dense score matrix; eval mode keeps dropout off.
        keep = self.bias[:, :, :tokens, :tokens].to(dtype=torch.bool)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        out = out.transpose(1, 2).reshape(x.shape[0], tokens, -1)
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, NUM_HIST * PATCHES, LATENT_DIM))
        self.dropout = nn.Dropout(0.0)
        layers = []
        for _ in range(6):
            layers.append(
                nn.ModuleList(
                    [
                        _Attention(LATENT_DIM, 16, 64, 0.1, PATCHES, NUM_HIST),
                        _FeedForward(LATENT_DIM, 2048, 0.1),
                    ]
                )
            )
        self.transformer = nn.Module()
        self.transformer.norm = nn.LayerNorm(LATENT_DIM)
        self.transformer.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(x)
        for attn, ff in self.transformer.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.transformer.norm(x)


class _LastFrameCost(nn.Module):
    """Last-frame visual patch MSE.

    The released noprop checkpoint still has a proprio slot, but training and
    planning feed it zeros and leave the proprio term out of the cost.
    """

    def forward(self, info):
        visual = info['predicted_pixels_emb'][:, :, -1:]
        goal_v = info['pixels_goal_emb']
        return (visual - goal_v).pow(2).mean(dim=tuple(range(2, visual.ndim)))


def _state_dict_from_pickled(module) -> dict[str, torch.Tensor]:
    out = {}

    def walk(mod, prefix):
        for key, value in getattr(mod, '_parameters', {}).items():
            if torch.is_tensor(value):
                out[prefix + key] = value.detach()
        for key, value in getattr(mod, '_modules', {}).items():
            if value is not None:
                walk(value, prefix + key + '.')

    walk(module, '')
    return out


def _load_raw(path: Path):
    class _Dummy:
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return type(name, (_Dummy,), {})

    class _Pickle:
        Unpickler = _Unpickler
        Pickler = pickle.Pickler
        dump = pickle.dump
        dumps = pickle.dumps
        load = pickle.load
        loads = pickle.loads

    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_Pickle)


class DinoWMAdapter(nn.Module):
    """Frozen DINO-WM checkpoint behind the planning ``rollout`` contract."""

    def __init__(self, ckpt_dir: Path = CKPT_DIR, chunk: int = 300):
        super().__init__()
        self.num_hist = NUM_HIST
        self.chunk = chunk
        self.resize = transforms.Resize(196)
        raw = _load_raw(ckpt_dir / 'checkpoints' / 'model_latest.pth')
        self.action_encoder = Embedder(in_chans=ACTION_EMB_DIM, emb_dim=ACTION_EMB_DIM)
        self.proprio_encoder = Embedder(in_chans=PROPRIO_RAW_DIM, emb_dim=PROPRIO_EMB_DIM)
        self.predictor = _Predictor()
        self.action_encoder.load_state_dict(_state_dict_from_pickled(raw['action_encoder']))
        self.proprio_encoder.load_state_dict(
            _state_dict_from_pickled(raw['proprio_encoder'])
        )
        missing, unexpected = self.predictor.load_state_dict(
            _state_dict_from_pickled(raw['predictor']), strict=False
        )
        missing = [key for key in missing if not key.endswith('bias')]
        if missing or unexpected:
            raise RuntimeError(
                f'predictor weight mismatch missing={missing} unexpected={unexpected}'
            )
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True
        )
        self.encoder.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        self._cached_pixels = None
        self._cached_visual = None

    def train(self, mode: bool = True):
        return super().train(False)

    def _images(self, pixels: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W). The policy transform matches upstream default_transform:
        # uint8 -> [0, 1], then Normalize(0.5, 0.5, 0.5). VWorldModel then
        # resizes 224 -> 196 so DINOv2 ViT-S/14 yields 14x14 patches.
        if pixels.shape[-1] == 3 and pixels.ndim == 4:
            pixels = pixels.permute(0, 3, 1, 2)
        return self.resize(pixels.float())

    @torch.no_grad()
    def _encode_visual(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels (B, T, C, H, W) -> (B, T, 196, 384)."""
        batch, time, _, _, _ = pixels.shape
        flat = self._images(pixels.reshape(batch * time, *pixels.shape[2:]))
        tokens = self.encoder.forward_features(flat)['x_norm_patchtokens']
        if tokens.shape[1] != PATCHES:
            raise RuntimeError(
                f'DINOv2 returned {tokens.shape[1]} patches, expected {PATCHES}'
            )
        return tokens.reshape(batch, time, PATCHES, VISUAL_DIM)

    def _encode_proprio(self, batch: int, time: int, device, dtype) -> torch.Tensor:
        # noprop arm: proprio_key is null, so every train and plan step feeds
        # a constant zero of shape (..., 1). The 10-d slot stays in the latent.
        raw = torch.zeros(batch, time, PROPRIO_RAW_DIM, device=device, dtype=dtype)
        return self.proprio_encoder(raw)

    def _fuse(self, visual, proprio, action):
        patches = visual.shape[2]
        proprio_t = repeat(proprio.unsqueeze(2), 'b t 1 d -> b t p d', p=patches)
        action_t = repeat(action.unsqueeze(2), 'b t 1 d -> b t p d', p=patches)
        return torch.cat([visual, proprio_t, action_t], dim=-1)

    def _predict_next(self, window: torch.Tensor) -> torch.Tensor:
        """window (B, S, T, P, D) -> next latent (B, S, 1, P, D)."""
        batch, samples, time, patches, dim = window.shape
        flat = window.reshape(batch * samples, time * patches, dim)
        pieces = []
        for start in range(0, flat.shape[0], self.chunk):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = self.predictor(flat[start : start + self.chunk])
            pieces.append(pred[:, -patches:].float())
        nxt = torch.cat(pieces, dim=0)
        return nxt.reshape(batch, samples, 1, patches, dim)

    def _replace_action(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # latent (B, S, T, P, D), action (B, S, T, 10)
        emb = self.action_encoder(action.reshape(-1, action.shape[-2], action.shape[-1]))
        emb = emb.reshape(*action.shape[:3], ACTION_EMB_DIM)
        tiled = repeat(emb.unsqueeze(3), 'b s t 1 d -> b s t p d', p=latent.shape[3])
        return torch.cat([latent[..., :-ACTION_EMB_DIM], tiled], dim=-1)

    @torch.no_grad()
    def rollout(self, info: dict, action_sequence: torch.Tensor) -> dict:
        pixels = info['pixels']
        context = pixels[:, 0]
        batch = context.shape[0]
        samples = action_sequence.shape[1]
        n_obs = context.shape[1]
        if (
            self._cached_pixels is not None
            and self._cached_visual.shape[1] == n_obs
            and torch.equal(self._cached_pixels, context)
        ):
            visual = self._cached_visual
        else:
            visual = self._encode_visual(context)
            self._cached_pixels = context.detach()
            self._cached_visual = visual
        proprio = self._encode_proprio(batch, n_obs, visual.device, visual.dtype)
        visual_s = visual.unsqueeze(1).expand(-1, samples, -1, -1, -1)
        proprio_s = proprio.unsqueeze(1).expand(-1, samples, -1, -1)

        past = info.get('action_history')
        if past is None:
            past = action_sequence.new_zeros(batch, samples, 0, action_sequence.shape[-1])
        act_0 = torch.cat([past, action_sequence[:, :, :1]], dim=2)
        zeros = visual_s.new_zeros(*visual_s.shape[:-1], ACTION_EMB_DIM)
        latent = torch.cat(
            [
                visual_s,
                repeat(proprio_s.unsqueeze(3), 'b s t 1 d -> b s t p d', p=PATCHES),
                zeros,
            ],
            dim=-1,
        )
        latent = self._replace_action(latent, act_0)

        future = action_sequence[:, :, 1:]
        for step in range(future.shape[2]):
            nxt = self._predict_next(latent[:, :, -self.num_hist :])
            nxt = self._replace_action(nxt, future[:, :, step : step + 1])
            latent = torch.cat([latent, nxt], dim=2)
        latent = torch.cat(
            [latent, self._predict_next(latent[:, :, -self.num_hist :])], dim=2
        )

        visual_out = latent[..., :VISUAL_DIM]
        proprio_out = latent[:, :, :, 0, VISUAL_DIM : VISUAL_DIM + PROPRIO_EMB_DIM]
        info['predicted_emb'] = visual_out
        info['predicted_pixels_emb'] = visual_out
        info['predicted_proprio_emb'] = proprio_out
        return info


def encode_goal(model: DinoWMAdapter, info: dict) -> torch.Tensor:
    goal = info['goal']
    # Solver-expanded goal is (B, samples, time, C, H, W). One frame is enough.
    if goal.ndim == 6:
        goal = goal[:, 0]
    if goal.ndim == 5:
        goal = goal[:, :1]
    elif goal.ndim == 4:
        goal = goal.unsqueeze(1)
    visual = model._encode_visual(goal)
    proprio = model._encode_proprio(visual.shape[0], 1, visual.device, visual.dtype)
    info['pixels_goal_emb'] = visual.unsqueeze(1)
    info['proprio_goal_emb'] = proprio.unsqueeze(1)
    info['goal_emb'] = visual
    return visual


def build_cost(model: DinoWMAdapter):
    from stable_worldmodel.planning import ShootingCostEvaluator

    return ShootingCostEvaluator(
        model, _LastFrameCost(), encode_goal=encode_goal
    )
