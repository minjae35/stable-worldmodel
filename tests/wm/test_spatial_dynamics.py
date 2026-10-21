"""Accuracy checks for the frozen-DINO spatial dynamics, before any training."""

import copy

import numpy as np
import pytest
import torch

from stable_worldmodel.wm.spatial.data import (
    apply_action_normalizer,
    blocked_action_dim,
    fit_action_normalizer,
    one_step_inputs,
    pack_blocked_action,
    select_episode_ids,
    split_episodes,
)
from stable_worldmodel.wm.spatial.metrics import moving_patch_errors, prediction_mse
from stable_worldmodel.wm.spatial.model import (
    FrozenDINOv2,
    SpatialDynamics,
    causal_time_mask,
)


class _Out:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _StubDino(torch.nn.Module):
    def __init__(self, dim=8, patches=16):
        super().__init__()
        self.proj = torch.nn.Linear(3, dim)
        self.patches = patches

    def forward(self, pixel_values):
        pooled = pixel_values.float().mean(dim=(2, 3))
        base = self.proj(pooled)
        offsets = torch.linspace(
            0, 1, 1 + self.patches, device=base.device, dtype=base.dtype
        )
        hidden = base[:, None, :] + offsets.view(1, -1, 1)
        return _Out(hidden)


def _tiny(**kwargs):
    cfg = dict(
        token_dim=8,
        hidden=32,
        depth=2,
        heads=4,
        grid=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    cfg.update(kwargs)
    return SpatialDynamics(**cfg)


def test_causal_mask_blocks_the_future_only():
    mask = causal_time_mask(3, torch.device('cpu'), torch.float32)
    assert mask.shape == (3, 3)
    assert torch.all(torch.diag(mask) == 0)
    assert mask[1, 0] == 0 and mask[2, 0] == 0 and mask[2, 1] == 0
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[0, 2])
    assert torch.isneginf(mask[1, 2])


def test_action_block_matches_dataset_layout_and_drops_the_future_block():
    raw = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    packed = pack_blocked_action(raw, num_steps=2, frameskip=5)
    assert packed.shape == (2, 10)
    assert np.allclose(packed[0], raw[:5].reshape(-1))
    assert np.allclose(packed[1], raw[5:].reshape(-1))
    tokens = torch.zeros(1, 2, 4, 8)
    actions = torch.from_numpy(packed).view(1, 2, 10)
    context, action_t, target = one_step_inputs(tokens, actions)
    assert context.shape == (1, 1, 4, 8)
    assert target.shape == (1, 1, 4, 8)
    assert action_t.shape == (1, 1, 10)
    assert torch.equal(action_t[0, 0], actions[0, 0])
    assert not torch.equal(action_t[0, 0], actions[0, 1])


def test_native_action_dims_and_train_only_normalization():
    assert blocked_action_dim(2, 5) == 10
    assert blocked_action_dim(5, 5) == 25
    model = _tiny()
    assert model.adapters['pusht'].in_dim == 10
    assert model.adapters['tworoom'].in_dim == 10
    assert model.adapters['cube'].in_dim == 25
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(64, 10)).astype(np.float32)
    heldout = rng.normal(loc=0.0, scale=1.0, size=(16, 10)).astype(np.float32)
    mean, std = fit_action_normalizer(train)
    scaled = apply_action_normalizer(train, mean, std)
    assert np.allclose(scaled.mean(0), 0.0, atol=1e-5)
    assert np.allclose(scaled.std(0), 1.0, atol=1e-5)
    held = apply_action_normalizer(heldout, mean, std)
    assert abs(float(held.mean())) > 0.5


def test_episode_split_is_disjoint():
    lengths = np.arange(100) + 20
    chosen = select_episode_ids(lengths, n_total=40, min_length=10, seed=3072)
    train, test = split_episodes(chosen, n_train=30, n_test=10, seed=3072)
    assert len(train) == 30 and len(test) == 10
    assert set(train.tolist()).isdisjoint(set(test.tolist()))
    assert len(np.unique(np.concatenate([train, test]))) == 40


def test_encoder_stays_frozen_through_train_and_backward():
    encoder = FrozenDINOv2(_StubDino())
    encoder.train()
    assert not encoder.training
    assert not encoder.backbone.training
    assert all(not p.requires_grad for p in encoder.parameters())
    before = encoder.backbone.proj.weight.detach().clone()
    pixels = torch.randn(2, 3, 8, 8)
    tokens = encoder.encode(pixels)
    assert tokens.shape == (2, 1, 16, 8)
    assert tokens.grad_fn is None
    model = _tiny()
    action = torch.randn(2, 1, 10)
    env = torch.zeros(2, dtype=torch.long)
    target = torch.randn(2, 1, 16, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss = prediction_mse(model(tokens, action, env), target)
    loss.backward()
    opt.step()
    assert encoder.backbone.proj.weight.grad is None
    assert torch.equal(encoder.backbone.proj.weight, before)


def test_env_id_and_action_change_the_prediction_through_the_right_adapter():
    torch.manual_seed(0)
    model = _tiny()
    tokens = torch.randn(2, 1, 16, 8)
    action = torch.randn(2, 1, 10)
    other = action + 3
    pusht = torch.zeros(2, dtype=torch.long)
    tworoom = torch.ones(2, dtype=torch.long)
    assert not torch.allclose(model(tokens, action, pusht), model(tokens, other, pusht))
    assert not torch.allclose(model(tokens, action, pusht), model(tokens, action, tworoom))
    for param in model.adapters['pusht'].parameters():
        param.data.zero_()
    assert torch.allclose(model(tokens, action, pusht), model(tokens, other, pusht))
    assert not torch.allclose(
        model(tokens, action, tworoom), model(tokens, other, tworoom)
    )


def test_no_future_frame_or_action_leaks_into_the_current_prediction():
    torch.manual_seed(1)
    model = _tiny()
    tokens = torch.randn(2, 2, 16, 8)
    actions = torch.randn(2, 2, 10)
    env = torch.zeros(2, dtype=torch.long)
    base = model(tokens, actions, env)
    changed_frame = tokens.clone()
    changed_frame[:, 1] = torch.randn_like(changed_frame[:, 1])
    leaked_frame = model(changed_frame, actions, env)
    assert torch.allclose(base[:, 0], leaked_frame[:, 0], atol=1e-5)
    assert not torch.allclose(base[:, 1], leaked_frame[:, 1], atol=1e-4)
    changed_action = actions.clone()
    changed_action[:, 1] = changed_action[:, 1] + 4
    leaked_action = model(tokens, changed_action, env)
    assert torch.allclose(base[:, 0], leaked_action[:, 0], atol=1e-5)
    assert not torch.allclose(base[:, 1], leaked_action[:, 1], atol=1e-4)


def test_spatial_attention_mixes_patches_inside_the_frame():
    torch.manual_seed(2)
    model = _tiny()
    tokens = torch.randn(1, 1, 16, 8)
    actions = torch.randn(1, 1, 10)
    env = torch.zeros(1, dtype=torch.long)
    base = model(tokens, actions, env)
    moved = tokens.clone()
    moved[:, :, 0] = moved[:, :, 0] + 5
    other = model(moved, actions, env)
    assert not torch.allclose(base[:, :, 1:], other[:, :, 1:], atol=1e-5)


def test_each_environment_sends_gradient_into_the_shared_trunk():
    torch.manual_seed(3)
    model = _tiny()
    tokens = torch.randn(2, 1, 16, 8)
    target = torch.randn(2, 1, 16, 8)
    widths = {0: 10, 1: 10, 2: 25}
    for env_id, width in widths.items():
        model.zero_grad(set_to_none=True)
        actions = torch.randn(2, 1, width)
        env = torch.full((2,), env_id, dtype=torch.long)
        prediction_mse(model(tokens, actions, env), target).backward()
        grad = model.blocks[0].spatial.in_proj_weight.grad
        adapter = model.adapters[model.env_names[env_id]].net[0].weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0
        assert adapter is not None and float(adapter.abs().sum()) > 0
    mixed = torch.tensor([0, 2])
    with pytest.raises(ValueError):
        model(tokens, torch.randn(2, 1, 10), mixed)


def test_rollout_is_causal_in_the_candidate_actions():
    torch.manual_seed(4)
    model = _tiny()
    context = torch.randn(2, 1, 16, 8)
    actions = torch.randn(2, 3, 10)
    env = torch.zeros(2, dtype=torch.long)
    out = model.rollout(context, actions, env)
    one = model(context, actions[:, :1], env)
    assert out.shape == (2, 3, 16, 8)
    assert torch.allclose(out[:, :1], one, atol=1e-5)
    later = actions.clone()
    later[:, 1:] = later[:, 1:] + 4
    shifted = model.rollout(context, later, env)
    assert torch.allclose(out[:, 0], shifted[:, 0], atol=1e-5)
    assert not torch.allclose(out[:, 1], shifted[:, 1], atol=1e-4)


def test_checkpoint_save_load_and_resume(tmp_path):
    torch.manual_seed(5)
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randn(4, 1, 16, 8)
    actions = torch.randn(4, 1, 10)
    env = torch.zeros(4, dtype=torch.long)
    target = torch.randn(4, 1, 16, 8)

    def step(dynamics, optim):
        optim.zero_grad(set_to_none=True)
        loss = prediction_mse(dynamics(tokens, actions, env), target)
        loss.backward()
        optim.step()

    step(model, opt)
    after_first = model(tokens, actions, env).detach()
    path = tmp_path / 'ckpt.pt'
    torch.save(
        {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': 1},
        path,
    )
    step(model, opt)
    continuous = model(tokens, actions, env).detach()

    restored = _tiny()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    blob = torch.load(path, weights_only=True)
    restored.load_state_dict(blob['model'])
    restored_opt.load_state_dict(blob['optimizer'])
    assert torch.allclose(restored(tokens, actions, env), after_first, atol=1e-6)
    step(restored, restored_opt)
    assert torch.allclose(
        restored(tokens, actions, env), continuous, atol=1e-6
    )
    assert blob['step'] == 1
    assert not any(
        a.data_ptr() == b.data_ptr()
        for a, b in zip(model.parameters(), restored.parameters())
    )
    _ = copy.deepcopy(blob)


def test_moving_patch_metric_isolates_the_patch_that_changes():
    current = torch.zeros(1, 4, 2)
    target = current.clone()
    target[0, 0] = 10
    pred = current.clone()
    stats = moving_patch_errors(pred, target, current)
    assert stats['moving_mse'] > stats['static_mse']
    assert stats['moving_mse'] == pytest.approx(stats['moving_persistence'])
    better = target.clone()
    better_stats = moving_patch_errors(better, target, current)
    assert better_stats['moving_mse'] < stats['moving_mse']
