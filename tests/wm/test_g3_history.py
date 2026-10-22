import numpy as np
import pytest
import torch

from stable_worldmodel.wm.spatial.g2 import tensor_state_sha256
from stable_worldmodel.wm.spatial.g3_history import (
    common_history_items,
    history_clip_rows,
    history_model_batch,
    normalize_action_history,
    select_history,
)
from stable_worldmodel.wm.spatial.model import SpatialDynamics
from stable_worldmodel.wm.spatial.model import fixed_sinusoidal_temporal_encoding


def _clip(start=10, row=110, length=40):
    return {'episode': 7, 'start': start, 'row': row, 'length': length}


def test_history_three_alignment_is_causal_and_target_shared_with_h1():
    h3_obs, h3_target, h3_actions = history_clip_rows(_clip(), 5, 3)
    h1_obs, h1_target, h1_actions = history_clip_rows(_clip(), 5, 1)
    assert h3_obs == [100, 105, 110]
    assert h3_actions == [
        [100, 101, 102, 103, 104],
        [105, 106, 107, 108, 109],
        [110, 111, 112, 113, 114],
    ]
    assert h1_obs == [110]
    assert h1_actions == [[110, 111, 112, 113, 114]]
    assert h1_target == h3_target == 115
    assert all(row < h3_target for block in h3_actions for row in block)


def test_history_three_rejects_episode_boundary_and_terminal_crossing():
    with pytest.raises(ValueError, match='lacks'):
        history_clip_rows(_clip(start=9), 5, 3)
    with pytest.raises(ValueError, match='terminal'):
        history_clip_rows(_clip(start=35, length=40), 5, 3)


def test_common_windows_filter_only_history_invalid_anchors():
    items = [_clip(start=5, row=105), _clip(start=10, row=110), _clip(start=15, row=115)]
    selected = common_history_items(items, 5, 3)
    assert [item['start'] for item in selected] == [10, 15]


def test_history_action_normalization_preserves_time_axis():
    actions = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    normalizer = {
        'mean': torch.arange(10, dtype=torch.float32),
        'std': torch.full((10,), 2.0),
    }
    normalized = normalize_action_history(actions, normalizer)
    assert normalized.shape == (2, 3, 10)
    assert torch.allclose(normalized[0, 0], (torch.arange(10) - torch.arange(10)) / 2)


def test_h1_and_h3_share_parameterization_and_initial_weights():
    torch.manual_seed(3072)
    h1 = SpatialDynamics()
    torch.manual_seed(3072)
    h3 = SpatialDynamics()
    assert sum(p.numel() for p in h1.parameters()) == sum(p.numel() for p in h3.parameters())
    assert tensor_state_sha256(h1.state_dict()) == tensor_state_sha256(h3.state_dict())


def test_h3_forward_backward_and_history_action_usage():
    torch.manual_seed(8)
    model = SpatialDynamics(hidden=32, depth=2, heads=4, grid=2)
    tokens = torch.randn(2, 3, 4, 384)
    actions = torch.randn(2, 3, 10)
    env_id = torch.zeros(2, dtype=torch.long)
    prediction = model(tokens, actions, env_id)
    assert prediction.shape == tokens.shape
    prediction[:, -1].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        no_history_actions = actions.clone()
        no_history_actions[:, :-1] = 0
        changed = model(tokens, no_history_actions, env_id)[:, -1]
    assert not torch.equal(prediction[:, -1].detach(), changed)


def test_history_order_perturbation_changes_last_prediction():
    torch.manual_seed(9)
    model = SpatialDynamics(hidden=32, depth=2, heads=4, grid=2).eval()
    tokens = torch.randn(2, 3, 4, 384)
    actions = torch.randn(2, 3, 10)
    env_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        original = model(tokens, actions, env_id)[:, -1]
        reordered_tokens = tokens[:, [1, 0, 2]]
        reordered_actions = actions[:, [1, 0, 2]]
        reordered = model(reordered_tokens, reordered_actions, env_id)[:, -1]
    assert not torch.equal(original, reordered)


def test_history_model_batch_uses_only_last_prediction_and_shared_target():
    torch.manual_seed(10)
    model = SpatialDynamics(hidden=32, depth=1, heads=4, grid=2)
    bundle = {
        'context': torch.randn(4, 3, 4, 384),
        'actions': torch.randn(4, 3, 10),
        'target': torch.randn(4, 4, 384),
    }
    _, _, _, target, prediction = history_model_batch(
        model, bundle, np.array([0, 2]), 0, torch.device('cpu')
    )
    assert target.shape == prediction.shape == (2, 1, 4, 384)
    h1 = select_history(bundle, 1)
    _, _, _, h1_target, h1_prediction = history_model_batch(
        model, h1, np.array([0, 2]), 0, torch.device('cpu')
    )
    assert torch.equal(target, h1_target)
    direct = model(h1['context'][[0, 2]], h1['actions'][[0, 2]], torch.zeros(2, dtype=torch.long))
    assert torch.equal(h1_prediction, direct[:, -1:])


def test_fixed_temporal_encoding_has_expected_shape_indices_and_no_parameters():
    positions = torch.tensor([0, 1, 2])
    encoding = fixed_sinusoidal_temporal_encoding(
        positions, 32, dtype=torch.float32
    )
    assert encoding.shape == (3, 32)
    assert torch.equal(encoding[0, 0::2], torch.zeros(16))
    assert torch.equal(encoding[0, 1::2], torch.ones(16))
    baseline = SpatialDynamics(hidden=32, depth=1, heads=4, grid=2)
    temporal = SpatialDynamics(
        hidden=32, depth=1, heads=4, grid=2,
        temporal_encoding='sinusoidal', temporal_encoding_scale=1.0,
    )
    assert list(baseline.state_dict()) == list(temporal.state_dict())
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in temporal.parameters()
    )


def test_temporal_encoding_is_causal_and_sensitive_to_explicit_time_indices():
    torch.manual_seed(11)
    model = SpatialDynamics(
        hidden=32, depth=2, heads=4, grid=2,
        temporal_encoding='sinusoidal', temporal_encoding_scale=1.0,
    ).eval()
    tokens = torch.randn(2, 3, 4, 384)
    actions = torch.randn(2, 3, 10)
    env_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        base = model(tokens, actions, env_id)
        shifted = model(
            tokens, actions, env_id, time_indices=torch.tensor([3, 4, 5])
        )
        future_changed = tokens.clone()
        future_changed[:, 2] += 10
        causal = model(future_changed, actions, env_id)
    assert not torch.allclose(base[:, -1], shifted[:, -1])
    assert torch.allclose(base[:, :2], causal[:, :2], atol=1e-5)


def test_temporal_encoding_forward_backward_is_finite():
    torch.manual_seed(12)
    model = SpatialDynamics(
        hidden=32, depth=2, heads=4, grid=2,
        temporal_encoding='sinusoidal', temporal_encoding_scale=1.0,
    )
    tokens = torch.randn(2, 3, 4, 384)
    actions = torch.randn(2, 3, 10)
    env_id = torch.zeros(2, dtype=torch.long)
    loss = model(tokens, actions, env_id)[:, -1].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
