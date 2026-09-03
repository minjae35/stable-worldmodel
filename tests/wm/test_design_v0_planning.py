"""Tests for the Design v0 planning Dynamics adapter."""

from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.design_v0 import (
    DesignV0Core,
    DesignV0PlanningAdapter,
    FrozenVisualEncoder,
    load_core_from_checkpoint,
    recursive_rollout,
)
from stable_worldmodel.wm.design_v0.planning import (
    env_id_from_metadata,
    resolve_planning_environment,
)


LATENT_DIM = 4
HISTORY_SIZE = 2
MAX_ACTION_DIM = 3


class _StubBackbone(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=latent_dim)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, pixels):
        cls = pixels.flatten(1)[:, : self.config.hidden_size] * self.scale
        patch = cls + 100
        return SimpleNamespace(
            last_hidden_state=torch.stack([cls, patch], dim=1)
        )


def _make_core() -> DesignV0Core:
    return DesignV0Core(
        FrozenVisualEncoder(_StubBackbone()),
        history_size=HISTORY_SIZE,
        max_action_dim=MAX_ACTION_DIM,
        action_embedding_dim=5,
        num_environments=1,
        environment_embedding_dim=3,
        dynamics_hidden_dim=6,
    )


def _adapter() -> DesignV0PlanningAdapter:
    return DesignV0PlanningAdapter(_make_core(), default_env_id=0)


def test_adapter_satisfies_dynamics_protocol():
    assert isinstance(_adapter(), Dynamics)


def test_encode_writes_emb_with_context_shape():
    adapter = _adapter()
    pixels = torch.randn(2, HISTORY_SIZE, 3, 2, 2)
    out = adapter.encode({'pixels': pixels})
    assert out['emb'].shape == (2, HISTORY_SIZE, LATENT_DIM)
    assert torch.isfinite(out['emb']).all()


def test_rollout_predicted_emb_is_context_plus_horizon():
    adapter = _adapter()
    batch, samples, context, horizon = 2, 3, 2, 4
    info = {
        'pixels': torch.randn(batch, samples, context, 3, 2, 2),
        'action_history': torch.zeros(
            batch, samples, context - 1, MAX_ACTION_DIM
        ),
    }
    candidates = torch.zeros(batch, samples, horizon, MAX_ACTION_DIM)
    out = adapter.rollout(info, candidates)
    assert out['predicted_emb'].shape == (
        batch,
        samples,
        context + horizon,
        LATENT_DIM,
    )
    assert torch.isfinite(out['predicted_emb']).all()
    torch.testing.assert_close(
        out['predicted_emb'][:, :, :context], out['emb']
    )


def test_rollout_uses_first_candidate_as_action_leaving_current():
    adapter = _adapter()
    seen = []
    original = adapter.core.predict_next

    def wrapped(history, action, mask, env_id):
        seen.append(action.detach().clone())
        return original(history, action, mask, env_id)

    adapter.core.predict_next = wrapped
    info = {
        'pixels': torch.randn(1, 1, 2, 3, 2, 2),
        'action_history': torch.ones(1, 1, 1, MAX_ACTION_DIM),
    }
    candidates = torch.arange(
        2 * MAX_ACTION_DIM, dtype=torch.float32
    ).reshape(1, 1, 2, MAX_ACTION_DIM)
    adapter.rollout(info, candidates)
    flat = candidates.reshape(2, MAX_ACTION_DIM)
    torch.testing.assert_close(seen[0], flat[:1])
    torch.testing.assert_close(seen[1], flat[1:2])


def test_short_context_is_left_padded_to_k():
    adapter = _adapter()
    histories = []
    original = adapter.core.predict_next

    def wrapped(history, action, mask, env_id):
        histories.append(history.detach().clone())
        return original(history, action, mask, env_id)

    adapter.core.predict_next = wrapped
    info = {'pixels': torch.randn(1, 1, 1, 3, 2, 2)}
    candidates = torch.zeros(1, 1, 1, MAX_ACTION_DIM)
    adapter.rollout(info, candidates)
    assert histories[0].shape == (1, HISTORY_SIZE, LATENT_DIM)
    torch.testing.assert_close(histories[0][:, 0], histories[0][:, 1])


def test_rollout_matches_recursive_rollout_on_encoded_history():
    adapter = _adapter()
    info = {
        'pixels': torch.randn(1, 1, 2, 3, 2, 2),
        'action_history': torch.zeros(1, 1, 1, MAX_ACTION_DIM),
    }
    candidates = torch.randn(1, 1, 3, MAX_ACTION_DIM)
    out = adapter.rollout(dict(info), candidates)
    latents = adapter.core.encode(info['pixels'][:, 0])
    flat = candidates.reshape(1, 3, MAX_ACTION_DIM)
    mask = torch.ones_like(flat, dtype=torch.bool)
    env_id = torch.zeros(1, dtype=torch.long)
    expected = recursive_rollout(
        adapter.core, latents, flat, mask, env_id
    )
    torch.testing.assert_close(out['predicted_emb'][0, 0, 2:], expected[0])


def test_mismatched_action_dim_is_rejected():
    adapter = _adapter()
    info = {'pixels': torch.randn(1, 1, 1, 3, 2, 2)}
    candidates = torch.zeros(1, 1, 2, MAX_ACTION_DIM + 1)
    try:
        adapter.rollout(info, candidates)
    except ValueError as exc:
        assert 'action last dim' in str(exc)
    else:
        raise AssertionError('expected action dim mismatch')


def test_shooting_cost_evaluator_returns_finite_cost():
    adapter = _adapter()
    info = {
        'pixels': torch.randn(2, 3, 2, 3, 2, 2),
        'goal': torch.randn(2, 3, 2, 3, 2, 2),
        'action': torch.zeros(2, 3, 2, MAX_ACTION_DIM),
        'action_history': torch.zeros(2, 3, 1, MAX_ACTION_DIM),
    }
    candidates = torch.zeros(2, 3, 2, MAX_ACTION_DIM)
    cost = ShootingCostEvaluator(adapter, GoalMSE()).get_cost(
        info, candidates
    )
    assert cost.shape == (2, 3)
    assert torch.isfinite(cost).all()


def test_load_core_from_checkpoint_restores_weights(tmp_path):
    core = _make_core()
    with torch.no_grad():
        core.dynamics.fc2.bias.fill_(0.25)
    state = {
        f'objective.core.{key}': value
        for key, value in core.state_dict().items()
    }
    metadata = {
        'env_names': ['TwoRoom'],
        'env_to_id': {'TwoRoom': 0},
        'backbone': 'stub',
        'history_size': HISTORY_SIZE,
        'max_action_dim': MAX_ACTION_DIM,
        'action_embedding_dim': 5,
        'environment_embedding_dim': 3,
        'dynamics_hidden_dim': 6,
    }
    path = tmp_path / 'last.ckpt'
    torch.save(
        {'state_dict': state, 'design_v0_metadata': metadata}, path
    )
    loaded, loaded_meta = load_core_from_checkpoint(
        path, visual_encoder=FrozenVisualEncoder(_StubBackbone())
    )
    torch.testing.assert_close(
        loaded.dynamics.fc2.bias, core.dynamics.fc2.bias
    )
    assert loaded_meta['env_names'] == ['TwoRoom']


def _joint_metadata():
    return {
        'env_names': ['TwoRoom', 'PushT', 'OGBCube'],
        'env_to_id': {'TwoRoom': 0, 'PushT': 1, 'OGBCube': 2},
        'effective_action_dims': {
            'TwoRoom': 10,
            'PushT': 10,
            'OGBCube': 25,
        },
        'max_action_dim': 25,
        'datasets': {
            'TwoRoom': {'effective_action_dim': 10},
        },
    }


def test_env_id_is_resolved_from_metadata_name():
    env_id = env_id_from_metadata(_joint_metadata(), 'TwoRoom')
    assert env_id == _joint_metadata()['env_to_id']['TwoRoom']
    name, resolved_id, action_dim = resolve_planning_environment(
        _joint_metadata(), 'TwoRoom'
    )
    assert name == 'TwoRoom'
    assert resolved_id == env_id
    assert action_dim == 10


def test_joint_resolve_requires_environment_name():
    try:
        resolve_planning_environment(_joint_metadata(), None)
    except ValueError as exc:
        assert 'environment is required' in str(exc)
    else:
        raise AssertionError('expected joint resolve to require a name')


def test_joint_candidates_are_zero_padded_with_leading_mask():
    core = DesignV0Core(
        FrozenVisualEncoder(_StubBackbone()),
        history_size=HISTORY_SIZE,
        max_action_dim=5,
        action_embedding_dim=5,
        num_environments=3,
        environment_embedding_dim=3,
        dynamics_hidden_dim=6,
    )
    adapter = DesignV0PlanningAdapter(
        core, default_env_id=2, action_dim=3
    )
    seen_action = []
    seen_mask = []
    seen_env = []
    original = adapter.core.predict_next

    def wrapped(history, action, mask, env_id):
        seen_action.append(action.detach().clone())
        seen_mask.append(mask.detach().clone())
        seen_env.append(env_id.detach().clone())
        return original(history, action, mask, env_id)

    adapter.core.predict_next = wrapped
    info = {'pixels': torch.randn(1, 1, 1, 3, 2, 2)}
    candidates = torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 3)
    adapter.rollout(info, candidates)

    padded = seen_action[0]
    mask = seen_mask[0]
    assert padded.shape[-1] == 5
    torch.testing.assert_close(padded[0, :3], candidates.reshape(3))
    torch.testing.assert_close(padded[0, 3:], torch.zeros(2))
    assert mask.tolist() == [[True, True, True, False, False]]
    assert seen_env[0].tolist() == [2]
