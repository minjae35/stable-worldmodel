"""Unit tests for Design v0 recursive rollout and multi-step MSE."""

from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm.design_v0 import (
    DesignV0Core,
    DesignV0Objective,
    FrozenVisualEncoder,
    recursive_rollout,
    transition_actions,
)


LATENT_DIM = 4


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


def _make_core(*, history_size: int = 2, max_action_dim: int = 3) -> DesignV0Core:
    return DesignV0Core(
        FrozenVisualEncoder(_StubBackbone()),
        history_size=history_size,
        max_action_dim=max_action_dim,
        action_embedding_dim=5,
        num_environments=2,
        environment_embedding_dim=3,
        dynamics_hidden_dim=6,
    )


def _clip(*, batch: int, steps: int, max_action_dim: int = 3):
    pixels = torch.arange(
        batch * steps * 1 * 2 * 2, dtype=torch.float32
    ).reshape(batch, steps, 1, 2, 2)
    action = torch.zeros(batch, steps, max_action_dim)
    for time in range(steps):
        action[:, time, 0] = float(time)
    action_mask = torch.ones(batch, steps, max_action_dim, dtype=torch.bool)
    env_id = torch.zeros(batch, dtype=torch.long)
    return pixels, action, action_mask, env_id


def test_one_step_loss_matches_frozen_target_and_core_predict_next():
    core = _make_core(history_size=2)
    pixels, action, action_mask, env_id = _clip(batch=2, steps=3)
    objective = DesignV0Objective(core)

    output = objective(pixels, action, action_mask, env_id)
    latents = core.encode(pixels)
    expected = core.predict_next(
        latents[:, :2],
        action[:, 1],
        action_mask[:, 1],
        env_id,
    )
    expected_loss = (expected - latents[:, 2].detach()).pow(2).mean()

    assert output['predicted'].shape == (2, 1, LATENT_DIM)
    torch.testing.assert_close(output['predicted'][:, 0], expected)
    torch.testing.assert_close(output['target'][:, 0], latents[:, 2])
    torch.testing.assert_close(output['loss'], expected_loss)
    assert not output['target'].requires_grad


def test_recursive_rollout_feeds_previous_prediction_not_ground_truth():
    core = _make_core(history_size=2)
    pixels, action, action_mask, env_id = _clip(batch=1, steps=4)
    latents = core.encode(pixels)
    histories = []
    original = core.predict_next

    def wrapped(history, step_action, step_mask, step_env):
        histories.append(history.detach().clone())
        return original(history, step_action, step_mask, step_env)

    core.predict_next = wrapped
    predicted = recursive_rollout(
        core,
        latents[:, :2],
        action[:, 1:3],
        action_mask[:, 1:3],
        env_id,
    )

    assert len(histories) == 2
    torch.testing.assert_close(histories[0], latents[:, :2])
    torch.testing.assert_close(histories[1][:, :-1], histories[0][:, 1:])
    torch.testing.assert_close(histories[1][:, -1], predicted[:, 0])
    assert not torch.allclose(histories[1][:, -1], latents[:, 2])
    assert predicted.shape == (1, 2, LATENT_DIM)


def test_transition_actions_start_from_action_leaving_last_history():
    action = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    mask = torch.ones(1, 6, 1, dtype=torch.bool)

    aligned, aligned_mask = transition_actions(
        action, mask, history_size=3, horizon=2
    )

    torch.testing.assert_close(
        aligned, torch.tensor([[[2.0], [3.0]]])
    )
    assert aligned_mask.shape == (1, 2, 1)


def test_objective_consumes_actions_t_then_t_plus_one():
    core = _make_core(history_size=2)
    pixels, action, action_mask, env_id = _clip(batch=1, steps=4)
    seen = []
    original = core.predict_next

    def wrapped(history, step_action, step_mask, step_env):
        seen.append(step_action.detach().clone())
        return original(history, step_action, step_mask, step_env)

    core.predict_next = wrapped
    DesignV0Objective(core)(pixels, action, action_mask, env_id)

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], action[:, 1])
    torch.testing.assert_close(seen[1], action[:, 2])


def test_loss_is_mean_of_per_horizon_mse():
    core = _make_core(history_size=2)
    with torch.no_grad():
        for parameter in core.dynamics.parameters():
            parameter.zero_()

    pixels, action, action_mask, env_id = _clip(batch=2, steps=4)
    output = DesignV0Objective(core)(pixels, action, action_mask, env_id)
    latents = core.encode(pixels)
    current = latents[:, 1]
    expected_per_horizon = torch.stack(
        [
            (current - latents[:, 2]).pow(2).mean(),
            (current - latents[:, 3]).pow(2).mean(),
        ]
    )

    assert output['predicted'].shape == (2, 2, LATENT_DIM)
    torch.testing.assert_close(output['per_horizon_mse'], expected_per_horizon)
    torch.testing.assert_close(output['loss'], expected_per_horizon.mean())


def test_frozen_encoder_does_not_receive_gradients():
    core = _make_core(history_size=2)
    pixels, action, action_mask, env_id = _clip(batch=2, steps=4)
    output = DesignV0Objective(core)(pixels, action, action_mask, env_id)
    output['loss'].backward()

    encoder_grads = [
        parameter.grad
        for parameter in core.visual_encoder.parameters()
    ]
    dynamics_grads = [
        parameter.grad
        for parameter in core.dynamics.parameters()
        if parameter.requires_grad
    ]

    assert all(grad is None for grad in encoder_grads)
    assert any(grad is not None and torch.count_nonzero(grad) > 0 for grad in dynamics_grads)
    assert all(
        not parameter.requires_grad
        for parameter in core.visual_encoder.parameters()
    )
