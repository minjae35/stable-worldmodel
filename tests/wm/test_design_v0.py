"""Unit tests for the independent Design v0 one-step model core."""

from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm.design_v0 import (
    ActionEncoder,
    DesignV0Core,
    FrozenVisualEncoder,
    concatenate_latent_history,
)


LATENT_DIM = 4


class _StubBackbone(nn.Module):
    """Token backbone whose CLS token is a deterministic pixel slice."""

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


def _make_core(
    *,
    history_size: int = 2,
    max_action_dim: int = 3,
    num_environments: int = 2,
) -> DesignV0Core:
    return DesignV0Core(
        FrozenVisualEncoder(_StubBackbone()),
        history_size=history_size,
        max_action_dim=max_action_dim,
        action_embedding_dim=5,
        num_environments=num_environments,
        environment_embedding_dim=3,
        dynamics_hidden_dim=6,
    )


def test_visual_encoder_returns_frozen_final_cls_without_projection():
    backbone = _StubBackbone()
    encoder = FrozenVisualEncoder(backbone)
    pixels = torch.arange(
        2 * 3 * 1 * 2 * 2, dtype=torch.float32
    ).reshape(2, 3, 1, 2, 2)

    encoder.train()
    latents = encoder(pixels)
    expected = pixels.reshape(6, -1)[:, :LATENT_DIM].reshape(2, 3, -1)

    torch.testing.assert_close(latents, expected)
    assert latents.shape == (2, 3, LATENT_DIM)
    assert not latents.requires_grad
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_latent_history_is_concatenated_oldest_to_newest():
    latents = torch.tensor(
        [
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ]
    )

    state = concatenate_latent_history(latents, history_size=2)

    torch.testing.assert_close(state, torch.tensor([[2.0, 3.0, 4.0, 5.0]]))


def test_action_encoder_concatenates_padded_action_and_boolean_mask():
    encoder = ActionEncoder(max_action_dim=3, embedding_dim=2)
    action = torch.tensor([[0.25, -0.5, 0.0]])
    action_mask = torch.tensor([[True, True, False]])
    captured = []
    hook = encoder.linear.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )

    output = encoder(action, action_mask)
    hook.remove()

    expected_input = torch.tensor([[0.25, -0.5, 0.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(captured[0], expected_input)
    assert output.shape == (1, 2)


def test_environment_embedding_conditions_shared_dynamics():
    core = _make_core()
    latents = torch.zeros(2, 2, LATENT_DIM)
    action = torch.zeros(2, 3)
    action_mask = torch.ones(2, 3, dtype=torch.bool)
    env_id = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        core.environment_embedding.weight[0].zero_()
        core.environment_embedding.weight[1].fill_(1.0)
        for parameter in core.dynamics.parameters():
            parameter.zero_()
        core.dynamics.fc1.weight[0, -3:] = 1.0
        core.dynamics.fc2.weight[0, 0] = 1.0

    prediction = core.predict_next(
        latents, action, action_mask, env_id
    )

    assert prediction[0, 0] == 0
    assert prediction[1, 0] > 0
    assert not torch.equal(prediction[0], prediction[1])


def test_end_to_end_core_returns_one_residual_next_latent():
    core = _make_core(
        history_size=3,
        max_action_dim=7,
        num_environments=3,
    )
    pixels = torch.arange(
        3 * 3 * 1 * 2 * 2, dtype=torch.float32
    ).reshape(3, 3, 1, 2, 2)
    action = torch.zeros(3, 7)
    action_mask = torch.tensor(
        [
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    env_id = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        for parameter in core.dynamics.parameters():
            parameter.zero_()

    prediction = core(pixels, action, action_mask, env_id)
    current_latent = core.encode(pixels)[:, -1]

    assert prediction.shape == (3, LATENT_DIM)
    torch.testing.assert_close(prediction, current_latent)
