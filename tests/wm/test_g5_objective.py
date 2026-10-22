import numpy as np
import pytest
import torch

from stable_worldmodel.wm.spatial.g5_objective import (
    G5_ARMS,
    detached_negative_ranking_loss,
    local_motion_weights,
    make_derangement_schedule,
    objective_components,
)


def _tokens(batch=2, dim=8):
    current = torch.zeros(batch, 1, 256, dim)
    target = current.clone()
    target[:, :, 17, :] = 1.0
    target[:, :, 34, :] = 4.0
    return current, target


def test_derangement_schedule_is_deterministic_and_has_no_fixed_points():
    a = make_derangement_schedule(11, 16, 501)
    b = make_derangement_schedule(11, 16, 501)
    assert np.array_equal(a, b)
    assert np.all(np.sort(a, axis=1) == np.arange(16))
    assert not np.any(a == np.arange(16)[None])


def test_local_motion_weights_are_tile_balanced_bounded_and_detached():
    current, target = _tokens()
    weights = local_motion_weights(current, target, blend=0.5, relative_cap=4.0)
    assert weights.shape == (2, 1, 256)
    assert not weights.requires_grad
    tiled = weights.reshape(2, 1, 4, 4, 4, 4).permute(0, 1, 2, 4, 3, 5)
    assert torch.allclose(tiled.reshape(2, 1, 4, 4, 16).mean(-1), torch.ones(2, 1, 4, 4))
    assert float(weights.min()) >= 0.0
    assert float(weights.max()) <= 2.0
    assert weights[0, 0, 17] > weights[0, 0, 0]


def test_detached_negative_never_pushes_the_negative_branch():
    target = torch.zeros(1, 1, 256, 4)
    correct = torch.ones_like(target, requires_grad=True)
    shuffled = torch.full_like(target, 0.5, requires_grad=True)
    loss, active = detached_negative_ranking_loss(correct, shuffled, target)
    loss.backward()
    assert active == 1
    assert correct.grad is not None and float(correct.grad.abs().sum()) > 0
    assert shuffled.grad is None


@pytest.mark.parametrize('arm', G5_ARMS)
def test_all_g5_objectives_have_finite_gradients(arm):
    current, target = _tokens()
    prediction = torch.randn_like(target, requires_grad=True)
    shuffled = torch.randn_like(target, requires_grad=True)
    result = objective_components(
        arm, prediction, target, current, shuffled,
        motion_blend=0.5, relative_cap=4.0, margin_fraction=0.05,
        ranking_lambda=0.2, objective_scale=1.0,
    )
    result['objective'].backward()
    assert torch.isfinite(result['objective'])
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    if arm in ('C_action_sensitive', 'D_combined'):
        assert shuffled.grad is None


def test_action_sensitive_arm_requires_shuffled_prediction():
    current, target = _tokens()
    with pytest.raises(ValueError, match='shuffled prediction'):
        objective_components(
            'C_action_sensitive', target, target, current, None,
            motion_blend=0.5, relative_cap=4.0, margin_fraction=0.05,
            ranking_lambda=1.0, objective_scale=1.0,
        )
