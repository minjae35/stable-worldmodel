"""Frozen DINOv2 spatial world model.

The LeWM trunk is not reused. Planning costs are intentionally not defined
here: uniform patch MSE is only the training loss.
"""

from stable_worldmodel.wm.spatial.data import (
    blocked_action_dim,
    clip_start,
    fit_action_normalizer,
    apply_action_normalizer,
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

__all__ = [
    'FrozenDINOv2',
    'SpatialDynamics',
    'apply_action_normalizer',
    'blocked_action_dim',
    'causal_time_mask',
    'clip_start',
    'fit_action_normalizer',
    'moving_patch_errors',
    'one_step_inputs',
    'pack_blocked_action',
    'prediction_mse',
    'select_episode_ids',
    'split_episodes',
]
