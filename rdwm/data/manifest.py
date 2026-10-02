"""RD-WM Phase 1 environment manifest.

Values that can be read from the Hub or the simulator are recorded here
after they were measured. Replay validation reads this table; it does not
invent a second set of environment choices.

Episode split: every Hub repo below publishes a single ``train.lance``
artifact. There is no val or test table. The gate therefore holds out a
deterministic slice of that train table (see ``HOLDOUT``) and does not
treat the holdout as an official training split.
"""

from __future__ import annotations

HOLDOUT = {
    'fraction': 0.2,
    'num_starts': 20,
    'horizon': 5,
    'goal_offset': 25,
    'seed': 0,
    'rule': (
        'Last 20% of episode indices, 20 evenly spaced episodes, '
        'start step = min(length // 3, latest step with 5 finite '
        'outgoing actions still inside the episode).'
    ),
}

# Action alignment for every env in this gate: action[t] is applied at
# state[t] and produces state[t+1]. The PointMaze Lance conversion already
# stored that outgoing convention, so replay must not shift it again.
ACTION_ALIGNMENT = 'outgoing'

ENVS: dict[str, dict] = {
    'pusht': {
        'lance_uri': 'hf://datasets/fracapuano/lewm-pusht/train.lance',
        'hf_revision': '26983123e22d0a871e52c0886c0aaee48f55e77c',
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/PushT-v1',
        'env_kwargs': {
            'resolution': 224,
            'relative': True,
            'with_target': True,
            'render_mode': 'rgb_array',
        },
        'model_input_hw': (224, 224),
        'action_semantics': (
            '2D relative agent-target delta in [-1, 1], scaled by '
            'action_scale=100 into a PD position target. '
            'state is [agent_xy, block_xy, block_angle, agent_vel].'
        ),
        'goal_restore': (
            'Planning path _set_goal_state. Episode goal_state / goal_pose '
            'set the success target and the rendered T marker. The '
            'predicate check places the simulator on that goal state.'
        ),
        'success_predicate': (
            'PushT.eval_state: block/agent position L2 < 20 and '
            'symmetry-aware angle difference < pi/9.'
        ),
        'camera': (
            'PushT pygame canvas 512, downscaled to resolution=224. '
            'Goal marker follows goal_pose when render_goal=1 (default).'
        ),
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'state': 'state',
            'goal_state': 'goal_state',
            'goal_pose': 'goal_pose',
        },
        'restore': ('state', 'goal_state', 'goal_pose'),
    },
    'tworoom': {
        'lance_uri': 'hf://datasets/fracapuano/lewm-tworooms/train.lance',
        'hf_revision': 'a6dc0810c55cffb74500906735fa6e9ac323810c',
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/TwoRoom-v1',
        'env_kwargs': {'render_mode': 'rgb_array', 'render_target': False},
        'model_input_hw': (224, 224),
        'action_semantics': (
            '2D direction in [-1, 1], multiplied by agent speed '
            '(default 5 px/step) then collided with the wall.'
        ),
        'goal_restore': (
            '_set_goal_state from pos_target. Door centers and wall axis '
            'are restored from observation[4:10].'
        ),
        'success_predicate': 'terminated when ||agent - target|| < 16 px.',
        'camera': 'Fixed orthographic 224 canvas. Target dot is not drawn.',
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'pos_agent': 'pos_agent',
            'pos_target': 'pos_target',
            'observation': 'observation',
        },
        'restore': (
            'pos_agent',
            'pos_target',
            'observation door centers',
        ),
    },
    'cube': {
        'lance_uri': 'hf://datasets/fracapuano/lewm-cube/train.lance',
        'hf_revision': '3838cba3575393ccc4e0cf1571d425b1b9514b86',
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/OGBCube-v0',
        'env_kwargs': {
            'env_type': 'single',
            'ob_type': 'pixels',
            'multiview': False,
            'width': 224,
            'height': 224,
            'visualize_info': False,
            'terminate_at_goal': False,
            'mode': 'data_collection',
            'render_mode': 'rgb_array',
            # lewm-cube frames match the opaque arm. The OGBench default
            # (pixel_transparent_arm=True) leaves a ~7.8 MAE silhouette.
            'pixel_transparent_arm': False,
        },
        'model_input_hw': (224, 224),
        'action_semantics': (
            '5D end-effector delta, normalized to [-1, 1]: '
            'xyz, yaw, gripper. Native unnormalized ranges are '
            '+/- [0.05, 0.05, 0.05, 0.3, 1.0].'
        ),
        'goal_restore': (
            'set_target_pos(cube 0) from the future-step '
            'privileged block position and quaternion. Visual eval '
            'uses ob_type=pixels and the front_pixels camera.'
        ),
        'success_predicate': (
            'Cube within 0.04 m of its mocap target '
            '(CubeEnv._compute_successes).'
        ),
        'camera': 'OGBench front_pixels, 224x224, opaque arm.',
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'qpos': 'qpos',
            'qvel': 'qvel',
        },
        'restore': ('qpos', 'qvel'),
    },
    'reacher': {
        'lance_uri': 'hf://datasets/fracapuano/lewm-reacher/train.lance',
        'hf_revision': '852681f88696cd00b0c8cd6eb3baf5eddfba1a65',
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/ReacherDMControl-v0',
        'env_kwargs': {'task': 'qpos_match', 'render_mode': 'rgb_array'},
        'model_input_hw': (224, 224),
        'action_semantics': (
            '2D torque in [-1, 1] after the dm_control action_scale '
            'wrapper. DMControlWrapper.action_repeat is 2.'
        ),
        'goal_restore': (
            'set_target_qpos from the future qpos. render_target '
            'defaults to 0, so the target geom is transparent.'
        ),
        'success_predicate': (
            'ReacherQPosMatchTask.get_termination: every |qpos - target| '
            '< 0.05 rad.'
        ),
        'camera': 'dm_control reacher camera_id=0, render 224x224.',
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'qpos': 'qpos',
            'qvel': 'qvel',
        },
        'restore': ('qpos', 'qvel'),
    },
    'scene': {
        'lance_uri': 'hf://datasets/mjb4835/rdwm-ogbench-scene/train.lance',
        'hf_revision': None,
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/OGBScene-v0',
        'env_kwargs': {
            'ob_type': 'pixels',
            'multiview': False,
            'width': 64,
            'height': 64,
            'visualize_info': False,
            'terminate_at_goal': False,
            'mode': 'data_collection',
            'render_mode': 'rgb_array',
            'pixel_transparent_arm': True,
        },
        'model_input_hw': (224, 224),
        'image_native_hw': (64, 64),
        'action_semantics': (
            'Same 5D normalized EE delta as Cube: xyz, yaw, gripper.'
        ),
        'goal_restore': (
            'Future-step cube pose, discrete button states, drawer '
            'position, and window position via set_cube_target_pos / '
            'set_target_button_state / set_target_drawer_pos / '
            'set_target_window_pos.'
        ),
        'success_predicate': (
            'SceneEnv._compute_successes: cube <= 0.04 m, buttons '
            'equal the discrete target, drawer and window within 0.04.'
        ),
        'camera': (
            'OGBench front_pixels at the dataset native 64x64, '
            'transparent arm. Replay compares that native frame.'
        ),
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'qpos': 'qpos',
            'qvel': 'qvel',
            'button_states': 'button_states',
        },
        'restore': ('qpos', 'qvel', 'button_states'),
        'button_mapping': (
            'Lance button_states length 2 is the discrete state of '
            'button 0 and button 1. Replay casts each entry to int and '
            'passes button_state_0 / button_state_1 to set_state. '
            'A length-4 or (2, 2) one-hot is reduced with argmax per '
            'button. No other encoding is accepted.'
        ),
    },
    'pointmaze': {
        'lance_uri': 'hf://datasets/mjb4835/rdwm-pointmaze/train.lance',
        'hf_revision': None,
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'PointMaze_UMaze-v3',
        'env_kwargs': {
            'continuing_task': True,
            'reset_target': False,
            'render_mode': 'rgb_array',
        },
        'model_input_hw': (224, 224),
        'action_semantics': (
            '2D PointMaze force in [-1, 1]. Already stored in the '
            'outgoing convention; replay applies action[t] at state[t] '
            'with no extra shift.'
        ),
        'goal_restore': (
            'desired goal XY from the future [x, y]. Native reach '
            'predicate is ||xy - goal|| <= 0.45. Goal-marker visibility '
            'is chosen by the first-frame image comparison and recorded '
            'on the result; it is not a second action shift.'
        ),
        'success_predicate': (
            'gymnasium-robotics PointMaze: Euclidean XY distance <= 0.45.'
        ),
        'camera': (
            'gymnasium-robotics PointMaze_UMaze-v3 at the dataset frame '
            'size. Top-down camera: azimuth 180, elevation -90, '
            'distance 8.8, lookat origin. The oblique default view does '
            'not match these frames.'
        ),
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'proprio': 'proprio',
            'desired_goal': 'desired_goal',
        },
        'restore': ('x', 'y', 'vx', 'vy'),
        'extra_action_shift': False,
    },
    'antmaze': {
        'lance_uri': 'hf://datasets/mjb4835/rdwm-ogbench-antmaze/train.lance',
        'hf_revision': None,
        'splits': {'train': 'train.lance', 'val': None, 'test': None},
        'env_id': 'swm/OGBMaze-v0',
        'env_kwargs': {
            'loco_env_type': 'ant',
            'maze_env_type': 'maze',
            'ob_type': 'pixels',
            'width': 64,
            'height': 64,
            'render_mode': 'rgb_array',
            # Official visual-antmaze uses the back camera, not the
            # overhead view MazeEnv falls back to when camera_name is unset.
            'camera_name': 'back',
        },
        'model_input_hw': (224, 224),
        'image_native_hw': (64, 64),
        'action_semantics': (
            'Ant torque vector. Replay compares native 64x64 renders '
            'and does not resize them to the 224 model input.'
        ),
        'goal_restore': (
            'Future qpos[:2] is written to the OGBench navigate target. '
            'Success uses the env threshold on that XY.'
        ),
        'success_predicate': (
            'OGBench maze navigate success on agent XY vs the target.'
        ),
        'camera': (
            'OGBench visual antmaze back camera at native 64x64, with '
            'pixel floor encoding left on. maze_type is locked by the '
            'first-frame comparison.'
        ),
        'columns': {
            'pixels': 'pixels',
            'action': 'action',
            'qpos': 'qpos',
            'qvel': 'qvel',
        },
        'restore': ('qpos', 'qvel'),
        'compare_native_pixels': True,
    },
}
