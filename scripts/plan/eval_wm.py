"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import json

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def episode_col(dataset):
    """Name of the episode-index column, robust across dataset formats.

    HDF5 lists every column (including index columns) in ``column_names``,
    but the Lance reader deliberately excludes its index columns
    (``episode_idx``/``step_idx``) from ``column_names`` and only exposes
    them via ``_schema_names``. Consult both so ``episode_idx`` is found
    regardless of format.
    """
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


def check_pusht_goal_alignment(dataset, episodes, starts, goal_offset):
    """Fail if the future dataset state is not the simulator goal.

    Also checks that a render of that state matches the goal image the
    planner will be given, including the goal marker baked into the
    dataset frames.
    """
    import gymnasium as gym

    from stable_worldmodel.world.world import (
        _apply_callables,
        _extract_init_goal,
    )

    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, list(episodes), list(starts), goal_offset
    )
    env = gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224)
    env.reset(seed=0)
    callables = [
        {
            'method': '_set_state',
            'args': {'state': {'value': 'state'}},
        },
        {
            'method': '_set_goal_state',
            'args': {'goal_state': {'value': 'goal_state'}},
        },
    ]
    stats = []
    try:
        for i in range(len(episodes)):
            _apply_callables(
                env.unwrapped,
                callables,
                {**init_rows[i], **goal_rows[i]},
            )
            got = np.asarray(
                env.unwrapped.goal_state, dtype=np.float64
            ).reshape(-1)
            target = np.asarray(
                goal_rows[i]['goal_state'], dtype=np.float64
            ).reshape(-1)
            if got.shape != target.shape or not np.allclose(
                got, target, rtol=0, atol=1e-4
            ):
                raise RuntimeError(
                    f'goal_state mismatch on pair {i}: sim {got} vs dataset {target}'
                )
            live = np.asarray(env.unwrapped.render())
            start_pix = np.asarray(init_rows[i]['pixels'])
            start_mae = float(
                np.abs(
                    live.astype(np.float32) - start_pix.astype(np.float32)
                ).mean()
            )
            env.unwrapped._set_state(target)
            goal_render = np.asarray(env.unwrapped.render())
            goal_pix = np.asarray(goal_rows[i]['goal'])
            goal_mae = float(
                np.abs(
                    goal_render.astype(np.float32)
                    - goal_pix.astype(np.float32)
                ).mean()
            )
            if start_mae > 8.0 or goal_mae > 8.0:
                raise RuntimeError(
                    f'goal image mismatch on pair {i}: '
                    f'start MAE {start_mae:.3f}, goal MAE {goal_mae:.3f}'
                )
            stats.append(
                {
                    'pair_index': i,
                    'start_render_mae': start_mae,
                    'goal_render_mae': goal_mae,
                    'goal_marker_pose': np.asarray(
                        env.unwrapped.goal_pose, dtype=np.float64
                    ).tolist(),
                    'goal_block_pose': target[2:5].tolist(),
                }
            )
    finally:
        env.close()
    return stats


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    # Same z-score as scripts/train/lewm.py (column_normalizer, fit before
    # the clip split). sklearn's StandardScaler is close but not the same
    # on this float32 table, so the planner would see a different action scale.
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = swm.data.ZScoreScaler()
        col_data = stats_dataset.get_col_data(col)
        if np.ndim(col_data) == 1:
            col_data = np.asarray(col_data).reshape(-1, 1)
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        objective = hydra.utils.instantiate(cfg.objective)
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    output_dir = cfg.eval.get('output_dir', None)
    if output_dir:
        results_path = Path(output_dir)
        video_path = results_path / 'videos'
    else:
        results_path = (
            Path(
                swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
                cfg.policy,
            ).parent
            if cfg.policy != 'random'
            else Path(__file__).parent
        )
        video_path = results_path

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: int(max_start_idx[i]) for i, ep_id in enumerate(ep_indices)
    }
    col_name = episode_col(dataset)
    g = np.random.default_rng(cfg.seed)
    heldout_fraction = float(cfg.eval.get('heldout_fraction', 0.0) or 0.0)
    split_info = {
        'dataset': str(cfg.eval.dataset_name),
        'official_test_split': False,
        'heldout_fraction': heldout_fraction,
    }

    if heldout_fraction > 0:
        # No published test table. Hold out the last fraction of episode
        # ids so each pair comes from a different episode than the others
        # and from the complement of the earliest ids. This is not the
        # clip-level split used in training (see the written report).
        order = np.sort(ep_indices)
        n_hold = max(1, int(len(order) * heldout_fraction))
        holdout = [
            ep for ep in order[-n_hold:] if max_start_idx_dict[ep] >= 0
        ]
        if len(holdout) < cfg.eval.num_eval:
            raise ValueError(
                'Not enough held-out episodes with sufficient length.'
            )
        picked = g.choice(len(holdout), size=cfg.eval.num_eval, replace=False)
        eval_episodes = []
        eval_start_idx = []
        for i in picked:
            ep = holdout[int(i)]
            start = int(g.integers(0, max_start_idx_dict[ep] + 1))
            eval_episodes.append(int(ep))
            eval_start_idx.append(start)
        eval_episodes = np.asarray(eval_episodes)
        eval_start_idx = np.asarray(eval_start_idx)
        split_info.update(
            {
                'method': 'episode holdout, one random start per episode',
                'n_episodes': int(len(order)),
                'n_holdout_episodes': int(len(holdout)),
                'holdout_episode_id_min': int(min(holdout)),
                'holdout_episode_id_max': int(max(holdout)),
                'limitation': (
                    'quentinll/lewm-pusht publishes only the training table '
                    '(pusht_expert_train). Training fits action normalization '
                    'on the full table and splits clips, not episodes. These '
                    'held-out episode ids are disjoint from the earliest '
                    f'{1 - heldout_fraction:.0%} of episode ids, but the '
                    'released checkpoint may still have been trained on them.'
                ),
            }
        )
        print(
            f'[eval] {len(holdout)} held-out episodes '
            f'(ids {split_info["holdout_episode_id_min"]}–'
            f'{split_info["holdout_episode_id_max"]})'
        )
    else:
        # Map each dataset row’s episode_idx to its max_start_idx
        max_start_per_row = np.array(
            [
                max_start_idx_dict[ep_id]
                for ep_id in dataset.get_col_data(col_name)
            ]
        )
        valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), 'valid starting points found for evaluation.')
        random_episode_indices = g.choice(
            len(valid_indices), size=cfg.eval.num_eval, replace=False
        )
        # sort increasingly to avoid issues with HDF5Dataset indexing
        random_episode_indices = np.sort(
            valid_indices[random_episode_indices]
        )
        print(random_episode_indices)
        eval_episodes = dataset.get_col_data(col_name)[
            random_episode_indices
        ]
        eval_start_idx = dataset.get_col_data('step_idx')[
            random_episode_indices
        ]
        split_info['method'] = 'random valid dataset rows'

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    print('[eval] pairs', list(zip(eval_episodes.tolist(), eval_start_idx.tolist())))

    goal_check = None
    if cfg.eval.get('check_goal_alignment', False):
        print('[eval] checking goal_state and goal image alignment')
        goal_check = check_pusht_goal_alignment(
            dataset,
            eval_episodes.tolist(),
            eval_start_idx.tolist(),
            cfg.eval.goal_offset_steps,
        )
        maes = [s['goal_render_mae'] for s in goal_check]
        print(
            f'[eval] goal image MAE mean {float(np.mean(maes)):.3f} '
            f'max {float(np.max(maes)):.3f}'
        )

    world.set_policy(policy)

    video_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {video_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=video_path,
            )
        print('Warmup done.')

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=video_path,
        )
    end_time = time.time()

    successes = np.asarray(metrics['episode_successes']).astype(bool)
    pairs = []
    for i, (ep, start, ok) in enumerate(
        zip(eval_episodes.tolist(), eval_start_idx.tolist(), successes)
    ):
        pairs.append(
            {
                'pair_index': i,
                'episode_idx': int(ep),
                'start_step': int(start),
                'goal_step': int(start) + int(cfg.eval.goal_offset_steps),
                'success': bool(ok),
                'video': str(video_path / f'env_{i}.mp4'),
            }
        )

    solve_times = [float(t) for t in getattr(policy, 'solve_times', [])]
    solve_batches = [int(n) for n in getattr(policy, 'solve_batches', [])]
    n_planned = int(sum(solve_batches)) if solve_batches else 0
    executed = getattr(policy, 'executed_actions', [])
    action_stats = None
    if executed:
        acts = np.concatenate(
            [
                np.asarray(a, dtype=np.float32).reshape(
                    -1, np.asarray(a).shape[-1]
                )
                for a in executed
            ],
            axis=0,
        )
        finite = acts[np.isfinite(acts).all(axis=1)]
        if len(finite):
            action_stats = {
                'n': int(len(finite)),
                'min': finite.min(axis=0).tolist(),
                'max': finite.max(axis=0).tolist(),
                'frac_at_bound': float(
                    np.any(np.abs(np.abs(finite) - 1.0) < 1e-5, axis=1).mean()
                ),
            }

    summary = {
        'seed': int(cfg.seed),
        'policy': str(cfg.policy),
        'plan_config': OmegaConf.to_container(cfg.plan_config, resolve=True),
        'solver': {
            'num_samples': int(cfg.solver.num_samples),
            'n_steps': int(cfg.solver.n_steps),
            'topk': int(cfg.solver.topk),
            'var_scale': float(cfg.solver.var_scale),
            'batch_size': int(cfg.solver.batch_size),
        },
        'eval_budget': int(cfg.eval.eval_budget),
        'goal_offset_steps': int(cfg.eval.goal_offset_steps),
        'split': split_info,
        'pairs': pairs,
        'success_rate': float(metrics['success_rate']),
        'n_success': int(successes.sum()),
        'n_pairs': int(len(pairs)),
        'evaluation_time_s': end_time - start_time,
        'planning_time_s': {
            'per_solve_call': solve_times,
            'envs_per_solve_call': solve_batches,
            'mean_solve_call_s': float(np.mean(solve_times)) if solve_times else None,
            'mean_per_env_replan_s': (
                float(sum(solve_times) / n_planned) if n_planned else None
            ),
        },
        'executed_action': action_stats,
        'goal_alignment': goal_check,
        'videos_dir': str(video_path.resolve()),
    }

    print(metrics)
    print(
        f'[eval] success {summary["n_success"]}/{summary["n_pairs"]} '
        f'({summary["success_rate"]:.1f}%)'
    )
    if solve_times:
        print(
            f'[eval] mean planning time per env replan '
            f'{summary["planning_time_s"]["mean_per_env_replan_s"]:.3f}s '
            f'({len(solve_times)} solve calls)'
        )
    print(f'[eval] videos saved to {video_path.resolve()}')

    results_path.mkdir(parents=True, exist_ok=True)
    report_path = results_path / cfg.output.filename
    with report_path.open('a') as f:
        f.write('\n')
        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')
        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')
        f.write(f'summary: {json.dumps(summary)}\n')

    if output_dir:
        metrics_path = results_path / 'metrics.json'
        with metrics_path.open('w') as f:
            json.dump(summary, f, indent=2)
        print(f'[eval] metrics saved to {metrics_path.resolve()}')


if __name__ == '__main__':
    run()
