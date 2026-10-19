"""Stage 0 PushT diagnosis (10 pairs) and the 100-pair baseline.

The 10-pair run replays the original joint batch (one CEM generator, seed
42) so its trajectories match that evaluation. The 100-pair run is
sequential: each episode is seeded with ``42 + pair_index`` so a resume
skips finished episodes without replaying their random numbers or
counting them twice.

Long output stays in ``eval.log``. The terminal is the tqdm bar.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

warnings.filterwarnings('ignore', module='gymnasium')

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_wm  # noqa: E402
import stable_worldmodel as swm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ORIG_METRICS = ROOT / 'results' / 'stage0' / 'metrics.json'
DIAG_DIR = ROOT / 'results' / 'stage0' / 'diagnosis10'
EVAL_DIR = ROOT / 'results' / 'stage0' / 'eval100'
BUNDLED_PAIRS = ROOT / 'scripts' / 'plan' / 'config' / 'stage0_eval100_pairs.json'
SEED = 42
N_EVAL = 100
GOAL_OFFSET = 25
EVAL_BUDGET = 50
POS_THRESH = 20.0
ANGLE_THRESH = float(np.pi / 9)
HOLDOUT_FRACTION = 0.1
DATASET_NAME = 'hf://datasets/fracapuano/lewm-pusht/train.lance'
POLICY_NAME = 'quentinll/lewm-pusht'
LIMITATION = (
    'quentinll/lewm-pusht publishes only the training table '
    '(pusht_expert_train / hf://datasets/fracapuano/lewm-pusht/train.lance). '
    'There is no official test split. Training fits action normalization on '
    'the full table and splits clips, not episodes, so the released '
    'checkpoint may still have been trained on these held-out episode ids.'
)

CALLABLES = [
    {
        'method': '_set_state',
        'args': {'state': {'value': 'state', 'in_dataset': True}},
    },
    {
        'method': '_set_goal_state',
        'args': {'goal_state': {'value': 'goal_state', 'in_dataset': True}},
    },
]

PLAN = dict(
    horizon=5,
    receding_horizon=1,
    action_block=5,
    history_len=3,
    warm_start=True,
)

LOG = logging.getLogger('stage0')
CEM_LOG = logging.getLogger('stable_worldmodel.planning.solver.cem')


class _Cfg:
    cache_dir = None

    class dataset:
        keys_to_cache = ['action', 'proprio', 'state']

    class eval:
        img_size = 224


def _attach_log(path: Path) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    )
    LOG.addHandler(handler)
    CEM_LOG.addHandler(handler)
    return handler


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _point(err: dict) -> dict:
    out = {}
    for key, value in err.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def _series(trace: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(p[key]) for p in trace], dtype=np.float64)


def _failure_tag(row: dict) -> str:
    if row['full_ever']:
        return 'success'
    if row['object_ever']:
        return 'agent_position'
    if row['block_pos_min'] < POS_THRESH and row['block_angle_min'] >= ANGLE_THRESH:
        return 'block_angle'
    if row['block_angle_min'] < ANGLE_THRESH and row['block_pos_min'] >= POS_THRESH:
        return 'block_position'
    if row['block_pos_improvement_best'] <= 1.0:
        return 'no_block_progress'
    return 'partial_block_progress'


def summarize_episode(pair, trace, official_success, wall_s, solve_times, solve_batches):
    if len(trace) < 2:
        raise RuntimeError(
            f'pair {pair["pair_index"]} recorded {len(trace)} states'
        )
    block = _series(trace, 'block_pos')
    angle = _series(trace, 'block_angle')
    agent = _series(trace, 'agent_pos')
    object_flags = np.asarray(
        [bool(p['object_success']) for p in trace], dtype=bool
    )
    official_flags = np.asarray(
        [bool(p['official_success']) for p in trace], dtype=bool
    )
    criterion_mismatch = int(
        np.sum(
            [
                bool(p['official_success']) != bool(p['full_success'])
                for p in trace[1:]
            ]
        )
    )
    n_planned = int(sum(solve_batches)) if solve_batches else 0
    row = {
        'pair_index': int(pair['pair_index']),
        'episode_idx': int(pair['episode_idx']),
        'start_step': int(pair['start_step']),
        'goal_step': int(pair['start_step']) + GOAL_OFFSET,
        'full_ever': bool(official_success),
        'full_final': bool(official_flags[-1]),
        'object_ever': bool(object_flags.any()),
        'object_final': bool(object_flags[-1]),
        'block_pos_init': float(block[0]),
        'block_pos_final': float(block[-1]),
        'block_pos_min': float(block.min()),
        'block_pos_improvement_final': float(block[0] - block[-1]),
        'block_pos_improvement_best': float(block[0] - block.min()),
        'block_angle_init': float(angle[0]),
        'block_angle_final': float(angle[-1]),
        'block_angle_min': float(angle.min()),
        'block_angle_improvement_final': float(angle[0] - angle[-1]),
        'block_angle_improvement_best': float(angle[0] - angle.min()),
        'agent_pos_init': float(agent[0]),
        'agent_pos_final': float(agent[-1]),
        'agent_pos_min': float(agent.min()),
        'agent_pos_improvement_final': float(agent[0] - agent[-1]),
        'agent_pos_improvement_best': float(agent[0] - agent.min()),
        'n_steps': int(len(trace) - 1),
        'n_replans': int(len(solve_times)),
        'planning_time_s': float(sum(solve_times)),
        'mean_replan_s': (
            float(sum(solve_times) / n_planned) if n_planned else None
        ),
        'wall_s': float(wall_s),
        'criterion_mismatch': criterion_mismatch,
        'video': pair.get('video'),
    }
    row['failure_tag'] = _failure_tag(row)
    row['trace'] = [_point(p) for p in trace]
    return row


def _stat(values: np.ndarray) -> dict:
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def aggregate(rows: list[dict], evaluation_time_s: float) -> dict:
    rows = sorted(rows, key=lambda r: r['pair_index'])
    n = len(rows)

    def col(name):
        return np.asarray([r[name] for r in rows], dtype=np.float64)

    full_ever = col('full_ever') > 0
    full_final = col('full_final') > 0
    object_ever = col('object_ever') > 0
    object_final = col('object_final') > 0
    tags: dict[str, int] = {}
    for row in rows:
        tags[row['failure_tag']] = tags.get(row['failure_tag'], 0) + 1
    replans = col('planning_time_s')
    n_replans = col('n_replans')
    walls = col('wall_s')
    return {
        'n_pairs': n,
        'full_state_success_rate': float(full_ever.mean() * 100.0),
        'full_state_success_count': int(full_ever.sum()),
        'full_state_final_success_rate': float(full_final.mean() * 100.0),
        'full_state_final_success_count': int(full_final.sum()),
        'full_state_ever_equals_final': bool(np.array_equal(full_ever, full_final)),
        'object_only_success_rate_ever': float(object_ever.mean() * 100.0),
        'object_only_success_count_ever': int(object_ever.sum()),
        'object_only_success_rate_final': float(object_final.mean() * 100.0),
        'object_only_success_count_final': int(object_final.sum()),
        'failure_tags': tags,
        'goal_distance': {
            'block_pos_init': _stat(col('block_pos_init')),
            'block_pos_final': _stat(col('block_pos_final')),
            'block_pos_min': _stat(col('block_pos_min')),
            'block_pos_improvement_final': _stat(col('block_pos_improvement_final')),
            'block_pos_improvement_best': _stat(col('block_pos_improvement_best')),
            'block_angle_init': _stat(col('block_angle_init')),
            'block_angle_final': _stat(col('block_angle_final')),
            'block_angle_min': _stat(col('block_angle_min')),
            'block_angle_improvement_final': _stat(
                col('block_angle_improvement_final')
            ),
            'block_angle_improvement_best': _stat(col('block_angle_improvement_best')),
            'agent_pos_init': _stat(col('agent_pos_init')),
            'agent_pos_final': _stat(col('agent_pos_final')),
            'agent_pos_min': _stat(col('agent_pos_min')),
            'agent_pos_improvement_final': _stat(col('agent_pos_improvement_final')),
            'agent_pos_improvement_best': _stat(col('agent_pos_improvement_best')),
        },
        'planning_time_s': {
            'total': float(replans.sum()),
            'mean_episode': float(replans.mean()),
            'mean_replan': (
                float(replans.sum() / n_replans.sum()) if n_replans.sum() else None
            ),
            'n_replans': int(n_replans.sum()),
        },
        'evaluation_time_s': float(evaluation_time_s),
        'mean_episode_wall_s': float(walls.mean()),
        'criterion_mismatches': int(col('criterion_mismatch').sum()),
    }


def write_table(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r['pair_index'])
    fields = [k for k in rows[0].keys() if k != 'trace']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def _load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row['pair_index'])] = row
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _action_report(policy) -> dict:
    chunks = []
    for action in policy.executed_actions:
        arr = np.asarray(action, dtype=np.float64)
        chunks.append(arr.reshape(-1, arr.shape[-1]))
    if not chunks:
        return {'n_finite': 0, 'n_nonfinite_rows': 0, 'out_of_bounds': 0}
    acts = np.concatenate(chunks, axis=0)
    finite_mask = np.isfinite(acts).all(axis=1)
    finite = acts[finite_mask]
    out = 0
    if len(finite):
        out = int(np.any(np.abs(finite) > 1.0 + 1e-3, axis=1).sum())
    return {
        'n_finite': int(len(finite)),
        'n_nonfinite_rows': int((~finite_mask).sum()),
        'out_of_bounds': out,
        'min': finite.min(axis=0).tolist() if len(finite) else None,
        'max': finite.max(axis=0).tolist() if len(finite) else None,
    }


def _make_world(num_envs: int):
    return swm.World(
        env_name='swm/PushT-v1',
        num_envs=num_envs,
        max_episode_steps=EVAL_BUDGET * 2,
        image_shape=(224, 224),
    )


def _fit_process(dataset):
    process = {}
    for col in _Cfg.dataset.keys_to_cache:
        processor = swm.data.ZScoreScaler()
        col_data = dataset.get_col_data(col)
        if np.ndim(col_data) == 1:
            col_data = np.asarray(col_data).reshape(-1, 1)
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = processor
    return process


def _make_policy(process):
    model = swm.wm.utils.load_pretrained(POLICY_NAME)
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**PLAN)
    cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
    solver = swm.planning.CEMSolver(
        cost=cost,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device='cuda',
        seed=SEED,
    )
    transform = {
        'pixels': eval_wm.img_transform(_Cfg),
        'goal': eval_wm.img_transform(_Cfg),
    }
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


def _holdout_ids(dataset):
    col = eval_wm.episode_col(dataset)
    episode_idx = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data('step_idx'))
    order = np.argsort(episode_idx, kind='mergesort')
    ep_sorted = episode_idx[order]
    step_sorted = step_idx[order]
    unique, group_start = np.unique(ep_sorted, return_index=True)
    max_step = np.maximum.reduceat(step_sorted, group_start)
    # length = max(step_idx) + 1; last valid start is length - offset - 1.
    max_start = {
        int(ep): int(step - GOAL_OFFSET)
        for ep, step in zip(unique, max_step)
    }
    ep_order = np.sort(unique)
    n_hold = max(1, int(len(ep_order) * HOLDOUT_FRACTION))
    holdout = [int(ep) for ep in ep_order[-n_hold:] if max_start[int(ep)] >= 0]
    return holdout, max_start, len(ep_order)


def _sample_eval_pairs(dataset) -> list[dict]:
    path = EVAL_DIR / 'pairs.json'
    if path.exists():
        return json.loads(path.read_text())
    if BUNDLED_PAIRS.exists():
        payload = json.loads(BUNDLED_PAIRS.read_text())
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        _json_dump(path, payload)
        return payload
    original = json.loads(ORIG_METRICS.read_text())['pairs']
    holdout, max_start, n_episodes = _holdout_ids(dataset)
    used = {int(p['episode_idx']) for p in original}
    remaining = [ep for ep in holdout if ep not in used]
    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(remaining), size=N_EVAL - len(original), replace=False)
    pairs = []
    for src in original:
        pairs.append(
            {
                'pair_index': len(pairs),
                'episode_idx': int(src['episode_idx']),
                'start_step': int(src['start_step']),
                'source': 'original10',
            }
        )
    for index in picked:
        ep = int(remaining[int(index)])
        pairs.append(
            {
                'pair_index': len(pairs),
                'episode_idx': ep,
                'start_step': int(rng.integers(0, max_start[ep] + 1)),
                'source': 'holdout_seed42_excluding_original10',
            }
        )
    payload = {
        'seed': SEED,
        'n_episodes': n_episodes,
        'n_holdout': len(holdout),
        'holdout_episode_id_min': int(min(holdout)),
        'holdout_episode_id_max': int(max(holdout)),
        'sampling': (
            'First 10 pairs are the original seed-42 holdout sample, copied '
            'from results/stage0/metrics.json. The other 90 are drawn with a '
            'fresh np.random.default_rng(42) from the same last-10% episode '
            'ids, excluding those 10 episodes, one random start each. This is '
            'not a prefix of sampling 100 pairs from the original generator.'
        ),
        'cem_seed': (
            'Diagnosis uses one shared CEM generator seeded with 42, matching '
            'the original 10-pair batch. Each of the 100 episodes uses '
            'manual_seed(42 + pair_index) so results do not depend on order '
            'and a resume does not replay finished episodes.'
        ),
        'limitation': LIMITATION,
        'pairs': pairs,
    }
    _json_dump(path, payload)
    return pairs


def _config_payload(extra=None) -> dict:
    payload = {
        'seed': SEED,
        'policy': POLICY_NAME,
        'plan_config': PLAN,
        'solver': {
            'num_samples': 300,
            'n_steps': 30,
            'topk': 30,
            'var_scale': 1.0,
            'batch_size': 1,
            'device': 'cuda',
        },
        'goal_offset_steps': GOAL_OFFSET,
        'eval_budget': EVAL_BUDGET,
        'dataset': DATASET_NAME,
        'heldout_fraction': HOLDOUT_FRACTION,
        'official_test_split': False,
        'limitation': LIMITATION,
        'full_state_criterion': 'L2(agent_xy, block_xy) < 20 and block angle < pi/9',
        'object_only_criterion': 'L2(block_xy) < 20 and block angle < pi/9',
    }
    if extra:
        payload.update(extra)
    return payload


def _run_pairs(world, policy, dataset, pairs, video_dir, step_bar=None, episode_bar=None):
    video_dir.mkdir(parents=True, exist_ok=True)
    policy.reset_episode_state()
    episodes = [int(p['episode_idx']) for p in pairs]
    starts = [int(p['start_step']) for p in pairs]

    def _on_step(done_steps):
        if step_bar is not None:
            step_bar.n = min(done_steps, step_bar.total or done_steps)
            step_bar.refresh()
        if episode_bar is not None:
            k = len(policy.solve_times)
            episode_bar.set_postfix_str(f'replan {k}', refresh=False)

    def _on_plan(_solve_time, _n_envs):
        if episode_bar is None:
            return
        k = len(policy.solve_times) + 1
        episode_bar.set_postfix_str(f'replan {k}', refresh=False)

    policy.solver.progress_callback = _on_plan
    started = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=starts,
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        callables=CALLABLES,
        video=video_dir,
        record_trace=True,
        step_callback=_on_step,
    )
    wall = time.perf_counter() - started
    return metrics, wall


def run_diagnosis(policy, dataset) -> list[dict]:
    metrics_path = DIAG_DIR / 'metrics.json'
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        if saved.get('episodes') and len(saved['episodes']) == 10:
            LOG.info('diagnosis already complete, reusing %s', metrics_path)
            return saved['episodes'], saved.get('action_stats', {})

    original = json.loads(ORIG_METRICS.read_text())['pairs']
    pairs = [
        {
            'pair_index': int(p['pair_index']),
            'episode_idx': int(p['episode_idx']),
            'start_step': int(p['start_step']),
        }
        for p in original
    ]
    world = _make_world(len(pairs))
    world.set_policy(policy)
    policy.solver.torch_gen.manual_seed(SEED)
    video_dir = DIAG_DIR / 'videos'
    LOG.info('diagnosis batch start: %s pairs, shared CEM seed %s', len(pairs), SEED)
    bar = tqdm(
        total=EVAL_BUDGET,
        desc='LeWM PushT diagnosis',
        unit='step',
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )
    try:
        metrics, wall = _run_pairs(
            world, policy, dataset, pairs, video_dir, step_bar=bar
        )
    finally:
        bar.close()
        world.close()

    successes = np.asarray(metrics['episode_successes']).astype(bool)
    action_stats = _action_report(policy)
    rows = []
    for i, pair in enumerate(pairs):
        video = video_dir / f'env_{i}.mp4'
        pair = dict(pair)
        pair['video'] = str(video) if video.exists() else None
        row = summarize_episode(
            pair,
            metrics['traces'][i],
            bool(successes[i]),
            wall / len(pairs),
            policy.solve_times,
            policy.solve_batches,
        )
        # The batch shares one planner, so per-episode planning time is the
        # call time divided by the live batch. Keep the shared call totals
        # on the run, and the per-env share on the row.
        n_calls = max(len(policy.solve_times), 1)
        row['planning_time_s'] = float(sum(policy.solve_times) / len(pairs))
        row['n_replans'] = n_calls
        row['mean_replan_s'] = float(
            sum(policy.solve_times) / max(sum(policy.solve_batches), 1)
        )
        row['original_success'] = bool(original[i]['success'])
        rows.append(row)

    summary = aggregate(rows, wall)
    summary['original_vs_reproduced'] = [
        {
            'pair_index': int(row['pair_index']),
            'episode_idx': int(row['episode_idx']),
            'original_full_success': bool(row['original_success']),
            'reproduced_full_success': bool(row['full_ever']),
            'object_ever': bool(row['object_ever']),
            'object_final': bool(row['object_final']),
            'failure_tag': row['failure_tag'],
        }
        for row in rows
    ]
    summary['note'] = (
        'Full-state ever-success is the simulator OR of terminated flags. '
        'Wait mode freezes an env on that flag, so ever and final match '
        'whenever the success step was recorded. Planning time on each row '
        'is the shared batch time divided by 10; mean_replan is per env.'
    )
    payload = {
        'config': _config_payload(
            {
                'protocol': 'joint batch of the original 10 pairs',
                'cem_seed': SEED,
                'num_envs': 10,
            }
        ),
        'action_stats': action_stats,
        'original_success_count': int(sum(bool(p['success']) for p in original)),
        'reproduced_success_count': int(successes.sum()),
        'summary': summary,
        'episodes': rows,
    }
    _json_dump(metrics_path, payload)
    write_table(DIAG_DIR / 'episode_results.csv', rows)
    _json_dump(
        DIAG_DIR / 'progress.json',
        {'completed_pair_indices': [r['pair_index'] for r in rows], 'done': True},
    )
    LOG.info(
        'diagnosis full %s/%s object-ever %s/%s wall %.1fs',
        summary['full_state_success_count'],
        summary['n_pairs'],
        summary['object_only_success_count_ever'],
        summary['n_pairs'],
        wall,
    )
    LOG.info('diagnosis tags %s', summary['failure_tags'])
    LOG.info('diagnosis actions %s', action_stats)
    return rows, action_stats


def _fatal_problems(rows: list[dict], actions: dict) -> list[str]:
    problems = []
    if not actions or actions.get('n_finite', 0) == 0:
        problems.append('no finite executed actions')
    if actions.get('out_of_bounds', 0):
        problems.append(f'{actions["out_of_bounds"]} actions outside [-1, 1]')
    mismatches = sum(int(r['criterion_mismatch']) for r in rows)
    if mismatches:
        problems.append(f'{mismatches} official/full-state criterion mismatches')
    if any(not r['trace'] for r in rows):
        problems.append('empty state trace')
    return problems


def run_eval100(policy, dataset) -> None:
    pairs_payload = json.loads((EVAL_DIR / 'pairs.json').read_text())
    pairs = pairs_payload['pairs']
    if len(pairs) != N_EVAL:
        raise RuntimeError(f'expected {N_EVAL} pairs, found {len(pairs)}')
    jsonl_path = EVAL_DIR / 'episodes.jsonl'
    done = _load_jsonl(jsonl_path)
    video_dir = EVAL_DIR / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(EVAL_DIR / 'config.json', _config_payload({
        'protocol': 'sequential episodes, CEM seed = 42 + pair_index',
        'sampling': pairs_payload['sampling'],
        'cem_seed': pairs_payload['cem_seed'],
        'n_holdout': pairs_payload['n_holdout'],
        'holdout_episode_id_min': pairs_payload['holdout_episode_id_min'],
        'holdout_episode_id_max': pairs_payload['holdout_episode_id_max'],
    }))

    world = _make_world(1)
    world.set_policy(policy)
    started = time.perf_counter()
    # Credit already-finished episodes so ETA is over the full 100,
    # using their recorded wall times as the elapsed baseline.
    elapsed_credit = float(sum(r['wall_s'] for r in done.values()))
    bar = tqdm(
        total=N_EVAL,
        initial=len(done),
        desc='LeWM PushT Stage 0',
        unit='episode',
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.3,
        smoothing=0.1,
    )
    # tqdm's clock starts now. Shift it back by the time already spent
    # so a resume still shows elapsed time and ETA against all 100.
    bar.start_t -= elapsed_credit
    bar.last_print_t = time.time()
    try:
        for pair in pairs:
            idx = int(pair['pair_index'])
            if idx in done:
                continue
            policy.reset_episode_state()
            policy.solver.torch_gen.manual_seed(SEED + idx)
            ep_start = time.perf_counter()
            metrics, wall = _run_pairs(
                world,
                policy,
                dataset,
                [pair],
                video_dir,
                episode_bar=bar,
            )
            wall = time.perf_counter() - ep_start
            success = bool(np.asarray(metrics['episode_successes'])[0])
            src = video_dir / 'env_0.mp4'
            dst = video_dir / (
                f'pair_{idx:03d}_ep{int(pair["episode_idx"])}_'
                f's{int(pair["start_step"])}.mp4'
            )
            if src.exists():
                src.replace(dst)
            pair = dict(pair)
            pair['video'] = str(dst) if dst.exists() else None
            row = summarize_episode(
                pair,
                metrics['traces'][0],
                success,
                wall,
                list(policy.solve_times),
                list(policy.solve_batches),
            )
            actions = _action_report(policy)
            if actions['out_of_bounds'] or actions['n_finite'] == 0:
                raise RuntimeError(f'pair {idx} action check failed: {actions}')
            if row['criterion_mismatch']:
                raise RuntimeError(
                    f'pair {idx} official success disagrees with pose_errors'
                )
            done[idx] = row
            _append_jsonl(jsonl_path, row)
            _write_eval_snapshot(done, time.perf_counter() - started + elapsed_credit)
            n_full = sum(bool(r['full_ever']) for r in done.values())
            n_obj = sum(bool(r['object_ever']) for r in done.values())
            bar.update(1)
            bar.set_postfix_str(
                f'full {n_full}/{len(done)} object {n_obj}/{len(done)}',
                refresh=False,
            )
            LOG.info(
                'pair %s ep %s full=%s object_ever=%s object_final=%s '
                'tag=%s wall=%.2fs replans=%s',
                idx,
                pair['episode_idx'],
                int(row['full_ever']),
                int(row['object_ever']),
                int(row['object_final']),
                row['failure_tag'],
                wall,
                row['n_replans'],
            )
    finally:
        bar.close()
        world.close()

    _write_eval_snapshot(done, time.perf_counter() - started + elapsed_credit, final=True)
    LOG.info('eval100 finished %s/%s episodes', len(done), N_EVAL)


def _write_eval_snapshot(done: dict[int, dict], elapsed: float, final: bool = False) -> None:
    rows = [done[k] for k in sorted(done)]
    if not rows:
        return
    summary = aggregate(rows, elapsed)
    payload = {
        'config': _config_payload(
            {'protocol': 'sequential episodes, CEM seed = 42 + pair_index'}
        ),
        'complete': final and len(rows) == N_EVAL,
        'summary': summary,
        'episodes': rows,
    }
    _json_dump(EVAL_DIR / 'metrics.json', payload)
    write_table(EVAL_DIR / 'episode_results.csv', rows)
    _json_dump(
        EVAL_DIR / 'progress.json',
        {
            'completed_pair_indices': [r['pair_index'] for r in rows],
            'n_completed': len(rows),
            'n_total': N_EVAL,
            'complete': final and len(rows) == N_EVAL,
        },
    )
    if final:
        deduped = EVAL_DIR / 'episodes.jsonl'
        with deduped.open('w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')


def main() -> None:
    logging.getLogger('gymnasium').setLevel(logging.ERROR)
    logging.root.handlers.clear()
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    CEM_LOG.setLevel(logging.INFO)
    CEM_LOG.propagate = False
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _attach_log(DIAG_DIR / 'eval.log')
    _attach_log(EVAL_DIR / 'eval.log')
    print(
        'LeWM PushT Stage 0. Reattach with: tmux attach -t stage0_pusht',
        file=sys.stderr,
        flush=True,
    )
    LOG.info('loading dataset and model')
    dataset = swm.data.load_dataset(
        DATASET_NAME,
        keys_to_cache=list(_Cfg.dataset.keys_to_cache),
    )
    process = _fit_process(dataset)
    policy = _make_policy(process)
    _sample_eval_pairs(dataset)
    rows, action_stats = run_diagnosis(policy, dataset)
    problems = _fatal_problems(rows, action_stats)
    if problems:
        LOG.error('fatal problems, not starting 100-pair eval: %s', problems)
        raise SystemExit(2)
    LOG.info('no fatal protocol error, starting 100-pair eval')
    run_eval100(policy, dataset)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        LOG.error('stage0 failed\n%s', traceback.format_exc())
        raise
