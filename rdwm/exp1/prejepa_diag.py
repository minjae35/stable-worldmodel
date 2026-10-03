"""DINO-WM / PreJEPA PushT parity on the RD-WM evaluator.

The baseline modules are not modified. This wrapper loads the public
checkpoint whose config is this repo's PreJEPA (frozen DINOv2-S, history 3,
one-step predictor, proprio + action) and scores it with the shipped
pixels+proprio GoalMSE. Candidates are the same native blocks the DINOv2
RD-WM prototype used.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.cost_diag import _load_pairs, _mean_metric, _rank_metrics, _sample_n
from rdwm.exp1.data import preprocess
from rdwm.exp1.diagnostics import val_clips
from rdwm.exp1.horizon_diag import _cfg_horizon
from rdwm.exp1.oracle_diag import _decision_prime, _elite_update, _sample
from rdwm.exp1.oracle_pool import _prime
from rdwm.exp1.planning import CEMPlanner, _pusht_goal_distance, open_episode_reader, open_eval_session
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import dataset_frame

REPO = 'kotmul/dinowm_patch_prop_pusht'
PROTO_JSON = Path('/workspace/rdwm_runs/exp1/logs/dino_proto/comparison_5k.json')


def _norm_from_stats(stats: dict, device: torch.device) -> dict:
    action_mean = np.asarray(stats['action']['mean'], dtype=np.float32).reshape(-1)
    action_std = np.maximum(np.asarray(stats['action']['std'], dtype=np.float32).reshape(-1), 1e-8)
    if 'proprio' in stats:
        proprio_mean = np.asarray(stats['proprio']['mean'], dtype=np.float32).reshape(-1)
        proprio_std = np.maximum(np.asarray(stats['proprio']['std'], dtype=np.float32).reshape(-1), 1e-8)
    else:
        proprio_mean = np.zeros(4, dtype=np.float32)
        proprio_std = np.ones(4, dtype=np.float32)
    return {
        'action_mean': torch.as_tensor(action_mean, device=device),
        'action_std': torch.as_tensor(action_std, device=device),
        'proprio_mean': torch.as_tensor(proprio_mean, device=device),
        'proprio_std': torch.as_tensor(proprio_std, device=device),
    }


def _prepare_model(model, device: torch.device):
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def load_dinowm(device: torch.device):
    """Load the checkpoint and the action/proprio stats stored beside it."""
    from hydra.utils import instantiate
    from stable_worldmodel.data.utils import get_cache_dir
    from stable_worldmodel.wm.utils import _resolve

    folder = get_cache_dir(sub_folder='checkpoints')
    checkpoint, config = _resolve(REPO, folder)
    config = dict(config)
    # This revision of PreJEPA does not take the checkpoint's bookkeeping key.
    config.pop('pixel_token', None)
    model = instantiate(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False))
    model = _prepare_model(model, device)
    folder = get_cache_dir(sub_folder='checkpoints') / f'models--{REPO.replace("/", "--")}'
    stats_path = folder / 'norm_stats.json'
    if not stats_path.exists():
        url = f'https://huggingface.co/{REPO}/resolve/main/norm_stats.json'
        urllib.request.urlretrieve(url, stats_path)
    stats = json.loads(stats_path.read_text())
    return model, _norm_from_stats(stats, device)


def load_local_dinowm(path: Path, device: torch.device):
    """Load a PreJEPA weights file plus the config and norm stats beside it."""
    from hydra.utils import instantiate

    weights = Path(path)
    folder = weights.parent
    config = json.loads((folder / 'config.json').read_text())
    config.pop('pixel_token', None)
    model = instantiate(config)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False))
    model = _prepare_model(model, device)
    stats = json.loads((folder / 'norm_stats.json').read_text())
    return model, _norm_from_stats(stats, device)


def _objective(cost: str = 'pixels_proprio'):
    from stable_worldmodel.planning import GoalMSE, WeightedSum

    # Mean reduction matches scripts/plan/config/objective/goal_mse_pixels_proprio.yaml.
    terms = [(1.0, GoalMSE('predicted_pixels_emb', 'pixels_goal_emb', reduction='mean'))]
    if cost == 'pixels_proprio':
        terms.insert(0, (1.0, GoalMSE('predicted_proprio_emb', 'proprio_goal_emb', reduction='mean')))
    elif cost != 'pixels':
        raise ValueError(f'unknown cost {cost}')
    return WeightedSum(terms)


def _evaluator(model, cost: str = 'pixels_proprio'):
    from stable_worldmodel.planning import ShootingCostEvaluator

    return ShootingCostEvaluator(model, _objective(cost))


def _proprio_np(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    return np.concatenate([state[:2], state[-2:]]).astype(np.float32)


def _norm_proprio(values: torch.Tensor, norm: dict) -> torch.Tensor:
    return (values - norm['proprio_mean']) / norm['proprio_std']


def _norm_actions(native: torch.Tensor, norm: dict) -> torch.Tensor:
    """[..., 5, 2] native -> [..., 10] z-scored block."""
    scaled = (native.float() - norm['action_mean']) / norm['action_std']
    return scaled.reshape(*native.shape[:-2], -1)


def _pixels_of(frames, cfg, device) -> torch.Tensor:
    stacked = np.stack([np.ascontiguousarray(frame) for frame in frames], axis=0)
    pixels = torch.from_numpy(stacked).permute(0, 3, 1, 2)
    return preprocess(pixels[:, None].to(device), cfg)[:, 0]


def _drop_cache(model) -> None:
    if hasattr(model, '_init_cached_info'):
        del model._init_cached_info


def _replace_proprio_embedding(model, mode: str) -> dict:
    """Swap the proprio embedder output. The checkpoint weights stay put.

    ``mean`` feeds the z-scored train mean, which is 0, through the frozen
    embedder. That layer is affine, so this is the train-set mean embedding.
    ``zero`` replaces that embedding with zeros.
    """
    if mode == 'original':
        return {'proprio_input': 'original'}
    if mode not in ('zero', 'mean'):
        raise ValueError(f'unknown proprio input {mode}')
    if 'proprio' not in model.extra_encoders:
        raise ValueError('checkpoint has no proprio embedder')
    module = model.extra_encoders['proprio']
    original = module.forward

    def forward(x):
        if mode == 'zero':
            return torch.zeros_like(original(x))
        return original(torch.zeros_like(x))

    module.forward = forward
    probe = torch.zeros(1, 1, module.in_chans, device=next(module.parameters()).device)
    with torch.no_grad():
        mean_emb = original(probe)
    return {
        'proprio_input': mode,
        'mean_embedding_max_abs': float(mean_emb.detach().abs().max()),
    }


@torch.no_grad()
def _outcome_cost(model, pixels, proprio, goal_pixels, goal_proprio, cost: str = 'pixels_proprio') -> torch.Tensor:
    """Encoded-outcome cost. ``pixels`` drops the proprio MSE and keeps patch MSE."""
    has_proprio = 'proprio' in model.extra_encoders
    obs = {'pixels': pixels[:, None]}
    goal_in = {'pixels': goal_pixels[:, None]}
    emb_keys: list[str] = []
    if has_proprio:
        obs['proprio'] = proprio[:, None]
        goal_in['proprio'] = goal_proprio[:, None]
        emb_keys = ['proprio']
    out = model.encode(obs, emb_keys=emb_keys)
    goal = model.encode(goal_in, emb_keys=emb_keys)
    pix = (out['pixels_emb'] - goal['pixels_emb']).float().square().mean(dim=(1, 2, 3))
    if cost == 'pixels':
        return pix
    pro = (out['proprio_emb'] - goal['proprio_emb']).float().square().mean(dim=(1, 2))
    return pix + pro


def _context_info(pixels, proprio, past, goal_pixels, goal_proprio, step: int):
    """pixels [T,3,H,W], proprio [T,4], past [T-1,10], goals [1,3,H,W] and [1,4]."""
    device = pixels.device
    return {
        'pixels': pixels[None, None],
        'proprio': proprio[None, None],
        'action_history': past[None],
        'goal': goal_pixels[None, None],
        'goal_proprio': goal_proprio[None, None],
        'id': torch.zeros(1, 1, 1, dtype=torch.long, device=device),
        'step_idx': torch.full((1, 1, 1), int(step), dtype=torch.long, device=device),
    }


@torch.no_grad()
def _predict_cost(evaluator, info, native, norm) -> torch.Tensor:
    """native [S, 1, 5, 2] -> cost [S]."""
    flat = _norm_actions(native[:, 0], norm)
    candidates = flat[None, :, None]
    past = info['action_history']
    info = dict(info)
    info['action_history'] = past[:, None].expand(
        past.shape[0], candidates.shape[1], past.shape[1], past.shape[2]
    ).contiguous()
    cost = evaluator.get_cost(info, candidates)
    return cost[0].float()


def _sim_candidates(session, cand, prime):
    frames, proprios, dists = [], [], []
    goal = np.asarray(prime['goal_state'], dtype=np.float64)
    blocks = cand.detach().cpu().numpy()[:, 0]
    for block in blocks:
        _prime(session, prime)
        for action in block:
            session.step(np.asarray(action, dtype=np.float64))
        frames.append(np.ascontiguousarray(session.render()))
        proprios.append(_proprio_np(session.state()))
        dists.append(float(np.linalg.norm(goal[:4] - session.state()[:4])))
    return frames, np.stack(proprios), np.asarray(dists, dtype=np.float64)


@torch.no_grad()
def _prediction(model, cfg, norm, device, n: int = 128) -> dict:
    import lance

    index, eps, starts = val_clips(cfg, 'pusht', n)
    table = lance.dataset(env_path(cfg, 'pusht'))
    base = index.offsets[eps] + starts
    obs_rows = base[:, None] + 5 * np.arange(5)[None]
    act_rows = base[:, None] + np.arange(20)[None]
    taken = table.take(obs_rows.reshape(-1).tolist(), columns=['pixels', 'state'])
    from rdwm.exp1.data import decode_jpegs

    frames = decode_jpegs(taken.column('pixels').to_pylist())
    states = np.asarray(taken.column('state').to_pylist(), dtype=np.float32).reshape(n, 5, -1)
    actions = np.asarray(index.actions[act_rows.reshape(-1)], dtype=np.float32).reshape(n, 4, 5, -1)
    pixels = preprocess(frames.reshape(n, 5, *frames.shape[1:]).to(device), cfg)
    proprio = _norm_proprio(torch.from_numpy(np.concatenate([states[..., :2], states[..., -2:]], axis=-1)).to(device), norm)
    native = torch.from_numpy(actions).to(device)
    blocks = _norm_actions(native, norm)

    pix_err = []
    pro_err = []
    both_err = []
    pix_persist = []
    both_persist = []
    log = Path('/workspace/rdwm_runs/exp1/logs/dinowm/pred_progress.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    bar = Progress(n, 0, 'DINO-WM pred | PushT', log)
    chunk = 8
    try:
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            b = pixels[sl].shape[0]
            _drop_cache(model)
            info = {
                'pixels': pixels[sl, :3][:, None],
                'proprio': proprio[sl, :3][:, None],
                'action_history': blocks[sl, :2][:, None],
                'id': torch.arange(b, device=device)[:, None, None],
                'step_idx': torch.full((b, 1, 1), 10_000 + start, dtype=torch.long, device=device),
            }
            rolled = model.rollout(info, blocks[sl, 2:4][:, None])
            pred_pix = rolled['predicted_pixels_emb'][:, 0].float()
            pred_pro = rolled['predicted_proprio_emb'][:, 0].float()
            encoded = model.encode(
                {'pixels': pixels[sl], 'proprio': proprio[sl]},
                emb_keys=['proprio'],
            )
            true_pix = encoded['pixels_emb'].float()
            true_pro = encoded['proprio_emb'].float()
            # Context is 3 frames. Horizon 2 appends t+1 then t+2 at the end.
            t1, t2 = -2, -1
            p1 = (pred_pix[:, t1] - true_pix[:, 3]).square().mean(dim=(1, 2))
            p2 = (pred_pix[:, t2] - true_pix[:, 4]).square().mean(dim=(1, 2))
            r1 = (pred_pro[:, t1] - true_pro[:, 3]).square().mean(dim=-1)
            r2 = (pred_pro[:, t2] - true_pro[:, 4]).square().mean(dim=-1)
            persist_p1 = (true_pix[:, 2] - true_pix[:, 3]).square().mean(dim=(1, 2))
            persist_p2 = (true_pix[:, 2] - true_pix[:, 4]).square().mean(dim=(1, 2))
            persist_b1 = persist_p1 + (true_pro[:, 2] - true_pro[:, 3]).square().mean(dim=-1)
            persist_b2 = persist_p2 + (true_pro[:, 2] - true_pro[:, 4]).square().mean(dim=-1)
            pix_err.append(torch.stack([p1, p2], dim=1))
            pro_err.append(torch.stack([r1, r2], dim=1))
            both_err.append(torch.stack([p1 + r1, p2 + r2], dim=1))
            pix_persist.append(torch.stack([persist_p1, persist_p2], dim=1))
            both_persist.append(torch.stack([persist_b1, persist_b2], dim=1))
            bar.set_postfix(f'clips {start + b}/{n}')
            bar.update(start + b)
    finally:
        bar.close()
        _drop_cache(model)

    def _cat(parts, index):
        return float(torch.cat(parts, dim=0)[:, index].mean())

    out = {}
    for label, err, persist in (
        ('pixels', pix_err, pix_persist),
        ('pixels_plus_proprio', both_err, both_persist),
    ):
        for step, index in (('t1', 0), ('t2', 1)):
            mse = _cat(err, index)
            base = _cat(persist, index)
            out[f'{label}_{step}'] = mse
            out[f'{label}_{step}_persist'] = base
            out[f'{label}_{step}_over_persist'] = mse / base if base else None
    out['n_clips'] = n
    out['proprio_t1'] = _cat(pro_err, 0)
    return out


def diagnose(cfg: dict, out: Path, device, workers: int = 16, n_candidates: int = 512, n_pairs: int = 10, seed0: int = 1000, cost: str = 'pixels_proprio', checkpoint: str | None = None, proprio_input: str = 'original') -> dict:
    torch.set_grad_enabled(False)
    if cost not in ('pixels_proprio', 'pixels'):
        raise ValueError(f'unknown cost {cost}')
    if checkpoint:
        model, norm = load_local_dinowm(Path(checkpoint), device)
        checkpoint_name = str(checkpoint)
    else:
        model, norm = load_dinowm(device)
        checkpoint_name = REPO
    proprio_info = _replace_proprio_embedding(model, proprio_input)
    has_proprio = 'proprio' in model.extra_encoders
    evaluator = _evaluator(model, cost)
    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pairs = _load_pairs(cfg, session, reader, 0, n_pairs)
    log = out.parent / ('dinowm_plan.log' if cost == 'pixels_proprio' else f'{out.stem}_progress.log')
    out.parent.mkdir(parents=True, exist_ok=True)
    prediction = None
    if cost == 'pixels_proprio':
        print('prediction | DINO-WM', flush=True)
        prediction = _prediction(model, cfg, norm, device)
    sampler = CEMPlanner(model, 'pusht', hcfg, session.action_low, session.action_high, device, seed0)
    rank_rows = []
    plan_rows = []
    label = cost if proprio_input == 'original' else f'{cost}/{proprio_input}'
    bar = Progress(n_pairs * n_candidates, 0, f'DINO-WM rank {label} | PushT', log, unit='cand/s')
    try:
        done = 0
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            start, goal_at = int(pair['start']), int(pair['goal'])
            session.restore(episode, start)
            session.set_goal(episode, goal_at)
            initial = _pusht_goal_distance(session)['pos']
            cand = _sample_n(sampler, n_candidates, seed0 + i)
            prime = _decision_prime(session, episode, start)
            frames, proprios, dists = _sim_candidates(session, cand, prime)
            goal_state = np.asarray(episode['state'][goal_at], dtype=np.float32)
            goal_pixels = _pixels_of([dataset_frame(episode, goal_at)], cfg, device)
            goal_proprio = _norm_proprio(torch.from_numpy(_proprio_np(goal_state))[None].to(device), norm)
            costs = []
            proprio_t = _norm_proprio(torch.from_numpy(proprios).to(device), norm)
            for start_at in range(0, len(frames), 32):
                chunk = frames[start_at : start_at + 32]
                pixels = _pixels_of(chunk, cfg, device)
                chunk_cost = _outcome_cost(
                    model, pixels, proprio_t[start_at : start_at + 32], goal_pixels, goal_proprio, cost
                )
                costs.append(chunk_cost.detach().cpu())
            pair_cost = torch.cat(costs).numpy()
            metrics = _rank_metrics(pair_cost, dists, initial)
            metrics.update({
                'episode': int(pair['episode']),
                'start': start,
                'goal': goal_at,
                'initial_distance': initial,
            })
            rank_rows.append(metrics)
            done += len(dists)
            bar.desc = f'DINO-WM rank {label} | PushT | pair {i + 1}/{n_pairs} | stage rank'
            bar.set_postfix(
                f"pair {i + 1}/{n_pairs} | stage rank | "
                f"spearman {metrics['spearman']:.3f} | best pct {metrics['best_distance_percentile']:.1f}"
            )
            bar.update(done)
        bar.close()

        rank_path = out.with_name('rank_pred.json' if cost == 'pixels_proprio' else f'{out.stem}_rank.json')
        rank_path.write_text(json.dumps({
            'checkpoint': checkpoint_name,
            'prediction': prediction,
            'rank_summary': {
                'mean_spearman': _mean_metric(rank_rows, 'spearman'),
                'mean_best_distance_percentile': _mean_metric(rank_rows, 'best_distance_percentile'),
                'mean_closer_rank_percentile': _mean_metric(rank_rows, 'closer_mean_rank_percentile'),
                'mean_n_closer': _mean_metric(rank_rows, 'n_closer'),
            },
            'rank_pairs': rank_rows,
        }, indent=2))
        print(f'wrote {rank_path}', flush=True)

        max_replans = int(cfg['eval']['exec_budget']) // int(cfg['model']['action_block'])
        plan_total = n_pairs * max_replans * int(cfg['cem']['n_steps'])
        bar = Progress(plan_total, 0, f'DINO-WM plan {label} | PushT', log, unit='cem/s')
        done_steps = 0

        def _tick():
            nonlocal done_steps
            done_steps += 1
            bar.update(done_steps)

        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            bar.desc = f'DINO-WM plan {label} | PushT | pair {i + 1}/{n_pairs} | stage plan'
            row, _used = _closed_loop(
                model, evaluator, session, episode, pair, hcfg, device, seed0 + i, cfg, norm,
                on_cem_step=_tick,
            )
            plan_rows.append(row)
            bar.set_postfix(
                f"pair {i + 1}/{n_pairs} | stage plan | success {int(row['success'])} | "
                f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
            )
            bar.update(done_steps)
        bar.close()
    finally:
        session.close()

    proto = json.loads(PROTO_JSON.read_text())
    if proprio_input == 'zero':
        note = (
            'Official checkpoint. Proprio embedding slots in the predictor input are zeros. '
            'Action embedding and pixels are unchanged. Planning and ranking costs are visual patch MSE.'
        )
    elif proprio_input == 'mean':
        note = (
            'Official checkpoint. Proprio embedding is replaced by the embedding of the '
            'z-scored train-set mean (the zero vector through the frozen affine embedder). '
            'Action embedding and pixels are unchanged. Planning and ranking costs are visual patch MSE.'
        )
    elif checkpoint and not has_proprio:
        note = (
            'PreJEPA trained with the official recipe except the proprio encoder is absent. '
            'Predictor input is RGB patch tokens plus action. '
            'Planning and ranking costs are visual patch MSE.'
        )
    else:
        note = (
            'galilai-group/dinowm-pusht and quentinll/dinowm-pusht are not on the Hub. '
            'kotmul/dinowm_patch_prop_pusht loads as stable_worldmodel.wm.PreJEPA with '
            'frozen DINOv2-S, history 3, num_pred 1, predictor depth 6 / heads 16 / mlp 2048, '
            'and proprio+action embedders. weights.pt is epoch 10, matching prejepa.yaml.'
        )
    report = {
        'checkpoint': checkpoint_name,
        'note': note,
        'protocol': {
            'pairs': n_pairs,
            'candidates': n_candidates,
            'seed0': seed0,
            'horizon_blocks': 1,
            'receding_blocks': 1,
            'cem': {
                'num_samples': int(cfg['cem']['num_samples']),
                'n_steps': int(cfg['cem']['n_steps']),
                'topk': int(cfg['cem']['topk']),
            },
            'cost': cost,
            'proprio_input': proprio_info,
            'rank_cost': (
                'encoded outcome visual patch mean MSE'
                if cost == 'pixels'
                else 'encoded outcome pixels+proprio mean MSE, same native candidates as the DINOv2 prototype'
            ),
            'plan_cost': (
                f'predicted visual patch mean MSE; proprio embedding replaced with {proprio_input}'
                if cost == 'pixels' and proprio_input != 'original'
                else 'predicted visual patch mean MSE; predictor input is pixels + action'
                if cost == 'pixels' and not has_proprio
                else 'predicted visual patch mean MSE; predictor forward still receives proprio'
                if cost == 'pixels'
                else 'predicted proprio mean MSE + predicted pixels mean MSE'
            ),
        },
        'prediction': prediction,
        'rank_summary': {
            'mean_spearman': _mean_metric(rank_rows, 'spearman'),
            'mean_best_distance_percentile': _mean_metric(rank_rows, 'best_distance_percentile'),
            'mean_closer_rank_percentile': _mean_metric(rank_rows, 'closer_mean_rank_percentile'),
            'mean_n_closer': _mean_metric(rank_rows, 'n_closer'),
        },
        'rank_pairs': rank_rows,
        'plan_summary': {
            'success_count': int(sum(row['success'] for row in plan_rows)),
            'goal_pos_initial': float(np.mean([row['goal_pos_initial'] for row in plan_rows])),
            'goal_pos_final': float(np.mean([row['goal_pos_final'] for row in plan_rows])),
            'goal_pos_reduction_pct': float(np.mean([row['goal_pos_reduction_pct'] for row in plan_rows])),
            'closer_count': int(sum(row['closer'] for row in plan_rows)),
        },
        'plan_pairs': plan_rows,
        'dinov2_rdwm_prototype': {
            'rank_summary': proto['dino_rank_plan']['rank_summary'],
            'plan_summary': proto['dino_rank_plan']['plan_summary'],
            'prediction': {
                'mse_t1_over_persist': proto['prediction']['frozen_dinov2_5k']['mse_t1_over_persist'],
                'mse_t2_over_persist': proto['prediction']['frozen_dinov2_5k']['mse_t2_over_persist'],
            },
        },
    }
    out.write_text(json.dumps(report, indent=2))
    _print(report)
    return report


def _goal_keys(model) -> tuple[str, ...]:
    keys = ['goal_emb', 'pixels_goal_emb']
    if 'proprio' in model.extra_encoders:
        keys.append('proprio_goal_emb')
    return tuple(keys)


def _attach_goal(model, info: dict) -> None:
    """Encode the goal once. Later CEM steps reuse the same tensors."""
    keys = _goal_keys(model)
    if all(key in info for key in keys):
        return
    from stable_worldmodel.planning.evaluator import split_goal_encode

    # action_history can be length 0 on the first frame. The goal encoder
    # never reads it, and slicing that axis raises.
    slim = {k: v for k, v in info.items() if k != 'action_history'}
    split_goal_encode(model, slim)
    for key in keys:
        info[key] = slim[key]


def _closed_loop(model, evaluator, session, episode, pair, cfg, device, seed, full_cfg, norm, on_cem_step=None):
    start, goal_at = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal_at)
    initial = _pusht_goal_distance(session)['pos']
    goal_pixels = _pixels_of([dataset_frame(episode, goal_at)], full_cfg, device)
    goal_state = np.asarray(episode['state'][goal_at], dtype=np.float32)
    goal_proprio = _norm_proprio(torch.from_numpy(_proprio_np(goal_state))[None].to(device), norm)
    planner = CEMPlanner(model, 'pusht', cfg, session.action_low, session.action_high, device, seed)
    hist_pix = [_pixels_of([session.render()], full_cfg, device)[0]]
    hist_pro = [_norm_proprio(torch.from_numpy(_proprio_np(session.state()))[None].to(device), norm)[0]]
    past: list[torch.Tensor] = []
    steps, success, replans = 0, False, 0
    budget = int(cfg['eval']['exec_budget'])
    history = int(cfg['cem']['history'])
    shape = (planner.horizon, planner.block, planner.low.shape[0])
    while steps < budget and not success:
        h = min(len(hist_pix), history)
        pixels = torch.stack(hist_pix[-h:], dim=0)
        proprio = torch.stack(hist_pro[-h:], dim=0)
        if h > 1:
            past_z = torch.stack(past[-(h - 1):], dim=0)
        else:
            past_z = pixels.new_zeros((0, planner.block * planner.low.shape[0]))
        _drop_cache(model)
        base = _context_info(pixels, proprio, past_z, goal_pixels, goal_proprio, replans)
        _attach_goal(model, base)
        mean = planner.center.expand(shape).clone()
        std = planner.init_std.expand(shape).clone()
        for _ in range(planner.iters):
            cand = _sample(planner, mean, std)
            cost = _predict_cost(evaluator, base, cand, norm)
            mean, std, _ = _elite_update(cand, cost, planner.topk)
            if on_cem_step is not None:
                on_cem_step()
        plan = planner.clamp(mean)
        replans += 1
        for action in plan[0].detach().cpu().numpy():
            session.step(np.asarray(action, dtype=np.float64))
            steps += 1
            if session.success():
                success = True
                break
            if steps >= budget:
                break
        if success or steps >= budget:
            break
        hist_pix.append(_pixels_of([session.render()], full_cfg, device)[0])
        hist_pro.append(_norm_proprio(torch.from_numpy(_proprio_np(session.state()))[None].to(device), norm)[0])
        past.append(_norm_actions(plan[0], norm))
    final = _pusht_goal_distance(session)['pos']
    reduction = (initial - final) / initial * 100.0 if initial > 0 else None
    return {
        'episode': int(pair['episode']),
        'start': start,
        'goal': goal_at,
        'success': bool(success),
        'goal_pos_initial': initial,
        'goal_pos_final': final,
        'goal_pos_reduction_pct': reduction,
        'closer': bool(final < initial),
        'steps': steps,
        'replans': replans,
    }, replans


def _print(report: dict) -> None:
    rank = report['rank_summary']
    plan = report['plan_summary']
    pred = report['prediction']
    base = report['dinov2_rdwm_prototype']
    print(
        f"DINO-WM rank | spearman {rank['mean_spearman']:.3f} | "
        f"best pct {rank['mean_best_distance_percentile']:.2f} | "
        f"closer rank {rank['mean_closer_rank_percentile']:.2f}",
        flush=True,
    )
    print(
        f"DINO-WM plan | success {plan['success_count']}/10 | closer {plan['closer_count']}/10 | "
        f"dist {plan['goal_pos_initial']:.1f}->{plan['goal_pos_final']:.1f}",
        flush=True,
    )
    if pred:
        print(
            f"DINO-WM pred | pixels t1/persist {pred['pixels_t1_over_persist']:.3f} | "
            f"combined t1/persist {pred['pixels_plus_proprio_t1_over_persist']:.3f}",
            flush=True,
        )
    br, bp = base['rank_summary'], base['plan_summary']
    print(
        f"DINOv2+RD-WM | spearman {br['mean_spearman']:.3f} | "
        f"best pct {br['mean_best_distance_percentile']:.2f} | "
        f"closer rank {br['mean_closer_rank_percentile']:.2f} | "
        f"success {bp['success_count']}/10 | closer {bp['closer_count']}/10 | "
        f"dist {bp['goal_pos_initial']:.1f}->{bp['goal_pos_final']:.1f}",
        flush=True,
    )
