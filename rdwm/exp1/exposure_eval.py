"""Ranking plus horizon-1 CEM on one exposure-matched PushT checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.cost_diag import rank_diagnose
from rdwm.exp1.evaluate import get_pairs
from rdwm.exp1.horizon_diag import _cfg_horizon, _closed_loop
from rdwm.exp1.lewm_diag import diagnose as lewm_diagnose
from rdwm.exp1.planning import open_episode_reader, open_eval_session
from rdwm.exp1.progress import Progress


def _notify(text: str) -> None:
    import os
    import urllib.request

    url = os.environ.get('SLACK_WEBHOOK_URL')
    if not url:
        print('slack unset', flush=True)
        return
    req = urllib.request.Request(
        url,
        data=json.dumps({'text': text}).encode(),
        headers={'Content-Type': 'application/json'},
    )
    urllib.request.urlopen(req, timeout=10).read()


def evaluate_rdwm(cfg: dict, ckpt: Path, out: Path, device, workers: int = 16) -> dict:
    rank_path = out.with_name(out.stem + '_rank.json')
    rank = rank_diagnose(
        cfg, ckpt, rank_path, device, workers=workers, n_candidates=512, n_pairs=10
    )
    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    from rdwm.exp1.evaluate import load_model

    model = load_model(cfg, ckpt, device)
    pairs = get_pairs(cfg, 'pusht', 'val', cfg['eval']['pilot_val_pairs'], session, reader)[:10]
    bar = Progress(len(pairs), 0, 'RD-WM CEM | PushT', out.parent / (out.stem + '_plan.log'))
    rows = []
    try:
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            row = _closed_loop(model, session, episode, pair, hcfg, device, 1000 + i)
            rows.append(row)
            bar.set_postfix(
                f"pair {i + 1}/10 | stage plan | success {int(row['success'])} | "
                f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
            )
            bar.update(i + 1)
    finally:
        bar.close()
        session.close()
    plan = {
        'success_count': int(sum(r['success'] for r in rows)),
        'goal_pos_initial': float(np.mean([r['goal_pos_initial'] for r in rows])),
        'goal_pos_final': float(np.mean([r['goal_pos_final'] for r in rows])),
        'goal_pos_reduction_pct': float(np.mean([r['goal_pos_reduction_pct'] for r in rows])),
        'closer_count': int(sum(r['closer'] for r in rows)),
        'pairs': rows,
    }
    mean = rank['costs']['mean']
    report = {
        'model': 'rdwm',
        'ckpt': str(ckpt),
        'rank_summary': {k: mean[k] for k in mean if k != 'pairs'},
        'plan_summary': {k: plan[k] for k in plan if k != 'pairs'},
        'plan_pairs': rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"RD-WM rank spearman {report['rank_summary']['mean_spearman']:.3f} | "
        f"best pct {report['rank_summary']['mean_best_distance_percentile']:.2f} | "
        f"plan {plan['success_count']}/10",
        flush=True,
    )
    return report


def evaluate_lewm(cfg: dict, run_dir: Path, out: Path, device, workers: int = 16) -> dict:
    meta = json.loads((run_dir / 'meta.json').read_text())
    return lewm_diagnose(
        cfg,
        out,
        device,
        workers=workers,
        checkpoint=str(run_dir / 'weights.pt'),
        action_mean=np.asarray(meta['action_mean'], dtype=np.float32),
        action_std=np.asarray(meta['action_std'], dtype=np.float32),
    )


def notify_result(model: str, updates: int, report: dict) -> None:
    if model == 'rdwm':
        rank = report['rank_summary']
        plan = report['plan_summary']
    else:
        rank = report['rank_summary']
        plan = report['plan_summary']
    text = (
        f"exposure {model} {updates} | done | "
        f"spearman {rank['mean_spearman']:.3f} | "
        f"best pct {rank['mean_best_distance_percentile']:.1f} | "
        f"plan {plan['success_count']}/10 | "
        f"dist {plan['goal_pos_initial']:.0f}->{plan['goal_pos_final']:.0f}"
    )
    print(text, flush=True)
    try:
        _notify(text)
    except Exception as exc:
        print(f'slack failed: {type(exc).__name__}', flush=True)
