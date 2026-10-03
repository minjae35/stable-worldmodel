"""Train the PushT frozen-DINOv2 specialist for 5k, then compare it.

Same seed, split, and schedule as the trainable-encoder 5k specialist.
Ranking and horizon-1 CEM use the same 10 val pairs. This does not revise
experiment-1.md and does not start the 7-env pilot.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.diagnostics import diagnose  # noqa: E402
from rdwm.exp1.evaluate import load_model  # noqa: E402
from rdwm.exp1.exposure_eval import evaluate_rdwm, notify_result  # noqa: E402
from rdwm.exp1.train import train  # noqa: E402

BASELINE_CKPT = Path(
    '/workspace/rdwm_runs/exp1/exposure_rdwm_5000/A/seed0/pusht/ckpt_005000.pt'
)
BASELINE_RANK = Path('/workspace/rdwm_runs/exp1/logs/exposure/rdwm_5000.json')
LOG_ROOT = Path('/workspace/rdwm_runs/exp1/logs/dino_proto')


def _ratio(mse, persist):
    if persist is None or persist == 0:
        return None
    return float(mse) / float(persist)


def _prediction(cfg, ckpt: Path, device) -> dict:
    model = load_model(cfg, ckpt, device).eval()
    stats = diagnose(model, cfg, 'pusht', n=128, device=device)
    stats['mse_t1_over_persist'] = _ratio(stats['mse_t1'], stats['persist_t1'])
    stats['mse_t2_over_persist'] = _ratio(stats['mse_t2'], stats['persist_t2'])
    del model
    torch.cuda.empty_cache()
    return stats


def main() -> None:
    cfg = load_config(str(DEFAULT_CONFIG))
    cfg['train']['ckpt_every'] = 5001
    if not BASELINE_CKPT.is_file():
        raise FileNotFoundError(BASELINE_CKPT)
    out = train(
        cfg,
        variant='A',
        seed=0,
        envs=['pusht'],
        updates=5000,
        stage='dino_proto_pusht',
        schedule_envs=['pusht', 'cube'],
        num_workers=8,
        encoder='frozen_dinov2_small',
    )
    ckpt = out / 'ckpt_005000.pt'
    device = torch.device('cuda')
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    print('prediction | baseline trainable encoder', flush=True)
    base_pred = _prediction(cfg, BASELINE_CKPT, device)
    print('prediction | frozen DINOv2 encoder', flush=True)
    dino_pred = _prediction(cfg, ckpt, device)
    print('rank and horizon-1 CEM | frozen DINOv2 encoder', flush=True)
    report = evaluate_rdwm(cfg, ckpt, LOG_ROOT / 'dino_5000.json', device, workers=16)
    baseline_rank = json.loads(BASELINE_RANK.read_text())
    comparison = {
        'protocol': {
            'pairs': 'same first 10 val pairs, seeds 1000+i',
            'cem': 'horizon 1 block, receding 1, 300/30/30',
            'prediction': '128 val clips, seed 0, mse against persistence',
            'train': 'PushT specialist, seed 0, 5000 updates, batch 32, own cosine',
            'note': 'prototype only; experiment-1.md is unchanged',
        },
        'baseline_ckpt': str(BASELINE_CKPT),
        'dino_ckpt': str(ckpt),
        'prediction': {'trainable_vit_5k': base_pred, 'frozen_dinov2_5k': dino_pred},
        'baseline_rank_plan': {
            'rank_summary': baseline_rank['rank_summary'],
            'plan_summary': baseline_rank['plan_summary'],
            'source': str(BASELINE_RANK),
        },
        'dino_rank_plan': {
            'rank_summary': report['rank_summary'],
            'plan_summary': report['plan_summary'],
        },
    }
    dest = LOG_ROOT / 'comparison_5k.json'
    dest.write_text(json.dumps(comparison, indent=2))
    rank = report['rank_summary']
    plan = report['plan_summary']
    base = baseline_rank['rank_summary']
    base_plan = baseline_rank['plan_summary']
    text = (
        f"dino proto 5k | spearman {rank['mean_spearman']:.3f} "
        f"(vit {base['mean_spearman']:.3f}) | "
        f"best pct {rank['mean_best_distance_percentile']:.2f} "
        f"(vit {base['mean_best_distance_percentile']:.2f}) | "
        f"closer {rank['mean_closer_rank_percentile']:.2f} "
        f"(vit {base['mean_closer_rank_percentile']:.2f}) | "
        f"cem {plan['success_count']}/10 (vit {base_plan['success_count']}/10) | "
        f"pred t1/persist {dino_pred['mse_t1_over_persist']:.3f} "
        f"(vit {base_pred['mse_t1_over_persist']:.3f})"
    )
    print(text, flush=True)
    notify_result('dino-proto', 5000, {'rank_summary': rank, 'plan_summary': plan})
    print(text, flush=True)


if __name__ == '__main__':
    main()
