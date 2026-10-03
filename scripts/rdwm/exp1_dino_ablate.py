"""Three PushT ablations of the frozen-DINOv2 RD-WM prototype.

RGB and action only. Proprio is not an input. experiment-1.md is unchanged.
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
from rdwm.exp1.exposure_eval import _notify, evaluate_rdwm  # noqa: E402
from rdwm.exp1.train import train  # noqa: E402

WIDE = {
    'dyn_depth': 6,
    'dyn_heads': 16,
    'dyn_head_dim': 64,
    'dyn_ffn': 2048,
    'dyn_dropout': 0.1,
}
OPT = {'lr': 5.0e-4, 'weight_decay': 0.0}
LOG_ROOT = Path('/workspace/rdwm_runs/exp1/logs/dino_ablate')
SPECS = {
    'onestep': {
        'stage': 'dino_ablate_onestep',
        'label': 'DINOv2 one-step',
        'model': {},
        'train': {},
        'reinit': False,
        'note': (
            'Removes the t+2 recursive step. One-step MSE uses weight '
            't1+t2=1.0 so SIGReg stays at 0.09. Predictor, AdaLN, and '
            'optimizer stay on the DINOv2 prototype.'
        ),
    },
    'wide': {
        'stage': 'dino_ablate_onestep_wide',
        'label': 'DINOv2 one-step wide',
        'model': dict(WIDE),
        'train': dict(OPT),
        'reinit': True,
        'note': (
            'One-step loss plus PreJEPA predictor width: depth 6, 16 heads, '
            'head dim 64, MLP 2048, dropout 0.1. AdamW lr 5e-4, weight decay 0. '
            'Token dim stays 192. Env and action both stay on AdaLN.'
        ),
    },
    'concat': {
        'stage': 'dino_ablate_onestep_concat',
        'label': 'DINOv2 one-step concat',
        'model': {**WIDE, 'action_conditioning': 'concat', 'action_emb_dim': 10},
        'train': dict(OPT),
        'reinit': True,
        'note': (
            'Wide one-step model, but action is a 10-d embedding concatenated '
            'onto each token and projected back to 192. Env conditioning stays AdaLN.'
        ),
    },
}


def main() -> None:
    name = sys.argv[1]
    if name not in SPECS:
        raise SystemExit(f'usage: exp1_dino_ablate.py {{{",".join(SPECS)}}}')
    spec = SPECS[name]
    cfg = load_config(str(DEFAULT_CONFIG))
    cfg['train']['ckpt_every'] = 5001
    out = train(
        cfg,
        variant='A',
        seed=0,
        envs=['pusht'],
        updates=5000,
        stage=spec['stage'],
        schedule_envs=['pusht', 'cube'],
        num_workers=8,
        encoder='frozen_dinov2_small',
        objective='one_step',
        model_overrides=spec['model'] or None,
        train_overrides=spec['train'] or None,
        allow_dynamics_reinit=spec['reinit'],
        progress_label=spec['label'],
    )
    ckpt = out / 'ckpt_005000.pt'
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    report = evaluate_rdwm(
        load_config(str(DEFAULT_CONFIG)),
        ckpt,
        LOG_ROOT / f'{name}.json',
        torch.device('cuda'),
        workers=16,
    )
    rank, plan = report['rank_summary'], report['plan_summary']
    text = (
        f"dino ablate {name} | spearman {rank['mean_spearman']:.3f} | "
        f"closer rank {rank['mean_closer_rank_percentile']:.1f} | "
        f"best pct {rank['mean_best_distance_percentile']:.1f} | "
        f"plan {plan['success_count']}/10 | closer {plan['closer_count']}/10"
    )
    print(text, flush=True)
    (LOG_ROOT / f'{name}_note.json').write_text(json.dumps({'name': name, 'note': spec['note'], 'text': text}, indent=2))
    try:
        _notify(text)
    except Exception as exc:
        print(f'slack failed: {type(exc).__name__}', flush=True)


if __name__ == '__main__':
    main()
