"""DINO-WM / PreJEPA parity against the frozen-DINOv2 RD-WM prototype."""

from __future__ import annotations

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
from rdwm.exp1.exposure_eval import _notify  # noqa: E402
from rdwm.exp1.prejepa_diag import diagnose  # noqa: E402


def main() -> None:
    cfg = load_config(str(DEFAULT_CONFIG))
    cost = sys.argv[1] if len(sys.argv) > 1 else 'pixels_proprio'
    proprio_input = sys.argv[2] if len(sys.argv) > 2 else 'original'
    tag = cost if proprio_input == 'original' else f'{cost}_{proprio_input}'
    out = Path(
        '/workspace/rdwm_runs/exp1/logs/dinowm/parity.json'
        if tag == 'pixels_proprio'
        else f'/workspace/rdwm_runs/exp1/logs/dinowm/parity_{tag}.json'
    )
    report = diagnose(
        cfg, out, torch.device('cuda'), workers=16, cost=cost, proprio_input=proprio_input
    )
    rank = report['rank_summary']
    plan = report['plan_summary']
    text = (
        f"dinowm parity | spearman {rank['mean_spearman']:.3f} | "
        f"closer rank {rank['mean_closer_rank_percentile']:.1f} | "
        f"best pct {rank['mean_best_distance_percentile']:.1f} | "
        f"plan {plan['success_count']}/10 | closer {plan['closer_count']}/10 | "
        f"dist {plan['goal_pos_initial']:.0f}->{plan['goal_pos_final']:.0f}"
    )
    print(text, flush=True)
    try:
        _notify(text)
    except Exception as exc:
        print(f'slack failed: {type(exc).__name__}', flush=True)


if __name__ == '__main__':
    main()
