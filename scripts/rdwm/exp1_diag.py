"""Held-out latent diagnostics for Experiment 1 checkpoints.

    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python scripts/rdwm/exp1_diag.py CKPT [CKPT ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.diagnostics import diagnose  # noqa: E402
from rdwm.exp1.evaluate import load_model  # noqa: E402
from rdwm.exp1.util import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ckpts', nargs='+')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--n', type=int, default=128)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda')
    for ckpt in map(Path, args.ckpts):
        model = load_model(cfg, ckpt, device)
        torch.manual_seed(0)
        report = {env: diagnose(model, cfg, env, args.n, device) for env in model.envs}
        write_json(ckpt.parent / 'eval' / f'{ckpt.stem}_diag.json', report)
        print(ckpt)
        print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
