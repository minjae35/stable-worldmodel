"""Train one from-scratch LeWM PushT run for a fixed update budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.lewm_train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--stage', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    train(
        load_config(args.config),
        seed=args.seed,
        updates=args.updates,
        stage=args.stage,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
