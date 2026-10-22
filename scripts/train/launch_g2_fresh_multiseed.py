"""Sequential launcher for reproducible G2 Fresh multi-seed A/B runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'


def build_commands(
    python: str,
    output: Path,
    seeds: tuple[int, ...],
    steps: int,
    checkpoint_every: int,
) -> list[list[str]]:
    commands = []
    for seed in seeds:
        seed_root = output / f'seed_{seed}'
        calibration = seed_root / 'gradient_calibration.json'
        for arm, name in (
            ('A', 'A_fresh_uniform_mse'),
            ('B', 'B_fresh_temporal_mse'),
        ):
            commands.append([
                python,
                str(ROOT / 'scripts/train/g2_fresh_spatial.py'),
                '--arm', arm,
                '--output-dir', str(seed_root / name),
                '--normalizer-file', str(NORMALIZERS),
                '--loss-calibration-file', str(calibration),
                '--steps', str(steps),
                '--seed', str(seed),
                '--checkpoint-every', str(checkpoint_every),
                '--defer-evaluation',
            ])
    return commands


def preflight(output: Path, commands: list[list[str]]) -> None:
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    if not NORMALIZERS.is_file():
        raise FileNotFoundError(NORMALIZERS)
    targets = [Path(command[command.index('--output-dir') + 1]) for command in commands]
    if len(targets) != len(set(targets)):
        raise ValueError('duplicate multi-seed output directories')
    calibrations: dict[int, set[str]] = {}
    for command in commands:
        seed = int(command[command.index('--seed') + 1])
        calibration = command[command.index('--loss-calibration-file') + 1]
        calibrations.setdefault(seed, set()).add(calibration)
    if any(len(paths) != 1 for paths in calibrations.values()):
        raise ValueError('A/B within a seed must share one calibration artifact')
    if len({next(iter(paths)) for paths in calibrations.values()}) != len(calibrations):
        raise ValueError('different seeds must use independent calibration artifacts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default='/venv/stable-wm/bin/python')
    parser.add_argument('--seeds', type=int, nargs='+', default=(3073, 3074, 3075))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('seeds must be unique and non-empty')
    if args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError('steps and checkpoint interval must be positive')
    output = args.output_root.resolve()
    commands = build_commands(
        args.python, output, seeds, args.steps, args.checkpoint_every
    )
    preflight(output, commands)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('RUNNING\n')
    report = {
        'seeds': list(seeds),
        'sequential': True,
        'stop_on_first_failure': True,
        'steps': args.steps,
        'checkpoint_every': args.checkpoint_every,
        'commands': commands,
        'runs': [],
    }
    config_path = output / 'launcher_config.json'
    config_path.write_text(json.dumps(report, indent=2) + '\n')
    for command in commands:
        seed = int(command[command.index('--seed') + 1])
        arm = command[command.index('--arm') + 1]
        target = Path(command[command.index('--output-dir') + 1])
        log = output / f'seed_{seed}_{arm}.log'
        began = time.perf_counter()
        with log.open('xb') as handle:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            assert process.stdout is not None
            for chunk in iter(process.stdout.readline, b''):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                handle.write(chunk)
                handle.flush()
            returncode = process.wait()
        report['runs'].append({
            'seed': seed,
            'arm': arm,
            'returncode': returncode,
            'seconds': time.perf_counter() - began,
            'log': str(log),
            'output': str(target),
        })
        config_path.write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            (output / 'STATUS').write_text(
                f'FAILED seed={seed} arm={arm} returncode={returncode}\n'
            )
            raise SystemExit(returncode)
    (output / 'STATUS').write_text('TRAINING_DONE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    main()
