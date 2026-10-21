"""Sequential launcher for the four G2 arms.

Recommended full-run invocation (do not run during pipeline validation)::

    tmux new-session -d -s stablewm_stage3_g2 \
      '/venv/stable-wm/bin/python scripts/train/launch_g2_2x2.py'
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
TRAIN = ROOT / 'scripts/train/g2_spatial.py'
MANIFEST = ROOT / 'results/stage3/g2_c1_followup/manifest.json'
CHECKPOINT = ROOT / 'results/stage3/g1_spatial/overfit500/checkpoint_step500.pt'
RUN_NAMES = {
    'A': 'A_uniform_mse',
    'B': 'B_temporal_mse',
    'C': 'C_uniform_l1',
    'D': 'D_temporal_l1',
}


def build_commands(
    python: str,
    output_root: Path,
    smoke: bool,
    defer_evaluation: bool = False,
) -> list[list[str]]:
    commands = []
    calibration = output_root / 'gradient_calibration.json'
    for arm, name in RUN_NAMES.items():
        command = [
            python,
            str(TRAIN),
            '--arm',
            arm,
            '--output-dir',
            str(output_root / name),
            '--loss-calibration-file',
            str(calibration),
        ]
        if smoke:
            command.extend(
                [
                    '--steps',
                    '2',
                    '--batch-per-env',
                    '2',
                    '--checkpoint-every',
                    '1',
                    '--max-train-items',
                    '8',
                    '--eval-items',
                    '8',
                    '--eval-probe-train-items',
                    '8',
                    '--encode-batch',
                    '8',
                    '--eval-batch',
                    '8',
                ]
            )
        if defer_evaluation:
            command.append('--defer-evaluation')
        commands.append(command)
    return commands


def preflight(output_root: Path, commands: list[list[str]]) -> None:
    if not MANIFEST.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError('G2 manifest or G1 checkpoint is missing')
    if output_root.exists():
        raise FileExistsError(f'refusing to overwrite output root {output_root}')
    targets = [Path(command[command.index('--output-dir') + 1]) for command in commands]
    if len(set(targets)) != 4 or any(target.exists() for target in targets):
        raise FileExistsError('one or more G2 arm output directories already exist')


def run_command_tee(command: list[str], log_path: Path) -> int:
    """Stream tqdm/log output to both the tmux pane and a durable file."""
    with log_path.open('xb', buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError('failed to capture child output')
        while chunk := process.stdout.read1(4096):
            log.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()


def run_launcher(output_root: Path, commands: list[list[str]]) -> int:
    output_root.mkdir(parents=True, exist_ok=False)
    status = output_root / 'STATUS'
    status.write_text('RUNNING\n')
    document = {
        'session_name': 'stablewm_stage3_g2',
        'sequential': True,
        'stop_on_first_failure': True,
        'commands': commands,
        'runs': [],
    }
    (output_root / 'launcher_config.json').write_text(json.dumps(document, indent=2) + '\n')
    for arm, command in zip(RUN_NAMES, commands):
        name = RUN_NAMES[arm]
        began = time.time()
        log_path = output_root / f'{name}.log'
        print(f'G2_LAUNCH arm={arm} log={log_path}', flush=True)
        returncode = run_command_tee(command, log_path)
        run_dir = output_root / name
        run_log = run_dir / 'train.log'
        if run_dir.is_dir():
            shutil.copy2(log_path, run_log)
        row = {
            'arm': arm,
            'name': name,
            'returncode': returncode,
            'seconds': time.time() - began,
            'log': str(log_path),
            'run_log': str(run_log) if run_log.is_file() else None,
        }
        document['runs'].append(row)
        (output_root / 'launcher_config.json').write_text(
            json.dumps(document, indent=2) + '\n'
        )
        if returncode != 0:
            status.write_text(f'FAILED arm={arm} returncode={returncode}\n')
            return returncode
    status.write_text('DONE\n')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ROOT / 'results/stage3/g2_2x2',
    )
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--defer-evaluation', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    commands = build_commands(
        sys.executable, output_root, args.smoke, args.defer_evaluation
    )
    preflight(output_root, commands)
    if args.dry_run:
        print(json.dumps({'commands': commands}, indent=2))
        return 0
    return run_launcher(output_root, commands)


if __name__ == '__main__':
    raise SystemExit(main())
