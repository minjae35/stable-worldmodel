"""Sequential launcher for the Stage-3 G2 fresh-initialization A/B pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
TRAIN = ROOT / 'scripts/train/g2_fresh_spatial.py'
DEFAULT_OUTPUT = ROOT / 'results/stage3/g2_fresh_ab'
RUN_NAMES = {'A': 'A_fresh_uniform_mse', 'B': 'B_fresh_temporal_mse'}


def build_commands(python: str, output_root: Path, smoke: bool) -> list[list[str]]:
    normalizer = output_root / 'action_normalizers.json'
    calibration = output_root / 'gradient_calibration.json'
    commands = []
    for arm, name in RUN_NAMES.items():
        command = [
            python,
            str(TRAIN),
            '--arm', arm,
            '--output-dir', str(output_root / name),
            '--normalizer-file', str(normalizer),
            '--loss-calibration-file', str(calibration),
            '--defer-evaluation',
        ]
        if smoke:
            command.extend([
                '--steps', '2', '--batch-per-env', '2',
                '--checkpoint-every', '1', '--max-train-items', '8',
                '--encode-batch', '8',
            ])
        commands.append(command)
    return commands


def preflight(output_root: Path, commands: list[list[str]]) -> None:
    if output_root.exists():
        raise FileExistsError(f'refusing to overwrite output root {output_root}')
    targets = [Path(command[command.index('--output-dir') + 1]) for command in commands]
    if len(set(targets)) != 2 or any(target.exists() for target in targets):
        raise FileExistsError('one or more fresh A/B output directories already exist')


def run_command_tee(command: list[str], log_path: Path) -> int:
    with log_path.open('xb', buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
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
        'session_name': 'stablewm_stage3_g2_fresh_ab',
        'sequential': True,
        'stop_on_first_failure': True,
        'commands': commands,
        'runs': [],
    }
    config_path = output_root / 'launcher_config.json'
    config_path.write_text(json.dumps(document, indent=2) + '\n')
    for arm, command in zip(RUN_NAMES, commands):
        name = RUN_NAMES[arm]
        began = time.time()
        log_path = output_root / f'{name}.log'
        print(f'G2_FRESH_LAUNCH arm={arm} log={log_path}', flush=True)
        returncode = run_command_tee(command, log_path)
        run_dir = output_root / name
        run_log = run_dir / 'train.log'
        if run_dir.is_dir() and not run_log.exists():
            shutil.copy2(log_path, run_log)
        document['runs'].append({
            'arm': arm,
            'name': name,
            'returncode': returncode,
            'seconds': time.time() - began,
            'log': str(log_path),
            'run_log': str(run_log) if run_log.is_file() else None,
        })
        config_path.write_text(json.dumps(document, indent=2) + '\n')
        if returncode != 0:
            status.write_text(f'FAILED arm={arm} returncode={returncode}\n')
            return returncode
    status.write_text('TRAINING_DONE_EVALUATION_PENDING\n')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    commands = build_commands(sys.executable, output_root, args.smoke)
    preflight(output_root, commands)
    if args.dry_run:
        print(json.dumps({'commands': commands}, indent=2))
        return 0
    return run_launcher(output_root, commands)


if __name__ == '__main__':
    raise SystemExit(main())
