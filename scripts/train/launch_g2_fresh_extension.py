"""Sequential tmux-friendly launcher for the G2 Fresh 500 -> 2,000 extension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
SOURCE = ROOT / 'results/stage3/g2_fresh_ab'


def commands(python: str, output: Path) -> list[list[str]]:
    result = []
    for arm, source_name, output_name in (
        ('A', 'A_fresh_uniform_mse', 'A_fresh_uniform_mse_extension'),
        ('B', 'B_fresh_temporal_mse', 'B_fresh_temporal_mse_extension'),
    ):
        source = SOURCE / source_name
        result.append([
            python, str(ROOT / 'scripts/train/g2_fresh_extend.py'),
            '--arm', arm,
            '--source-dir', str(source),
            '--source-checkpoint', str(source / 'checkpoint_step500.pt'),
            '--output-dir', str(output / output_name),
            '--normalizer-file', str(SOURCE / 'action_normalizers.json'),
            '--loss-calibration-file', str(SOURCE / 'gradient_calibration.json'),
            '--total-steps', '2000',
            '--checkpoint-steps', '1000', '1500', '2000',
        ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default='/venv/stable-wm/bin/python')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('RUNNING\n')
    planned = commands(args.python, output)
    report = {'sequential': True, 'stop_on_first_failure': True, 'commands': planned, 'runs': []}
    (output / 'launcher_config.json').write_text(json.dumps(report, indent=2) + '\n')
    for command in planned:
        arm = command[command.index('--arm') + 1]
        target = Path(command[command.index('--output-dir') + 1])
        log = output / f'{target.name}.log'
        began = time.perf_counter()
        with log.open('xb') as handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.stdout is not None
            for chunk in iter(process.stdout.readline, b''):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                handle.write(chunk)
                handle.flush()
            returncode = process.wait()
        report['runs'].append({
            'arm': arm, 'returncode': returncode,
            'seconds': time.perf_counter() - began, 'log': str(log),
        })
        (output / 'launcher_config.json').write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            (output / 'STATUS').write_text(f'FAILED arm={arm} returncode={returncode}\n')
            raise SystemExit(returncode)
    (output / 'STATUS').write_text('TRAINING_DONE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    main()
