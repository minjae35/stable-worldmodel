"""Short 1/2/4 GPU bench of the Stage 2-A generalist. Not a joint run.

The clip tables for TwoRoom and Cube are not on this machine, so the
timed loop uses cached synthetic clips through the real LeWM. A separate
PushT read at the end is labeled as real data and is not mixed into the
speedup table.
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get('BENCH_PYTHON', '/venv/stable-worldmodel/bin/python')
TRAIN = ROOT / 'scripts' / 'train' / 'lewm_generalist.py'
OUT = ROOT / 'results' / 'stage2a'


def _run(name: str, devices: int, steps: int, extra: list[str] | None = None):
    cmd = [
        PY,
        str(TRAIN),
        'data.source=synthetic',
        'data.synthetic_clips=2560',
        f'trainer.devices={devices}',
        f'trainer.max_steps={steps}',
        'trainer.max_epochs=1',
        'trainer.num_sanity_val_steps=0',
        'trainer.limit_val_batches=0',
        'per_env_batch_size=32',
        'num_workers=2',
        'loader.prefetch_factor=2',
        'weights_every_n_epochs=0',
        'checkpoint_every_n_steps=100000',
        f'output_model_name={name}',
        'hydra.run.dir=.',
        'hydra.output_subdir=null',
        'hydra.job.chdir=false',
    ]
    if extra:
        cmd.extend(extra)
    print('\n=== ' + ' '.join(cmd) + '\n', flush=True)
    env = os.environ.copy()
    env['STAGE2A_PROFILE'] = '1' if steps >= 16 else '0'
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _load(name: str) -> dict:
    folder = OUT / name
    bench = json.loads((folder / 'benchmark.json').read_text())
    topo = json.loads((folder / 'topology.json').read_text())
    bench.update({f'topo_{key}': value for key, value in topo.items()})
    bench['sigreg_finite'] = _sigreg_finite(folder / 'metrics.csv')
    bench['lr_series'] = _lr_series(folder / 'metrics.csv')
    return bench


def _sigreg_finite(path: Path) -> bool:
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get('stage') == 'train' and row.get('sigreg_loss')
        ]
    if not rows:
        return False
    values = [float(row['sigreg_loss']) for row in rows]
    return all(value == value and abs(value) < 1e6 for value in values)


def _lr_series(path: Path) -> list[float]:
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get('stage') == 'train' and row.get('lr')
        ]
    return [float(row['lr']) for row in rows]


def _resume_ok(name: str) -> dict:
    folder = OUT / name
    with (folder / 'metrics.csv').open() as handle:
        steps = [
            int(row['global_step'])
            for row in csv.DictReader(handle)
            if row.get('stage') == 'train'
        ]
    lrs = _lr_series(folder / 'metrics.csv')
    increasing = steps == sorted(steps) and len(steps) == len(set(steps))
    continued = steps[:2] == [1, 2] and steps[-1] == 4 and 3 in steps
    return {
        'steps': steps,
        'lrs': lrs,
        'monotonic_unique_steps': increasing,
        'continued_to_4': continued,
        'lr_not_reset': len(lrs) >= 4 and lrs[2] != lrs[0],
    }


def _pusht_read() -> dict:
    path = Path('/workspace/datasets/lewm-pusht/train.lance')
    if not path.exists():
        return {'available': False}
    sys.path.insert(0, str(ROOT))
    import stable_worldmodel as swm

    started = time.perf_counter()
    dataset = swm.data.load_dataset(
        str(path),
        num_steps=4,
        frameskip=5,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    opened = time.perf_counter() - started
    samples = []
    for index in range(4):
        t0 = time.perf_counter()
        sample = dataset[index]
        samples.append(time.perf_counter() - t0)
        assert tuple(sample['action'].shape) == (4, 10)
    return {
        'available': True,
        'kind': 'real_pusht_getitem',
        'open_s': opened,
        'getitem_s': samples,
        'clips': len(dataset),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for devices in (1, 2, 4):
        _run(f'ddp_bench_syn_{devices}gpu', devices, 16)
    rows = [_load(f'ddp_bench_syn_{devices}gpu') for devices in (1, 2, 4)]
    base = rows[0]['clips_per_s'] or 1.0
    for row in rows:
        clips = row['clips_per_s'] or 0.0
        row['speedup_vs_1gpu'] = clips / base
        row['efficiency'] = row['speedup_vs_1gpu'] / row['world_size']

    _run(
        'ddp_bench_resume_1gpu',
        1,
        2,
        extra=[
            'per_env_batch_size=8',
            'data.synthetic_clips=512',
            'checkpoint_every_n_steps=1',
        ],
    )
    _run(
        'ddp_bench_resume_1gpu',
        1,
        4,
        extra=[
            'per_env_batch_size=8',
            'data.synthetic_clips=512',
            'checkpoint_every_n_steps=1',
        ],
    )
    _run(
        'ddp_bench_resume_2gpu',
        2,
        2,
        extra=[
            'per_env_batch_size=8',
            'data.synthetic_clips=512',
            'checkpoint_every_n_steps=1',
        ],
    )
    _run(
        'ddp_bench_resume_2gpu',
        2,
        4,
        extra=[
            'per_env_batch_size=8',
            'data.synthetic_clips=512',
            'checkpoint_every_n_steps=1',
        ],
    )
    summary = {
        'kind': 'synthetic_cached_real_model',
        'note': (
            'TwoRoom and Cube lance tables are not local. '
            'Per-GPU batch is 32 clips per environment. '
            'Effective batch grows with the GPU count. '
            'Learning rate stays 5e-5.'
        ),
        'runs': rows,
        'resume_1gpu': _resume_ok('ddp_bench_resume_1gpu'),
        'resume_2gpu': _resume_ok('ddp_bench_resume_2gpu'),
        'pusht': _pusht_read(),
    }
    dest = OUT / 'ddp_bench_summary.json'
    dest.write_text(json.dumps(summary, indent=2))
    print('\nBENCH_SUMMARY', dest, flush=True)
    for row in rows:
        print(
            f"gpus={row['world_size']} clips/s={row['clips_per_s']:.2f} "
            f"steps/s={row['steps_per_s']:.3f} "
            f"data_s={row['median_data_s']} "
            f"sigreg_comm_s={row['median_sigreg_comm_s']} "
            f"nccl_frac={row['nccl_fraction']} "
            f"speedup={row['speedup_vs_1gpu']:.3f} "
            f"eff_batch={row['effective_clips_per_env']} "
            f"grad={row['grad_norm']} sigreg_ok={row['sigreg_finite']}",
            flush=True,
        )
    print('RESUME_1', summary['resume_1gpu'], flush=True)
    print('RESUME_2', summary['resume_2gpu'], flush=True)


if __name__ == '__main__':
    main()
