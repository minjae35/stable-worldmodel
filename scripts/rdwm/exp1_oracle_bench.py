"""Small PushT rollout benchmark. Does not start the full oracle diagnostic."""

from __future__ import annotations

import json
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MUJOCO_GL', 'egl')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from rdwm.exp1.config import DEFAULT_CONFIG, env_path, load_config
from rdwm.exp1.evaluate import get_pairs
from rdwm.exp1.oracle_diag import _bank, _decision_prime, _latent_cost, _sim_outcomes
from rdwm.exp1.oracle_pool import PushTPool
from rdwm.exp1.planning import open_episode_reader, open_eval_session
from rdwm.envs.sessions import dataset_frame


def _cpu_times():
    parts = open('/proc/stat').readline().split()[1:]
    return np.asarray(parts, dtype=np.float64)


def _cpu_util(before, after) -> float:
    delta = after - before
    idle = delta[3] + delta[4]
    total = float(delta.sum())
    if total <= 0:
        return 0.0
    return 100.0 * (1.0 - float(idle) / total)


def _gpu_util() -> list[float]:
    import subprocess
    text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
    )
    return [float(line) for line in text.splitlines() if line.strip()]


def _time_rollouts(fn, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    ckpt = Path('/workspace/rdwm_runs/exp1/diag_pusht/A/seed0/pusht/ckpt_005000.pt')
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pair = get_pairs(cfg, 'pusht', 'val', cfg['eval']['pilot_val_pairs'], session, reader)[0]
    episode = reader.load_episode(int(pair['episode']))
    session.restore(episode, int(pair['start']))
    session.set_goal(episode, int(pair['goal']))
    prime = _decision_prime(session, episode, int(pair['start']))
    rng = np.random.default_rng(0)
    blocks = rng.uniform(-1.0, 1.0, size=(300, 5, 2)).astype(np.float32)
    cand = torch.as_tensor(blocks[:, None])

    serial_frames, serial_dists = _sim_outcomes(
        session, cand[:32], render=True, pool=None, prime=prime
    )
    repeats = 4

    def serial_render():
        _sim_outcomes(session, cand, render=True, pool=None)

    def serial_state():
        _sim_outcomes(session, cand, render=False, pool=None)

    print('serial timing', flush=True)
    cpu0 = _cpu_times()
    serial_render_s = _time_rollouts(serial_render, repeats)
    cpu1 = _cpu_times()
    serial_state_s = _time_rollouts(serial_state, repeats)
    serial_render_cps = len(blocks) / serial_render_s
    serial_state_cps = len(blocks) / serial_state_s

    def bench_workers(workers: int) -> dict:
        print(f'pool {workers}', flush=True)
        pool = PushTPool(workers)
        try:
            frames, dists = _sim_outcomes(session, cand[:32], render=True, pool=pool, prime=prime)
            frame_ok = np.array_equal(frames, serial_frames)
            pixel_diff = int(np.max(np.abs(frames.astype(np.int16) - serial_frames.astype(np.int16))))
            dist_ok = float(np.max(np.abs(dists - serial_dists)))
            def run():
                _sim_outcomes(session, cand, render=True, pool=pool, prime=prime)
            def run_state():
                _sim_outcomes(session, cand, render=False, pool=pool, prime=prime)
            run()  # warmup
            before = _cpu_times()
            elapsed = _time_rollouts(run, repeats)
            after = _cpu_times()
            state_elapsed = _time_rollouts(run_state, repeats)
        finally:
            pool.close()
        return {
            'workers': workers,
            'render_candidates_per_s': len(blocks) / elapsed,
            'state_candidates_per_s': len(blocks) / state_elapsed,
            'render_speedup_vs_serial': (len(blocks) / elapsed) / serial_render_cps,
            'cpu_util_pct': _cpu_util(before, after),
            'frames_match_serial': bool(frame_ok),
            'max_abs_pixel_diff': pixel_diff,
            'max_abs_distance_diff': dist_ok,
        }

    report = {
        'candidates_per_batch': len(blocks),
        'repeats': repeats,
        'serial_render_candidates_per_s': serial_render_cps,
        'serial_state_candidates_per_s': serial_state_cps,
        'serial_cpu_util_pct': _cpu_util(cpu0, cpu1),
        'workers': [bench_workers(32)],
    }
    best = report['workers'][0]
    if not best['frames_match_serial'] or best['max_abs_distance_diff'] > 1e-6:
        out = Path('/workspace/rdwm_runs/exp1/logs/diag_pusht/oracle_bench.json')
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit('parallel rollout does not match serial')
    if best['render_speedup_vs_serial'] < 12 or best['cpu_util_pct'] > 50:
        for workers in (16, 64):
            report['workers'].append(bench_workers(workers))
        best = max(report['workers'], key=lambda row: row['render_candidates_per_s'])

    bank = _bank(cfg, ckpt)
    probe_frames, _ = _sim_outcomes(session, cand, render=True, pool=None)
    from rdwm.exp1.planning import _encode_frame
    z_goal = _encode_frame(bank[0], dataset_frame(episode, int(pair['goal'])), cfg, torch.device('cuda:0'))
    torch.cuda.synchronize()
    for _ in range(2):
        _latent_cost(bank, probe_frames, z_goal, cfg)
    torch.cuda.synchronize()
    gpu_before = _gpu_util()
    t0 = time.perf_counter()
    encode_n = 0
    while time.perf_counter() - t0 < 2.0:
        _latent_cost(bank, probe_frames, z_goal, cfg)
        encode_n += len(probe_frames)
    torch.cuda.synchronize()
    encode_s = time.perf_counter() - t0
    gpu_after = _gpu_util()
    encode_cps = encode_n / encode_s

    b_candidates = 10 * 10 * 30 * 300
    c_candidates = b_candidates
    corr_candidates = 10 * 300
    sim_b = b_candidates / best['render_candidates_per_s']
    sim_c = c_candidates / best['state_candidates_per_s']
    sim_corr = corr_candidates / best['render_candidates_per_s']
    encode_b = (b_candidates + corr_candidates) / encode_cps
    report['encode_candidates_per_s'] = encode_cps
    report['gpu_util_during_encode_pct'] = gpu_after
    report['gpu_util_before_encode_pct'] = gpu_before
    report['chosen_workers'] = best['workers']
    report['estimated_full_seconds'] = {
        'A_model_cem': 40,
        'correlation_sim': sim_corr,
        'B_sim': sim_b,
        'B_encode': encode_b,
        'C_sim': sim_c,
        'total': 40 + sim_corr + sim_b + encode_b + sim_c,
    }
    session.close()
    out = Path('/workspace/rdwm_runs/exp1/logs/diag_pusht/oracle_bench.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
