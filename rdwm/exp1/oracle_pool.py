"""Parallel PushT candidate rollouts.

Each worker is one process with one simulator. Thread pools inside the
worker stay at 1. One rollout returns both the rendered frame and the
PushT position goal distance, so a candidate is never simulated twice.
"""

from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MUJOCO_GL', 'egl')

import multiprocessing as mp
from multiprocessing import resource_tracker, shared_memory

import numpy as np

_SESSION = None
_FRAME_SHAPE = (224, 224, 3)


def _init_worker() -> None:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from rdwm.exp1.planning import open_eval_session

    global _SESSION
    _SESSION = open_eval_session('pusht')


def _prime(session, prime: dict) -> None:
    """Same reset as PushTSession.restore + set_goal, including one physics step."""
    from rdwm.envs.sessions import _core

    core = _core(session.env)
    pose = prime.get('goal_pose')
    if pose is not None:
        core.goal_pose = np.asarray(pose, dtype=np.float64).copy()
    core._set_goal_state(np.asarray(prime['goal_state'], dtype=np.float64).copy())
    core._set_state(np.asarray(prime['state'], dtype=np.float64).copy())


def _attach_frames(name: str, n: int) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=name)
    # The parent owns the segment. A worker must not unlink it on exit.
    try:
        resource_tracker.unregister(shm._name, 'shared_memory')
    except Exception:
        pass
    buf = np.ndarray((n, *_FRAME_SHAPE), dtype=np.uint8, buffer=shm.buf)
    return shm, buf


def _release_frames(shm: shared_memory.SharedMemory) -> None:
    name = shm.name
    shm.close()
    try:
        shm.unlink()
    except Exception:
        import _posixshmem
        try:
            _posixshmem.shm_unlink(name if name.startswith('/') else '/' + name)
        except FileNotFoundError:
            pass


def _rollout_chunk(task: tuple) -> np.ndarray:
    prime, blocks, render, indices, shm_name, n_total = task
    session = _SESSION
    goal = np.asarray(prime['goal_state'], dtype=np.float64)
    dists = np.empty(len(blocks), dtype=np.float64)
    shm = None
    frames = None
    if render:
        shm, frames = _attach_frames(shm_name, n_total)
    try:
        for i, block in enumerate(blocks):
            _prime(session, prime)
            for action in block:
                session.step(np.asarray(action, dtype=np.float64))
            state = session.state()
            dists[i] = float(np.linalg.norm(goal[:4] - state[:4]))
            if frames is not None:
                frames[int(indices[i])] = np.ascontiguousarray(session.render())
    finally:
        if shm is not None:
            shm.close()
    return dists


class PushTPool:
    def __init__(self, workers: int):
        self.workers = int(workers)
        ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(self.workers, initializer=_init_worker)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def rollout(self, prime: dict, blocks: np.ndarray, render: bool):
        blocks = np.asarray(blocks, dtype=np.float32)
        n = int(len(blocks))
        if n == 0:
            return None, np.zeros(0, dtype=np.float64)
        pieces = min(self.workers, n)
        index = [idx for idx in np.array_split(np.arange(n), pieces) if len(idx)]
        shm = None
        try:
            name = None
            if render:
                shm = shared_memory.SharedMemory(create=True, size=n * int(np.prod(_FRAME_SHAPE)))
                name = shm.name
            tasks = [
                (prime, blocks[idx], render, idx.astype(np.int64), name, n)
                for idx in index
            ]
            parts = self.pool.map(_rollout_chunk, tasks, chunksize=1)
            dists = np.concatenate(parts)
            if not render:
                return None, dists
            frames = np.ndarray((n, *_FRAME_SHAPE), dtype=np.uint8, buffer=shm.buf).copy()
            return frames, dists
        finally:
            if shm is not None:
                _release_frames(shm)
