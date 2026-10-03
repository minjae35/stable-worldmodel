import numpy as np
import torch

from rdwm.exp1.data import (
    build_schedule,
    clip_table,
    restrict_schedule,
    schedule_digest,
    split_episodes,
)
from rdwm.exp1.init import lr_factor, param_groups
from rdwm.exp1.model import RDWM

ENVS7 = ['pusht', 'tworoom', 'cube', 'scene', 'reacher', 'pointmaze', 'antmaze']
PROBS = {3: 0.8, 2: 0.1, 1: 0.1}


def _sched(seed=0, envs=ENVS7, updates=700):
    counts = {e: 1000 + 10 * i for i, e in enumerate(envs)}
    return build_schedule(seed, envs, counts, updates, 32, PROBS)


def test_rounds_visit_each_env_once_and_are_deterministic():
    s = _sched()
    rounds = s['env'].reshape(-1, 7)
    assert all(sorted(r.tolist()) == list(range(7)) for r in rounds)
    assert len({tuple(r) for r in rounds}) > 1
    again = _sched()
    for key in s:
        assert np.array_equal(s[key], again[key])
    other = _sched(seed=1)
    assert not np.array_equal(s['env'], other['env'])
    assert all(len(set(row.tolist())) == 32 for row in s['clip'])


def test_history_mix():
    s = _sched(updates=7000)
    frac = {h: float((s['history'] == h).mean()) for h in (1, 2, 3)}
    assert abs(frac[3] - 0.8) < 0.03 and abs(frac[2] - 0.1) < 0.02


def test_specialist_schedule_is_generalist_subsequence():
    s = _sched(updates=7000)
    s = {'env': s['env'], 'history': s['history'], 'episode': s['clip'], 'start': s['clip']}
    for i, env in enumerate(ENVS7):
        part = restrict_schedule(s, ENVS7, env)
        assert len(part['env']) == 1000
        rows = s['env'] == i
        assert np.array_equal(part['episode'], s['episode'][rows])
        assert np.array_equal(part['history'], s['history'][rows])
    assert schedule_digest(s) == schedule_digest({k: v.copy() for k, v in s.items()})


class _FakeIndex:
    env = 'fake'

    def __init__(self, lengths, nan_rows=()):
        self.lengths = np.asarray(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        acts = np.zeros((int(self.lengths.sum()), 2), dtype=np.float32)
        for r in nan_rows:
            acts[r] = np.nan
        self.actions = acts


def test_clip_table_respects_terminal_nan_and_boundaries():
    lengths = [21, 25, 20]
    terminal = [int(o + n - 1) for o, n in zip(np.cumsum([0] + lengths[:-1]), lengths)]
    idx = _FakeIndex(lengths, nan_rows=terminal)
    eps, starts = clip_table(idx, np.arange(3), num_obs=5, stride=5)
    # ep0: T=21, start 0 needs obs at 20 (ok) and actions 0..19 (finite).
    # ep1: T=25 -> starts 0..4. ep2: T=20 -> no clip.
    assert eps.tolist() == [0] + [1] * 5
    assert starts.tolist() == [0, 0, 1, 2, 3, 4]


def test_split_is_disjoint_and_complete():
    split = split_episodes(1000, {'order': 'contiguous', 'val_frac': 0.1, 'test_frac': 0.1})
    joined = np.concatenate([split['train'], split['val'], split['test']])
    assert np.array_equal(joined, np.arange(1000))
    assert len(split['val']) == 100 and len(split['test']) == 100


def test_weight_decay_groups_and_lr_schedule(cfg):
    model = RDWM(cfg, 'C', ['pusht'], {'pusht': 2})
    decay, plain = param_groups(model, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    plain_names = {names[id(p)] for p in plain['params']}
    assert all(n.endswith('.bias') or 'norm' in n or '.net.1.' in n for n in plain_names)
    decay_names = {names[id(p)] for p in decay['params']}
    assert not any(n.endswith('.bias') for n in decay_names)
    assert 'encoder.blocks.0.norm1.weight' in plain_names
    assert plain['weight_decay'] == 0.0 and decay['weight_decay'] == 1e-3
    total, warmup = 70000, 700
    assert lr_factor(0, warmup, total) == 1 / 700
    assert lr_factor(699, warmup, total) == 1.0
    assert abs(lr_factor(700 + 34650, warmup, total) - 0.5) < 1e-6
    assert lr_factor(total - 1, warmup, total) < 1e-6
    assert torch.is_tensor(next(model.parameters()))
