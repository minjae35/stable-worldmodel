import copy

import pytest
import torch

from rdwm.exp1.config import load_config

_BASE = load_config()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(_BASE)
    c['paths']['run_root'] = str(tmp_path / 'runs')
    c['paths']['cache_root'] = str(tmp_path / 'cache')
    return c


def make_batch(cfg, env, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    size = cfg['model']['image_size']
    d = cfg['envs'][env]['action_dim']
    pixels = torch.randn(b, 5, 3, size, size, generator=g)
    actions = torch.randn(b, 4, cfg['model']['action_block'], d, generator=g)
    return pixels, actions
