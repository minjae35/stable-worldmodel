"""Clip loader against a downloaded Lance table (skipped when absent)."""

import numpy as np
import pytest
import torch

from rdwm.exp1.config import load_config
from rdwm.exp1.data import (
    EnvData,
    ScheduleBatches,
    build_schedule,
    resolve_schedule,
    save_schedule,
)

_CFG = load_config()


def _local(env):
    from pathlib import Path

    path = Path(_CFG['paths']['data_root']) / _CFG['envs'][env]['local']
    return str(path) if path.exists() else None


@pytest.mark.parametrize('env', ['pusht', 'cube'])
def test_clip_rows_and_actions_line_up(env, tmp_path):
    import lance

    path = _local(env)
    if path is None:
        pytest.skip(f'{env} not downloaded')
    data = EnvData(_CFG, env, path)
    raw = build_schedule(0, [env], {env: len(data.clip_ep)}, 6, 32, _CFG['data']['history_probs'])
    resolved = resolve_schedule(raw, [env], {env: data})
    save_schedule(tmp_path, resolved, {})
    ds = ScheduleBatches([env], {env: path}, {env: str(data.index.folder)}, str(tmp_path), 0, 5, 5, 5)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=2, multiprocessing_context='spawn'
    )
    table = lance.dataset(path)
    d = _CFG['envs'][env]['action_dim']
    hw = _CFG['envs'][env]['native_hw']
    seen = 0
    for batch in loader:
        seen += 1
        assert batch['pixels'].shape == (32, 5, 3, hw, hw)
        assert batch['pixels'].dtype == torch.uint8
        assert batch['actions'].shape == (32, 4, 5, d)
        eps, starts = batch['episode'].numpy(), batch['start'].numpy()
        assert np.isin(eps, data.splits['train']).all()
        base = data.index.offsets[eps] + starts
        rows = base[:, None] + np.arange(20)[None]
        taken = table.take(rows.reshape(-1).tolist(), columns=['episode_idx', 'step_idx', 'action'])
        ep = np.asarray(taken.column('episode_idx').to_pylist()).reshape(32, 20)
        st = np.asarray(taken.column('step_idx').to_pylist()).reshape(32, 20)
        acts = np.asarray(taken.column('action').to_pylist(), dtype=np.float32).reshape(32, 4, 5, d)
        assert (ep == eps[:, None]).all()
        assert (st == starts[:, None] + np.arange(20)[None]).all()
        assert np.allclose(acts, batch['actions'].numpy())
        frames = table.take((base[:2, None] + 5 * np.arange(5)[None]).reshape(-1).tolist(), columns=['pixels'])
        from rdwm.exp1.data import decode_jpegs

        ref = decode_jpegs(frames.column('pixels').to_pylist()).reshape(2, 5, 3, hw, hw)
        assert torch.equal(ref, batch['pixels'][:2])
    assert seen == 6
