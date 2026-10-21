"""Stage 2-A generalist: shared trunk, separate action adapters."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.data.multienv import (
    BalancedEnvBatchSampler,
    ConcatEnvDataset,
    EnvTaggedDataset,
    collate_by_env,
)
from stable_worldmodel.wm.lewm.generalist import (
    GeneralistLeWM,
    generalist_loss,
    variance_hinge,
)
from stable_worldmodel.wm.lewm.module import Embedder, Predictor
from stable_worldmodel.wm.loss import SIGReg


class _StubOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _StubEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.float().mean(dim=(2, 3))
        hidden = self.proj(pooled)[:, None, :]
        return _StubOutput(hidden)


def _model():
    dim = 8
    adapters = {
        'pusht': Embedder(input_dim=10, emb_dim=dim),
        'tworoom': Embedder(input_dim=10, emb_dim=dim),
        'cube': Embedder(input_dim=25, emb_dim=dim),
    }
    predictor = Predictor(
        num_frames=3,
        depth=1,
        heads=2,
        mlp_dim=16,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        dim_head=4,
        dropout=0.0,
    )
    return GeneralistLeWM(
        encoder=_StubEncoder(dim),
        predictor=predictor,
        action_adapters=adapters,
        env_names=['pusht', 'tworoom', 'cube'],
        active_env='pusht',
    )


def _batch(per_env=2, steps=4):
    batch = {}
    widths = {'pusht': 10, 'tworoom': 10, 'cube': 25}
    for name, width in widths.items():
        batch[name] = {
            'pixels': torch.rand(per_env, steps, 3, 8, 8),
            'action': torch.randn(per_env, steps, width),
        }
    return batch


def test_adapters_are_separate_and_trunk_is_shared():
    model = _model()
    pusht = model.action_encoder.adapters['pusht']
    cube = model.action_encoder.adapters['cube']
    assert pusht is not cube
    assert pusht.patch_embed.in_channels == 10
    assert cube.patch_embed.in_channels == 25
    assert model.encoder is model.encoder
    assert model.predictor is model.predictor
    # One embedding table, one row per environment.
    assert model.env_embed.weight.shape == (3, 8)


def test_loss_backward_reaches_shared_trunk_and_every_adapter():
    torch.manual_seed(0)
    model = _model()
    model.train()
    # AdaLN-zero starts at 0, so the action path has no gradient until
    # the modulation layer moves. Open it for this connectivity check.
    for block in model.predictor.transformer.layers:
        linear = block.adaLN_modulation[-1]
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.normal_(linear.bias, std=0.02)
    sigreg = SIGReg(knots=5, num_proj=8)
    loss, logs = generalist_loss(
        model,
        sigreg,
        _batch(),
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
    )
    assert torch.isfinite(loss)
    for name in ('pusht', 'tworoom', 'cube'):
        assert torch.isfinite(logs[f'{name}/pred_loss'])
        assert torch.isfinite(logs[f'{name}/sigreg_loss'])
        assert torch.isfinite(logs[f'{name}/latent_dim_std_min'])
    loss.backward()
    assert model.encoder.proj.weight.grad is not None
    assert model.encoder.proj.weight.grad.abs().sum() > 0
    modulation = model.predictor.transformer.layers[0].adaLN_modulation[-1]
    assert modulation.weight.grad is not None
    assert modulation.weight.grad.abs().sum() > 0
    assert model.env_embed.weight.grad.abs().sum() > 0
    for name in ('pusht', 'tworoom', 'cube'):
        grad = model.action_encoder.adapters[name].embed[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_env_embedding_changes_the_rollout():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    # AdaLN-zero ignores the condition until the modulation layer is nonzero.
    for block in model.predictor.transformer.layers:
        linear = block.adaLN_modulation[-1]
        nn.init.normal_(linear.weight, std=0.5)
        nn.init.normal_(linear.bias, std=0.5)
    pixels = torch.rand(1, 2, 3, 3, 8, 8)
    info = {
        'pixels': pixels,
        'action_history': torch.randn(1, 2, 2, 10),
    }
    actions = torch.randn(1, 2, 2, 10)
    model.set_active_env('pusht')
    with torch.no_grad():
        first = model.rollout(dict(info), actions)['predicted_emb'].clone()
    # Same pixels and actions, different environment row.
    model.env_embed.weight.data[1] = model.env_embed.weight.data[0] + 1
    model.set_active_env('tworoom')
    with torch.no_grad():
        second = model.rollout(dict(info), actions)['predicted_emb']
    assert first.shape == (1, 2, 5, 8)
    assert not torch.allclose(first, second)


def test_cube_rollout_accepts_blocked_width_25():
    torch.manual_seed(0)
    model = _model()
    model.set_active_env('cube')
    info = {
        'pixels': torch.rand(1, 1, 3, 3, 8, 8),
        'action_history': torch.randn(1, 1, 2, 25),
    }
    actions = torch.randn(1, 1, 2, 25)
    out = model.rollout(info, actions)
    assert out['predicted_emb'].shape == (1, 1, 5, 8)


def test_sharded_batches_partition_one_global_epoch():
    lengths = {'pusht': 40, 'tworoom': 50, 'cube': 60}
    per_env = 4
    world = 4
    whole = list(
        BalancedEnvBatchSampler(
            lengths, per_env=per_env * world, seed=5, world_size=1
        )
    )
    parts = [
        list(
            BalancedEnvBatchSampler(
                lengths,
                per_env=per_env,
                seed=5,
                rank=rank,
                world_size=world,
            )
        )
        for rank in range(world)
    ]
    assert len(whole) == 2
    assert [len(rank_batches) for rank_batches in parts] == [2, 2, 2, 2]
    for step, batch in enumerate(whole):
        covered = []
        for rank_batches in parts:
            covered.extend(rank_batches[step])
        assert len(covered) == len(set(covered)) == len(batch)
        assert sorted(covered) == sorted(batch)


def test_distributed_sampler_shards_without_overlap():
    lengths = {'pusht': 16, 'tworoom': 20, 'cube': 24}
    ranks = [
        list(
            BalancedEnvBatchSampler(
                lengths, per_env=4, seed=3, rank=rank, world_size=2
            )
        )
        for rank in (0, 1)
    ]
    assert len(ranks[0]) == len(ranks[1]) == 2
    for left, right in zip(ranks[0], ranks[1]):
        assert set(left).isdisjoint(right)
        assert len(left) == len(right) == 12


def test_sigreg_global_sums_match_the_full_batch():
    from stable_worldmodel.wm.loss import SIGReg, epps_pulley

    torch.manual_seed(0)
    sig = SIGReg(knots=17, num_proj=8)
    proj = torch.randn(3, 8, 6)
    projection = torch.randn(6, 8)
    projection = projection / projection.norm(p=2, dim=0)
    full = (proj @ projection).unsqueeze(-1) * sig.t
    reference = (full.cos().mean(-3) - sig.phi).square()
    reference = reference + full.sin().mean(-3).square()
    reference = ((reference @ sig.weights) * proj.shape[1]).mean()
    left, right = proj[:, :3], proj[:, 3:]

    def sums(part):
        viewed = (part @ projection).unsqueeze(-1) * sig.t
        return viewed.cos().sum(1), viewed.sin().sum(1), part.shape[1]

    cos_a, sin_a, n_a = sums(left)
    cos_b, sin_b, n_b = sums(right)
    merged = epps_pulley(
        cos_a + cos_b, sin_a + sin_b, n_a + n_b, sig.phi, sig.weights
    )
    assert torch.allclose(merged, reference, rtol=1e-5, atol=1e-5)


def test_balanced_sampler_counts_each_environment():
    lengths = {'pusht': 10, 'tworoom': 20, 'cube': 30}
    sampler = BalancedEnvBatchSampler(lengths, per_env=4, seed=0)
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 12
        pusht = [i for i in batch if i < 10]
        tworoom = [i for i in batch if 10 <= i < 30]
        cube = [i for i in batch if i >= 30]
        assert len(pusht) == len(tworoom) == len(cube) == 4


def test_collate_keeps_action_widths():
    samples = []
    for name, width in (('pusht', 10), ('cube', 25)):
        for _ in range(2):
            samples.append(
                {
                    'env_name': name,
                    'pixels': torch.zeros(4, 3, 8, 8),
                    'action': torch.zeros(4, width),
                }
            )
    batch = collate_by_env(samples)
    assert batch['pusht']['action'].shape == (2, 4, 10)
    assert batch['cube']['action'].shape == (2, 4, 25)


def _write_toy_h5(path: Path, action_dim: int):
    rng = np.random.default_rng(0)
    ep_len = 40
    with HDF5Writer(path, mode='overwrite') as writer:
        for _ in range(2):
            pixels = rng.integers(
                0, 255, size=(ep_len, 16, 16, 3), dtype=np.uint8
            )
            actions = rng.uniform(-1, 1, size=(ep_len, action_dim)).astype(
                np.float32
            )
            writer.write_episode(
                {
                    'pixels': [pixels[t] for t in range(ep_len)],
                    'action': [actions[t] for t in range(ep_len)],
                }
            )


def test_frameskip_packs_raw_actions(tmp_path):
    path = tmp_path / 'toy.h5'
    _write_toy_h5(path, action_dim=5)
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    dataset = HDF5Dataset(
        path=path,
        frameskip=5,
        num_steps=4,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    sample = dataset[0]
    assert tuple(sample['pixels'].shape) == (4, 3, 16, 16)
    assert tuple(sample['action'].shape) == (4, 25)
    ep, start = dataset.clip_indices[0]
    raw = dataset.get_col_data('action')
    window = raw[int(dataset.offsets[ep]) + start :][: 4 * 5]
    packed = torch.from_numpy(np.asarray(window)).reshape(4, 25)
    torch.testing.assert_close(sample['action'], packed)


@pytest.mark.skipif(
    not Path('/workspace/datasets/lewm-pusht/train.lance').exists(),
    reason='local lewm-pusht table is not on disk',
)
def test_pusht_lance_action_block_matches_frameskip():
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        '/workspace/datasets/lewm-pusht/train.lance',
        num_steps=4,
        frameskip=5,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    assert dataset.get_dim('action') == 2
    sample = dataset[0]
    assert tuple(sample['action'].shape) == (4, 10)
    assert sample['pixels'].shape[0] == 4
    ep, start = dataset.clip_indices[0]
    raw = np.asarray(dataset.get_col_data('action'))
    window = raw[int(dataset.offsets[ep]) + start :][:20]
    packed = torch.from_numpy(window.reshape(4, 10))
    torch.testing.assert_close(sample['action'].float(), packed.float())


def test_variance_hinge_penalizes_only_flat_dimensions():
    # Batch std is 1 on dim 0 and 0 on dim 1, at both timesteps.
    emb = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [2.0, 0.0]],
        ]
    )
    loss = variance_hinge(emb, gamma=1.0, eps=0.0)
    assert torch.allclose(loss, torch.tensor(0.5))
    assert variance_hinge(emb * 10, gamma=1.0, eps=0.0).item() == 0.5


def test_variance_term_and_detached_target_stay_finite():
    torch.manual_seed(0)
    model = _model()
    model.train()
    sigreg = SIGReg(knots=5, num_proj=8)
    batch = _batch()
    plain, plain_logs = generalist_loss(
        model,
        sigreg,
        batch,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
    )
    hinged, hinged_logs = generalist_loss(
        model,
        sigreg,
        batch,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
        variance_weight=1.0,
        variance_gamma=1.0,
    )
    detached, detached_logs = generalist_loss(
        model,
        sigreg,
        batch,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
        detach_pred_target=True,
    )
    assert torch.isfinite(plain) and torch.isfinite(hinged)
    assert torch.isfinite(detached)
    assert hinged_logs['variance_loss'] > 0
    assert plain_logs['variance_loss'].item() == 0
    assert hinged_logs['pusht/variance_loss'] > 0
    detached.backward()
    assert model.encoder.proj.weight.grad is not None


def test_concat_dataset_tags_environment():
    class _Tiny(torch.utils.data.Dataset):
        def __init__(self, n, width):
            self.n = n
            self.width = width

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {'action': torch.full((self.width,), idx)}

    tagged = {
        'pusht': EnvTaggedDataset(_Tiny(2, 10), 'pusht'),
        'cube': EnvTaggedDataset(_Tiny(3, 25), 'cube'),
    }
    data = ConcatEnvDataset(tagged)
    assert data[0]['env_name'] == 'pusht'
    assert data[2]['env_name'] == 'cube'
    assert tuple(data[2]['action'].shape) == (25,)
