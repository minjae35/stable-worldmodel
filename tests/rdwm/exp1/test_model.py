import pytest
import torch

from rdwm.exp1.init import build_model, canonical_shared_state, shared_subset_hash
from rdwm.exp1.model import Dynamics, frame_causal_mask

from .conftest import make_batch

ENVS2 = ['pusht', 'cube']


@pytest.fixture(scope='module')
def shared(request):
    from rdwm.exp1.config import load_config

    return canonical_shared_state(load_config(), seed=0)


def test_token_shapes(cfg, shared):
    model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    pixels, actions = make_batch(cfg, 'pusht')
    tokens = model.encoder(pixels[:, 0])
    assert tokens.shape == (2, 256, 192)
    z = model.encode(pixels)
    assert z.shape == (2, 5, 64, 192)
    nxt = model.predict_next(z[:, :3], actions[:, :3], 'pusht')
    assert nxt.shape == (2, 64, 192)


def test_causal_mask_blocks_future_frames(cfg):
    torch.manual_seed(0)
    dyn = Dynamics(cfg['model'], tokens=4, envs=['pusht'], adapters=False).eval()
    for block in dyn.blocks:
        torch.nn.init.normal_(block.adaln[-1].weight, std=0.1)
    z = torch.randn(1, 3, 4, 192)
    cond = torch.randn(1, 3, 192)
    out = dyn(z, cond, 'pusht')
    z2 = z.clone()
    z2[:, 2] += 1.0
    cond2 = cond.clone()
    cond2[:, 2] += 1.0
    out2 = dyn(z2, cond2, 'pusht')
    assert torch.allclose(out[:, :2], out2[:, :2], atol=1e-6)
    assert not torch.allclose(out[:, 2], out2[:, 2])
    mask = frame_causal_mask(3, 2, 'cpu')
    assert mask[0, 1] and not mask[0, 2] and mask[5, 0]


def test_b_and_c_identical_at_init_fp32(cfg, shared):
    b_model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    c_model, report = build_model(cfg, 'C', ENVS2, 0, shared)
    assert shared_subset_hash(b_model) == shared_subset_hash(c_model)
    assert all(k.startswith('dynamics.adapters.') for k in report['own_init'])
    assert sorted(c_model.dynamics.adapters.keys()) == ['4', '5']
    for per_env in c_model.dynamics.adapters.values():
        assert sorted(per_env.keys()) == sorted(ENVS2)
        for adapter in per_env.values():
            assert adapter.up.weight.abs().max() == 0 and adapter.up.bias.abs().max() == 0
    for env in ENVS2:
        pixels, actions = make_batch(cfg, env, seed=3)
        for train_mode in (False, True):
            outs = []
            for model in (b_model, c_model):
                model.train(train_mode)
                torch.manual_seed(123)
                z = model.encode(pixels)
                preds = model.rollout(z[:, :3], actions[:, :2], actions[:, 2:4], env)
                outs.append((z, preds))
            assert (outs[0][0] - outs[1][0]).abs().max() < 1e-6
            assert (outs[0][1] - outs[1][1]).abs().max() < 1e-6


def test_pooled_d_matches_mean_of_b_tokens(cfg, shared):
    b_model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    d_model, report = build_model(cfg, 'D', ENVS2, 0, shared)
    assert report['own_init'] == ['dynamics.spatial_pos']
    pixels, actions = make_batch(cfg, 'cube')
    b_model.eval()
    d_model.eval()
    zb = b_model.encode(pixels)
    zd = d_model.encode(pixels)
    assert zd.shape == (2, 5, 1, 192)
    assert torch.allclose(zd, zb.mean(dim=2, keepdim=True), atol=1e-5)
    assert d_model.predict_next(zd[:, :3], actions[:, :3], 'cube').shape == (2, 1, 192)


def test_specialist_a_is_single_env_dense(cfg, shared):
    a_model, report = build_model(cfg, 'A', ['cube'], 0, shared)
    assert report['own_init'] == []
    assert list(a_model.action_adapter.keys()) == ['cube']
    assert len(a_model.dynamics.adapters) == 0 and not a_model.pooled
    b_model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    for key, value in a_model.state_dict().items():
        if not key.startswith('action_norm.'):
            assert torch.equal(value, b_model.state_dict()[key]), key


def test_adaln_last_linear_zero(cfg, shared):
    model, _ = build_model(cfg, 'C', ENVS2, 0, shared)
    for block in model.dynamics.blocks:
        assert block.adaln[-1].weight.abs().max() == 0
        assert block.adaln[-1].bias.abs().max() == 0


def test_env_embedding_not_in_encoder(cfg, shared):
    model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    pixels, _ = make_batch(cfg, 'pusht')
    model.eval()
    z1 = model.encode(pixels)
    with torch.no_grad():
        for p in model.env_embed.values():
            p.add_(1.0)
    assert torch.equal(z1, model.encode(pixels))


def test_action_normalized_once_inside_model(cfg, shared):
    model, _ = build_model(cfg, 'B', ENVS2, 0, shared)
    model.action_norm['pusht'].load([1.0, -2.0], [2.0, 4.0])
    a = torch.tensor([[[[3.0, 2.0]] * 5]])
    normed = model.action_norm['pusht'](a)
    assert torch.allclose(normed, torch.tensor([[[[1.0, 1.0]] * 5]]))
