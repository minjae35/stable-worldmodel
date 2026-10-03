import torch

from rdwm.exp1.init import build_model, canonical_shared_state
from rdwm.exp1.losses import SIGReg, recursive_loss, sigreg_groups

from .conftest import make_batch


def _model(cfg, variant='B'):
    shared = canonical_shared_state(cfg, 0)
    model, _ = build_model(cfg, variant, ['pusht', 'cube'], 0, shared)
    return model.eval()


def test_second_prediction_consumes_first_without_detach(cfg):
    model = _model(cfg)
    calls = []
    original = model.predict_next

    def spy(z, actions, env):
        out = original(z, actions, env)
        calls.append((z, out))
        return out

    model.predict_next = spy
    pixels, actions = make_batch(cfg, 'pusht')
    recursive_loss(model, pixels, actions, 'pusht', 3, SIGReg(), cfg)
    assert len(calls) == 2
    (z1, out1), (z2, _) = calls
    assert z1.shape[1] == 3 and z2.shape[1] == 3
    assert torch.equal(z2[:, -1], out1)
    assert out1.requires_grad
    grad = torch.autograd.grad(z2[:, -1].sum(), out1, retain_graph=True)[0]
    assert grad.abs().sum() > 0


def test_targets_get_gradient_and_history_drops_context(cfg):
    model = _model(cfg)
    # AdaLN-zero makes every dynamics block the identity at init, so earlier
    # context frames only matter once the gates move off zero.
    with torch.no_grad():
        for block in model.dynamics.blocks:
            block.adaln[-1].weight.normal_(std=0.05)
            block.adaln[-1].bias.normal_(std=0.05)
    pixels, actions = make_batch(cfg, 'cube')
    for history in (1, 2, 3):
        px = pixels.clone().requires_grad_(True)
        terms = recursive_loss(model, px, actions, 'cube', history, SIGReg(), cfg)
        terms['mse_t1'].backward()
        g = px.grad.abs().flatten(2).sum(-1).sum(0)
        dropped = 3 - history
        assert torch.all(g[:dropped] == 0)
        assert g[3] > 0
        assert torch.all(g[dropped:3] > 0)


def test_window_lengths_per_history(cfg):
    model = _model(cfg)
    lengths = []
    original = model.predict_next

    def spy(z, actions, env):
        lengths.append(z.shape[1])
        return original(z, actions, env)

    model.predict_next = spy
    pixels, actions = make_batch(cfg, 'pusht')
    for history in (1, 2, 3):
        recursive_loss(model, pixels, actions, 'pusht', history, SIGReg(), cfg)
    assert lengths == [1, 2, 2, 3, 3, 3]


def test_loss_composition(cfg):
    model = _model(cfg)
    pixels, actions = make_batch(cfg, 'pusht')
    torch.manual_seed(0)
    t = recursive_loss(model, pixels, actions, 'pusht', 3, SIGReg(), cfg)
    expect = 0.5 * t['mse_t1'] + 0.5 * t['mse_t2'] + 0.09 * t['sigreg']
    assert torch.allclose(t['loss'], expect)


def test_sigreg_groups_and_statistic():
    z = torch.randn(32, 4, 64, 192)
    g = sigreg_groups(z, 8)
    assert g.shape == (4 * 8, 32, 192)
    pooled = sigreg_groups(torch.randn(32, 4, 1, 192), 8)
    assert pooled.shape == (4, 32, 192)
    reg = SIGReg()
    torch.manual_seed(0)
    gaussian = reg(torch.randn(16, 256, 192))
    collapsed = reg(torch.randn(16, 1, 192).expand(16, 256, 192) * 0.01 + 3.0)
    assert gaussian < 0.2 * collapsed
