"""Forward/backward smoke on GPU with the real batch size and BF16 autocast."""

import pytest
import torch

from rdwm.exp1.data import preprocess
from rdwm.exp1.init import build_model, canonical_shared_state, make_optimizer
from rdwm.exp1.losses import SIGReg, recursive_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

CASES = [('A', ['cube']), ('B', ['pusht', 'cube']), ('C', ['pusht', 'cube']), ('D', ['pusht', 'cube'])]


@pytest.mark.parametrize('variant,envs', CASES)
def test_train_step(cfg, variant, envs):
    device = torch.device('cuda')
    shared = canonical_shared_state(cfg, 0)
    model, _ = build_model(cfg, variant, envs, 0, shared)
    model.to(device).train()
    opt, sched = make_optimizer(model, cfg, total_updates=100)
    sigreg = SIGReg().to(device)
    torch.cuda.reset_peak_memory_stats()
    for step, env in enumerate(envs * 2):
        d = cfg['envs'][env]['action_dim']
        raw = torch.randint(0, 256, (32, 5, 3, 224, 224), dtype=torch.uint8, device=device)
        pixels = preprocess(raw, cfg)
        actions = torch.randn(32, 4, 5, d, device=device)
        terms = recursive_loss(model, pixels, actions, env, (3, 2, 1)[step % 3], sigreg, cfg)
        opt.zero_grad(set_to_none=True)
        terms['loss'].backward()
        assert torch.isfinite(terms['loss'])
        enc = sum(p.grad.float().norm() for p in model.encoder.parameters() if p.grad is not None)
        assert enc > 0
        if variant == 'C':
            ups = [a[env].up.weight.grad for a in model.dynamics.adapters.values()]
            assert all(g is not None and g.abs().sum() > 0 for g in ups)
            other = [e for e in envs if e != env][0]
            assert all(a[other].up.weight.grad is None for a in model.dynamics.adapters.values())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        opt.step()
        sched.step()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f'{variant}: peak {peak:.2f} GiB')
    assert peak < 30
