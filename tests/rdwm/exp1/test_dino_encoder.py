"""Wiring test for the frozen-encoder prototype. Does not download DINOv2."""

import torch
from torch import nn

from rdwm.exp1.losses import SIGReg, recursive_loss
from rdwm.exp1.model import RDWM

from .conftest import make_batch


class _FakeFrozen(nn.Module):
    frozen = True

    def __init__(self):
        super().__init__()
        self.grid = 16
        self.width = 384
        self.mark = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.mark.weight.fill_(0.5)
        self.requires_grad_(False)

    def forward(self, pixels):
        b = pixels.shape[0]
        return torch.zeros(b, self.grid * self.grid, self.width, device=pixels.device)


def test_frozen_encoder_keeps_pretrained_weights_and_64_tokens(cfg):
    model = RDWM(cfg, 'A', ['pusht'], {'pusht': 2}, encoder=_FakeFrozen())
    assert model.visual_encoder == 'frozen_dinov2_small'
    assert model.spatial_tokens == 64
    assert model.projector.net[0].in_features == 384
    assert model.projector.net[-1].out_features == 192
    assert torch.equal(model.encoder.mark.weight, torch.full((4, 4), 0.5))
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not model.dynamics.adapters

    pixels, actions = make_batch(cfg, 'pusht')
    model.train()
    assert not model.encoder.training
    z = model.encode(pixels)
    assert z.shape == (2, 5, 64, 192)
    sigreg = SIGReg()
    terms = recursive_loss(model, pixels, actions, 'pusht', 3, sigreg, cfg)
    terms['loss'].backward()
    assert model.encoder.mark.weight.grad is None
    assert model.projector.net[0].weight.grad is not None
    assert model.dynamics.blocks[0].attn.qkv.weight.grad is not None


def test_default_encoder_stays_trainable_vit(cfg):
    model = RDWM(cfg, 'A', ['pusht'], {'pusht': 2})
    assert model.visual_encoder == 'vit'
    assert model.projector.net[0].in_features == 192
    assert any(p.requires_grad for p in model.encoder.parameters())
