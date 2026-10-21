"""Compare DDP SIGReg with the single-process statistic on one batch.

Run with two GPUs:

    python -m torch.distributed.run --nproc_per_node=2 \\
        tests/wm/check_sigreg_ddp.py
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from stable_worldmodel.wm.loss import SIGReg


def _reference(module, samples, sig, projection):
    emb = module(samples)
    viewed = (emb @ projection).unsqueeze(-1) * sig.t
    err = (viewed.cos().mean(-3) - sig.phi).square()
    err = err + viewed.sin().mean(-3).square()
    loss = ((err @ sig.weights) * emb.shape[1]).mean()
    loss.backward()
    return loss.detach(), module.weight.grad.detach().clone()


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    layer = nn.Linear(8, 8, bias=False).cuda()
    reference_layer = nn.Linear(8, 8, bias=False).cuda()
    with torch.no_grad():
        reference_layer.weight.copy_(layer.weight)
    for param in layer.parameters():
        dist.broadcast(param.data, src=0)
    with torch.no_grad():
        reference_layer.weight.copy_(layer.weight)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(1)
    full = torch.randn(2, 4 * world, 8, generator=generator)
    local_n = full.shape[1] // world
    local = full[:, rank * local_n : (rank + 1) * local_n].cuda()
    projection = torch.randn(8, 4, generator=generator)
    projection = projection / projection.norm(dim=0)
    projection = projection.cuda()
    dist.broadcast(projection, src=0)

    if rank == 0:
        ref_loss, ref_grad = _reference(
            reference_layer,
            full.cuda(),
            SIGReg(knots=9, num_proj=4).cuda(),
            projection,
        )
    dist.barrier()

    real_randn = torch.randn

    def shared_randn(*args, **kwargs):
        return projection.detach().clone()

    torch.randn = shared_randn
    try:
        ddp = DDP(layer, device_ids=[rank])
        sig = SIGReg(knots=9, num_proj=4).cuda()
        loss = sig(ddp(local))
        loss.backward()
    finally:
        torch.randn = real_randn

    loss_sum = loss.detach().clone()
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    grad = layer.weight.grad.detach().clone()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    # DDP already averaged. Summing again multiplies by world size,
    # so divide to recover the averaged parameter gradient once.
    grad = grad / world

    if rank == 0:
        got = loss.detach()
        scale = got / ref_loss
        grad_scale = (grad - ref_grad).abs().max()
        print(
            f'SIGREG_DDP loss={float(got):.8f} '
            f'ref={float(ref_loss):.8f} ratio={float(scale):.6f}',
            flush=True,
        )
        err = float(grad_scale)
        print(f'SIGREG_DDP max_grad_abs_err={err:.8e}', flush=True)
        ok = torch.allclose(got, ref_loss, rtol=1e-4, atol=1e-5)
        ok = ok and torch.allclose(grad, ref_grad, rtol=1e-4, atol=1e-5)
        print('SIGREG_DDP', 'PASS' if ok else 'FAIL', flush=True)
    else:
        ok = True
    flag = torch.zeros(1, device=f'cuda:{rank}')
    if rank == 0 and not ok:
        flag.fill_(1)
    dist.broadcast(flag, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if flag.item():
        raise SystemExit(1)


if __name__ == '__main__':
    if 'RANK' not in os.environ:
        raise SystemExit('launch with torch.distributed.run')
    main()
