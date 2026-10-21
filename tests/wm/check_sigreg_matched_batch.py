"""Same global batch: one process versus 4-GPU DDP.

Global batch is 32, split into 8 samples per rank. This is setting B's
per-environment batch. The single-process run sees all 32.

    python -m torch.distributed.run --nproc_per_node=4 \
        tests/wm/check_sigreg_matched_batch.py
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

import stable_worldmodel.wm.loss as loss_mod
from stable_worldmodel.wm.loss import SIGReg

GLOBAL = 32
DIM = 64
STEPS_T = 4
PROJ = 1024
KNOTS = 17
WEIGHT = 0.09
CLIP = 1.0
LR = 5e-5
WD = 1e-3


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.head = nn.Linear(DIM, DIM, bias=False)

    def forward(self, samples):
        emb = self.proj(samples)
        return emb, self.head(emb)


def _name(name):
    return name[7:] if name.startswith('module.') else name


def _grads(module):
    return {
        _name(name): param.grad.detach().float().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }


def _objective(module, samples, target, sig, weight):
    emb, pred = module(samples)
    pred_loss = (pred - target).pow(2).mean()
    sig_loss = sig(emb)
    return pred_loss, sig_loss, pred_loss + weight * sig_loss


def _apply_update(module, loss):
    module.zero_grad(set_to_none=True)
    # The unscaled toy loss can sit under the clip threshold. Both
    # sides use the same factor, so the gradient comparison is unchanged
    # and clipping has to fire.
    (loss * 50).backward()
    raw = _grads(module)
    total = clip_grad_norm_(module.parameters(), CLIP)
    clipped = _grads(module)
    opt = torch.optim.AdamW(module.parameters(), lr=LR, weight_decay=WD)
    before = {
        _name(name): param.detach().float().clone()
        for name, param in module.named_parameters()
    }
    opt.step()
    delta = {
        _name(name): param.detach().float() - before[_name(name)]
        for name, param in module.named_parameters()
    }
    return raw, float(total), clipped, delta


def _compare(name, got, ref, rtol, atol):
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    return ok, err


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise SystemExit(f'expected 4 processes, got {world}')
    torch.cuda.set_device(rank)
    local_n = GLOBAL // world

    torch.manual_seed(0)
    module = Tiny().cuda()
    for param in module.parameters():
        dist.broadcast(param.data, src=0)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(1)
    full = torch.randn(STEPS_T, GLOBAL, DIM, generator=generator)
    target = torch.randn(STEPS_T, GLOBAL, DIM, generator=generator)
    projection = torch.randn(DIM, PROJ, generator=generator)
    local = full[:, rank * local_n : (rank + 1) * local_n].cuda()
    local_target = target[:, rank * local_n : (rank + 1) * local_n].cuda()

    real_randn = torch.randn

    def shared_randn(*args, **kwargs):
        out = projection.detach().clone()
        device = kwargs.get('device', None)
        if device is not None:
            out = out.to(device)
        return out

    real_distributed = loss_mod._distributed

    def run_once(precision):
        work = Tiny().cuda()
        work.load_state_dict(module.state_dict())
        reference = Tiny().cuda()
        reference.load_state_dict(module.state_dict())
        autocast = torch.autocast(
            'cuda',
            dtype=torch.bfloat16,
            enabled=precision == 'bf16',
        )
        torch.randn = shared_randn
        try:
            if rank == 0:
                # The process group is already up. SIGReg would all-reduce
                # here, and the other ranks are not in that call.
                loss_mod._distributed = lambda: False
                with autocast:
                    ref_pred, ref_sig, ref_loss = _objective(
                        reference,
                        full.cuda(),
                        target.cuda(),
                        SIGReg(knots=KNOTS, num_proj=PROJ).cuda(),
                        WEIGHT,
                    )
                ref_raw, ref_norm, ref_clip, ref_delta = _apply_update(
                    reference, ref_loss
                )
                bare = Tiny().cuda()
                bare.load_state_dict(module.state_dict())
                with autocast:
                    _, _, bare_loss = _objective(
                        bare,
                        full.cuda(),
                        target.cuda(),
                        SIGReg(knots=KNOTS, num_proj=PROJ).cuda(),
                        0.0,
                    )
                _, _, _, bare_delta = _apply_update(bare, bare_loss)
                loss_mod._distributed = real_distributed
            dist.barrier()

            ddp = DDP(work, device_ids=[rank])
            with autocast:
                pred, sig, loss = _objective(
                    ddp,
                    local,
                    local_target,
                    SIGReg(knots=KNOTS, num_proj=PROJ).cuda(),
                    WEIGHT,
                )
            raw, total, clipped, delta = _apply_update(ddp, loss)
        finally:
            torch.randn = real_randn
            loss_mod._distributed = real_distributed

        pred_mean = pred.detach().float().clone()
        dist.all_reduce(pred_mean, op=dist.ReduceOp.AVG)
        sig_value = sig.detach().float()

        reduced_raw = {
            name: value.clone() for name, value in raw.items()
        }
        reduced_clip = {
            name: value.clone() for name, value in clipped.items()
        }
        reduced_delta = {
            name: value.clone() for name, value in delta.items()
        }
        for bucket in (reduced_raw, reduced_clip, reduced_delta):
            for value in bucket.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value.div_(world)

        if rank != 0:
            return True

        rtol, atol = (1e-4, 1e-5) if precision == 'fp32' else (2e-2, 2e-2)
        ok = True
        sig_err = (sig_value - ref_sig.detach().float()).abs().item()
        pred_err = (pred_mean - ref_pred.detach().float()).abs().item()
        sig_ok = torch.allclose(
            sig_value, ref_sig.detach().float(), rtol=rtol, atol=atol
        )
        pred_ok = torch.allclose(
            pred_mean, ref_pred.detach().float(), rtol=rtol, atol=atol
        )
        ok = ok and bool(sig_ok and pred_ok)
        ref_pred_f = float(ref_pred.detach().float())
        ref_sig_f = float(ref_sig.detach().float())
        ref_ratio = (WEIGHT * ref_sig_f) / max(abs(ref_pred_f), 1e-12)
        got_ratio = (WEIGHT * float(sig_value)) / max(
            abs(float(pred_mean)), 1e-12
        )
        print(
            f'MATCH {precision} sigreg={float(sig_value):.8f} '
            f'ref={ref_sig_f:.8f} abs_err={sig_err:.8e}',
            flush=True,
        )
        print(
            f'MATCH {precision} pred={float(pred_mean):.8f} '
            f'ref={ref_pred_f:.8f} abs_err={pred_err:.8e}',
            flush=True,
        )
        print(
            f'MATCH {precision} weighted_sigreg/pred '
            f'ddp={got_ratio:.6f} ref={ref_ratio:.6f}',
            flush=True,
        )
        grad_err = 0.0
        for name, ref in ref_raw.items():
            good, err = _compare(name, reduced_raw[name], ref, rtol, atol)
            grad_err = max(grad_err, err)
            ok = ok and good
            scale = reduced_raw[name].norm() / ref.norm().clamp_min(1e-12)
            print(
                f'MATCH {precision} raw_grad {name} '
                f'abs_err={err:.8e} norm_ratio={float(scale):.6f}',
                flush=True,
            )
        clip_err = 0.0
        for name, ref in ref_clip.items():
            good, err = _compare(name, reduced_clip[name], ref, rtol, atol)
            clip_err = max(clip_err, err)
            ok = ok and good
        delta_err = 0.0
        sig_moves = False
        for name, ref in ref_delta.items():
            good, err = _compare(name, reduced_delta[name], ref, rtol, atol)
            delta_err = max(delta_err, err)
            ok = ok and good
            sig_moves = sig_moves or not torch.allclose(
                ref, bare_delta[name], rtol=1e-4, atol=1e-6
            )
        print(
            f'MATCH {precision} pre_clip_norm ddp={total:.6f} '
            f'ref={ref_norm:.6f}',
            flush=True,
        )
        post = clip_grad_norm_scale(reduced_clip)
        post_ref = clip_grad_norm_scale(ref_clip)
        print(
            f'MATCH {precision} post_clip_norm ddp={post:.6f} '
            f'ref={post_ref:.6f} clip_abs_err={clip_err:.8e}',
            flush=True,
        )
        print(
            f'MATCH {precision} update_abs_err={delta_err:.8e} '
            f'sigreg_changes_update={sig_moves}',
            flush=True,
        )
        clipped = total > CLIP + 1e-3 and abs(post - CLIP) < 1e-3
        clipped = clipped and abs(post_ref - CLIP) < 1e-3
        print(
            f'MATCH {precision} clip_fired={clipped}',
            flush=True,
        )
        if precision == 'fp32':
            ok = ok and sig_moves and clipped
        print(
            f'MATCH {precision}',
            'PASS' if ok else 'FAIL',
            flush=True,
        )
        return ok

    fine = run_once('fp32')
    fine = run_once('bf16') and fine
    flag = torch.zeros(1, device=f'cuda:{rank}')
    if rank == 0 and not fine:
        flag.fill_(1)
    dist.broadcast(flag, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if flag.item():
        raise SystemExit(1)


def clip_grad_norm_scale(grads):
    square = 0.0
    for value in grads.values():
        square += float(value.float().pow(2).sum())
    return square**0.5


if __name__ == '__main__':
    if 'RANK' not in os.environ:
        raise SystemExit('launch with torch.distributed.run')
    main()
