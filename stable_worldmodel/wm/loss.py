import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import einsum


def _distributed() -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def epps_pulley(cos_sum, sin_sum, n: int, phi, weights):
    """Epps-Pulley statistic from summed cos/sin, not from per-rank means.

    ``n`` is the global batch. Averaging per-rank SIGReg values is a
    different estimator: the characteristic function is a mean, and the
    statistic then multiplies by the sample count.
    """
    cos_mean = cos_sum / n
    sin_mean = sin_sum / n
    err = (cos_mean - phi).square() + sin_mean.square()
    return ((err @ weights) * n).mean()


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    One process uses the original local formula. Under DDP every rank
    shares one projection and reduces the cos/sin sums, then applies
    the same Epps-Pulley formula to the global batch.
    Reference: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)
        self.last_comm_s = 0.0
        self._comm_events: list[tuple] = []

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # Every rank draws so the RNG stays aligned, then rank 0's draw
        # is shared. A different A per rank is a different statistic.
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        if _distributed():
            dist.broadcast(A, src=0)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        if not _distributed():
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
                -3
            ).square()
            statistic = (err @ self.weights) * proj.size(-2)
            self.last_comm_s = 0.0
            return statistic.mean()
        return self._global_epps(x_t)

    def _global_epps(self, x_t: torch.Tensor) -> torch.Tensor:
        start = end = None
        if x_t.is_cuda:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
        cos_sum = _SumAllReduce.apply(x_t.cos().sum(dim=1))
        sin_sum = _SumAllReduce.apply(x_t.sin().sum(dim=1))
        with torch.no_grad():
            count = torch.tensor(
                float(x_t.shape[1]), device=x_t.device, dtype=torch.float32
            )
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if end is not None:
            end.record()
            self._comm_events.append((start, end))
        n = int(round(count.item()))
        return epps_pulley(cos_sum, sin_sum, n, self.phi, self.weights)

    def read_comm_s(self) -> float:
        """Sum the SIGReg reductions since the previous read.

        One joint step calls this once per environment. Keeping only the
        last pair would drop the other environments.
        """
        total = 0.0
        for start, end in self._comm_events:
            end.synchronize()
            total += start.elapsed_time(end) / 1000.0
        self._comm_events.clear()
        self.last_comm_s = total
        return total


class _SumAllReduce(torch.autograd.Function):
    """Sum-allreduce whose backward also sums.

    Every rank evaluates the same global Epps-Pulley statistic, so each
    local ``grad_output`` is ``∂L/∂(global sum)``. Summing those copies
    multiplies the embedding gradient by the world size. DDP then
    averages parameter gradients, and the product is the single-process
    gradient on the concatenated batch.
    """

    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDMLoss(torch.nn.Module):
    """VCReg anti-collapse + Temporal Alignment + Inverse Dynamics Modeling losses
    reference: https://arxiv.org/abs/2502.14819
    """

    def __init__(self):
        super().__init__()
        self.vc_reg = VCReg()

    def forward(self, z, a_pred=None, a_target=None):
        """
        z: (B, T, D)
        a_pred: (B, T-1, A)
        a_target: (B, T-1, A)
        """

        output = {}
        if a_pred is not None and a_target is not None:
            output['idm_loss'] = F.mse_loss(a_pred, a_target)

        output['temp_align_loss'] = F.mse_loss(z[:, :-1], z[:, 1:])  # detach?
        output.update(self.vc_reg(z))

        return output


class TemporalStraighteningLoss(torch.nn.Module):
    """Temporal Straightening Loss Module (Mean Pairwise Negative Cosine Similarity)
    reference: https://arxiv.org/abs/2603.12231
    """

    def __init__(self):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        v = x[:, 1:] - x[:, :-1]  # velocities
        sim = self.cos_sim(v[:, :-1], v[:, 1:])
        return -sim.mean()


__all__ = [
    'PLDMLoss',
    'SIGReg',
    'TemporalStraighteningLoss',
    'VCReg',
]
