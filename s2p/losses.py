"""Losses added by S2P on top of the base UDA method, plus the KD baselines."""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "KD_INPUTS",
    "ATLoss",
    "LwFLoss",
    "MMDLoss",
    "SPLoss",
    "SimilarityPreservationLoss",
    "build_kd_loss",
]


class SimilarityPreservationLoss(nn.Module):
    """S2P distillation: keep the pairwise similarities of the support set.

    ``student`` and ``teacher`` are (N, D) features of the same N support
    images. Both are L2-normalised, their N x N cosine-similarity matrices are
    compared, and the loss is ``sqrt(N * ||G_s - G_t||_F^2 + eps)``.

    The factor N reproduces the loss scale of the original implementation,
    which summed the squared difference over an (N, N, N) expansion of the two
    matrices; the published loss weights were tuned with it.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, student, teacher):
        student = F.normalize(student, p=2, dim=1)
        teacher = F.normalize(teacher, p=2, dim=1)
        gram_student = student @ student.t()
        gram_teacher = teacher @ teacher.t()
        n = student.size(0)
        return torch.sqrt(n * (gram_student - gram_teacher).pow(2).sum() + self.eps)


class MMDLoss(nn.Module):
    """Multi-kernel (Gaussian) maximum mean discrepancy between two batches."""

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        l2_distance = (total.unsqueeze(0) - total.unsqueeze(1)).pow(2).sum(2)
        # bandwidth from the mean pairwise distance, treated as a constant
        bandwidth = l2_distance.detach().sum() / (n_samples ** 2 - n_samples)
        bandwidth = bandwidth / self.kernel_mul ** (self.kernel_num // 2)
        return sum(
            torch.exp(-l2_distance / (bandwidth * self.kernel_mul ** i))
            for i in range(self.kernel_num)
        )

    def forward(self, source, target):
        n = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        xx = kernels[:n, :n]
        yy = kernels[n:, n:]
        xy = kernels[:n, n:]
        yx = kernels[n:, :n]
        return torch.mean(xx + yy - xy - yx)


class LwFLoss(nn.Module):
    """Baseline: logit distillation (Hinton et al. 2015, as used by LwF)."""

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        t = self.temperature
        return (
            F.kl_div(
                F.log_softmax(student_logits / t, dim=1),
                F.softmax(teacher_logits / t, dim=1),
                reduction="batchmean",
            )
            * t
            * t
        )


class SPLoss(nn.Module):
    """Baseline: Similarity-Preserving KD (Tung & Mori, ICCV 2019)."""

    def forward(self, student_maps, teacher_maps):
        student = student_maps.reshape(student_maps.size(0), -1)
        teacher = teacher_maps.reshape(teacher_maps.size(0), -1)
        gram_student = F.normalize(student @ student.t(), p=2, dim=1)
        gram_teacher = F.normalize(teacher @ teacher.t(), p=2, dim=1)
        return F.mse_loss(gram_student, gram_teacher)


class ATLoss(nn.Module):
    """Baseline: Attention Transfer (Zagoruyko & Komodakis, ICLR 2017)."""

    def __init__(self, p=2, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def attention_map(self, feature_maps):
        am = feature_maps.abs().pow(self.p).sum(dim=1, keepdim=True)
        return am / (torch.norm(am, dim=(2, 3), keepdim=True) + self.eps)

    def forward(self, student_maps, teacher_maps):
        return F.mse_loss(
            self.attention_map(student_maps), self.attention_map(teacher_maps)
        )


# which network output each distillation loss is computed on
KD_INPUTS = {"s2p": "feat", "lwf": "prob", "sp": "activation", "at": "activation"}

_KD_LOSSES = {
    "s2p": SimilarityPreservationLoss,
    "lwf": LwFLoss,
    "sp": SPLoss,
    "at": ATLoss,
}


def build_kd_loss(name):
    if name not in _KD_LOSSES:
        raise KeyError(
            f"Unknown distillation loss '{name}', expected one of {sorted(_KD_LOSSES)}"
        )
    return _KD_LOSSES[name]()
