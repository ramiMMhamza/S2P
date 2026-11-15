import pytest
import torch
import torch.nn.functional as F

from s2p.losses import ATLoss, LwFLoss, MMDLoss, SimilarityPreservationLoss, SPLoss, build_kd_loss


def reference_similarity_preservation(student, teacher):
    """The loss as written in the code released with the paper."""
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher, p=2, dim=1)
    r_student = torch.mm(student, student.T)
    r_teacher = torch.mm(teacher, teacher.T)
    n = r_student.size(0)
    r_student_ = r_student.unsqueeze(0).expand(n, n, n)
    r_teacher_ = r_teacher.unsqueeze(0).expand(n, n, n)
    return torch.sqrt(((r_student_ - r_teacher_) ** 2).sum() + 0.00001)


def reference_mmd(source, target, kernel_mul=2.0, kernel_num=5):
    """The MMD loss as written in the code released with the paper."""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2 = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(l2.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernels = sum(torch.exp(-l2 / (bandwidth * kernel_mul ** i)) for i in range(kernel_num))
    b = int(source.size()[0])
    return torch.mean(kernels[:b, :b] + kernels[b:, b:] - kernels[:b, b:] - kernels[b:, :b])


@pytest.mark.parametrize("n", [4, 16])
def test_similarity_preservation_matches_reference(n):
    torch.manual_seed(n)
    student, teacher = torch.randn(n, 32), torch.randn(n, 32)
    loss = SimilarityPreservationLoss()(student, teacher)
    assert torch.allclose(loss, reference_similarity_preservation(student, teacher), rtol=1e-5)


def test_similarity_preservation_is_zero_for_identical_similarities():
    torch.manual_seed(0)
    features = torch.randn(8, 16)
    # scaling does not change cosine similarities
    loss = SimilarityPreservationLoss()(features, 3.0 * features)
    assert loss.item() < 1e-2


def test_similarity_preservation_gradient_only_reaches_student():
    torch.manual_seed(0)
    student = torch.randn(8, 16, requires_grad=True)
    teacher = torch.randn(8, 16)
    SimilarityPreservationLoss()(student, teacher).backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_mmd_matches_reference():
    torch.manual_seed(0)
    source, target = torch.randn(16, 32), torch.randn(16, 32) + 0.5
    assert torch.allclose(MMDLoss()(source, target), reference_mmd(source, target), rtol=1e-5)


def test_mmd_separates_distributions():
    torch.manual_seed(0)
    a, b = torch.randn(32, 8), torch.randn(32, 8)
    mmd = MMDLoss()
    assert mmd(a, b + 3.0) > 10 * mmd(a, b).abs()


def test_baseline_losses_are_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    logits = torch.randn(8, 10)
    maps = torch.randn(8, 6, 4, 2)
    assert LwFLoss()(logits, logits).abs().item() < 1e-6
    assert SPLoss()(maps, maps).item() == 0
    assert ATLoss()(maps, maps).item() == 0
    assert SPLoss()(maps, torch.randn(8, 6, 4, 2)).item() > 0


def test_build_kd_loss():
    assert isinstance(build_kd_loss("s2p"), SimilarityPreservationLoss)
    with pytest.raises(KeyError):
        build_kd_loss("unknown")
