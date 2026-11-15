"""Domain-specific batch norm for batches that carry a support set."""

import torch
from openunreid.models.layers.domain_specific_bn import DSBN

__all__ = ["SupportDSBN", "disable_support_branch", "enable_support_branch"]


class SupportDSBN(DSBN):
    """DSBN whose training batch is ``[domain_0 | ... | domain_k | support]``.

    The support set is made of *source* images, so it is normalised together
    with the last domain chunk (the source domain in S2P, where the training
    datasets are ordered ``[target, source]``). All ``k + 2`` chunks must have
    the same size. Evaluation is unchanged.
    """

    def _forward_train(self, x):
        bs = x.size(0)
        num_chunks = self.num_domains + 1
        assert bs % num_chunks == 0, (
            "the batch size should be a multiple of the number of BN groups + 1 "
            f"(one chunk per domain plus the support set), got {bs}"
        )
        chunks = list(torch.split(x, bs // num_chunks, 0))
        support = chunks.pop()
        chunks[-1] = torch.cat((chunks[-1], support), 0)
        out = [self.dsbn[idx](subx.contiguous()) for idx, subx in enumerate(chunks)]
        return torch.cat(out, 0)


def enable_support_branch(model):
    """Make every DSBN layer of ``model`` expect a trailing support chunk.

    Only the forward routine changes: the layers keep their parameters,
    buffers and state-dict keys, so checkpoints stay interchangeable.
    """
    count = 0
    for module in model.modules():
        if type(module) is DSBN:
            module.__class__ = SupportDSBN
            count += 1
    return count


def disable_support_branch(model):
    count = 0
    for module in model.modules():
        if type(module) is SupportDSBN:
            module.__class__ = DSBN
            count += 1
    return count
