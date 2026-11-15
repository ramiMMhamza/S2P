import copy

import torch
from openunreid.models.layers.domain_specific_bn import DSBN
from torch import nn

from s2p.dsbn import SupportDSBN, disable_support_branch, enable_support_branch


def make_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 1), DSBN(4, 2, target_bn_idx=0))
    return model


def test_support_chunk_goes_through_the_source_bn():
    model = make_model()
    reference = copy.deepcopy(model)
    assert enable_support_branch(model) == 1
    assert isinstance(model[1], SupportDSBN)

    model.train()
    reference.train()
    x = torch.randn(12, 3, 5, 5)  # [target | source | support], 4 each
    out = model(x)

    h = reference[0](x)
    target = reference[1].dsbn[0](h[:4])
    source_and_support = reference[1].dsbn[1](h[4:])
    assert torch.allclose(out, torch.cat((target, source_and_support)), atol=1e-6)
    # the running statistics of each domain saw the same data
    for bn, bn_ref in zip(model[1].dsbn, reference[1].dsbn):
        assert torch.allclose(bn.running_mean, bn_ref.running_mean, atol=1e-6)


def test_state_dict_and_eval_are_unchanged():
    model = make_model()
    reference = copy.deepcopy(model)
    enable_support_branch(model)
    assert list(model.state_dict().keys()) == list(reference.state_dict().keys())

    model.eval()
    reference.eval()
    x = torch.randn(5, 3, 5, 5)
    assert torch.equal(model(x), reference(x))

    assert disable_support_branch(model) == 1
    assert type(model[1]) is DSBN


def test_batch_must_be_three_equal_chunks():
    model = make_model()
    enable_support_branch(model)
    model.train()
    try:
        model(torch.randn(8, 3, 5, 5))
    except AssertionError:
        return
    raise AssertionError("a batch of 8 cannot be split in 3 chunks")
