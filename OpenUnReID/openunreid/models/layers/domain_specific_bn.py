# Obtained from: https://github.com/open-mmlab/OpenUnReID
# Modified to support OUDA and KD Loss of S2P

import torch
import torch.nn as nn


class DSBN(nn.Module):
    def __init__(
        self,
        num_features,
        num_domains,
        batchnorm_layer=nn.BatchNorm2d,
        eps=1e-5,
        momentum=0.1,
        target_bn_idx=-1,
        weight_requires_grad=True,
        bias_requires_grad=True,
        kd_flag=False,
    ):
        super(DSBN, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.target_bn_idx = target_bn_idx
        self.batchnorm_layer = batchnorm_layer
        self.kd_flag = kd_flag

        dsbn = [batchnorm_layer(num_features, eps=eps, momentum=momentum) 
                    for _ in range(num_domains)]
        for idx in range(num_domains):
            dsbn[idx].weight.requires_grad_(weight_requires_grad)
            dsbn[idx].bias.requires_grad_(bias_requires_grad)
        self.dsbn = nn.ModuleList(dsbn)

    def forward(self, x):
        if self.training and not self.kd_flag:
            return self._forward_train(x)
        elif self.training and self.kd_flag:
            return self._forward_train_kd(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        bs = x.size(0)
        assert bs % self.num_domains == 0, "the batch size should be times of BN groups"

        split = torch.split(x, int(bs // self.num_domains), 0)
        out = []
        for idx, subx in enumerate(split):
            out.append(self.dsbn[idx](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_test(self, x):
        # Default: the last BN is adopted for target domain
        return self.dsbn[self.target_bn_idx](x)
    def _forward_train_kd(self, x):
        bs = x.size(0)
        assert bs % (self.num_domains+1) == 0, "the batch size should be times of BN groups + 1"

        split = torch.split(x, int(bs // (self.num_domains+1)), 0) # target source support_set
        out = []
        out_ = torch.cat((split[1], split[2]), 0)
        split = [split[0],out_]
        for idx, subx in enumerate(split):
            out.append(self.dsbn[idx](subx.contiguous()))
        return torch.cat(out, 0)
