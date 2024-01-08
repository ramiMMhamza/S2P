# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Obtained from: https://github.com/open-mmlab/OpenUnReID
# Modified to support OUDA

import torch
import torch.nn.functional as F
from torch import autograd, nn

from ...utils.dist_utils import all_gather_tensor

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, num_memory_target, num_memory_source, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory
        self.num_memory_target = num_memory_target
        self.num_memory_source = num_memory_source

        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("features_target", torch.zeros(num_memory_target, num_features))
        self.register_buffer("features_source", torch.zeros(num_memory_source, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("labels_target", torch.zeros(num_memory_target).long())
        self.register_buffer("labels_source", torch.zeros(num_memory_source).long())

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels, labels_target, labels_source):
        self.labels.data.copy_(labels.long().to(self.labels.device))
        self.labels_target.data.copy_(labels_target.long().to(self.labels_target.device))
        self.labels_source.data.copy_(labels_source.long().to(self.labels_source.device))

    def forward(self, results, indexes):
        inputs = results["feat"]
        inputs = F.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        
        B = inputs.size(0)
        inputs /= self.temp
        def masked_softmax(vec, mask, coef,  dim=1, epsilon=1e-6,):
            exps = torch.exp(vec)
            if coef is not None :
                masked_exps = (exps * mask.float().clone()).clone()
                masked_exps_weighted = masked_exps*coef
                masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
                
                return masked_exps_weighted/masked_sums
            # print("exps after weighted{}".format(exps_))
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums #masked_exps / masked_sums

        len_ = int(len(results["feat"])/2) 
        nt = self.labels_target.max()+1 #self.num_memory_target
        indexes_target, indexes_source = indexes[:len_], indexes[len_:]
        targets = self.labels[indexes].clone()
        targets_target = self.labels_target[indexes_target].clone()
        targets_source = self.labels_source[indexes_source-indexes_source.min()].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
        
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        ################### For weighted Loss ###################
        a, b = sim.size()
        coef = torch.zeros((a,b)).cuda()
        coef[:,:int(b/2)]= 3/4 #2/3 #Ct
        coef[:,int(b/2):]= 1/4 #1/3 #Cs
        ########################################################
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous(), coef = None) # coef.t().contiguous() , coef_target = test_target, coef_source = test_source)
        # masked_sim = masked_sim_target+masked_sim_source

        # return F.nll_loss(torch.log(masked_sim_target + 1e-6), targets_target) + F.nll_loss(torch.log(masked_sim_source + 1e-6), targets_source) #
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)