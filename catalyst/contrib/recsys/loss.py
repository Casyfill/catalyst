"""
"""
from typing import Optional

import numpy as np
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """ Implementation of 
    `BPRLoss, based on Bayesian Personalized Ranking`_.    

    .. _BPRLoss: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/pdf/1205.2618.pdf

    Args:
        gamma(float): Small value to avoid division by zero

    Example:

    .. code-block:: python
        import torch
        from torch.contrib import recsys
        
        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)
        
        output = recsys.BPRLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self, gamma: float = 1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        loss = -torch.log(self.gamma + torch.sigmoid(positive_score - negative_score)).mean()
        return loss


class WARP(Function):
    """
    autograd function of WARP loss
    """

    @staticmethod
    def forward(
        ctx: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        max_num_trials: Optional[int] = None,
    ):

        batch_size = target.size()[0]
        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = torch.zeros(input.size())
        negative_indices = torch.zeros(input.size())
        L = torch.zeros(input.size()[0])

        all_labels_idx = torch.arange(target.size()[1])

        Y = float(target.size()[1])
        J = torch.nonzero(target)

        for i in range(batch_size):

            msk = torch.ones(target.size()[1], dtype=bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while (sample_score_margin < 0) and (num_trials < max_num_trials):  # type: ignore

                # randomly sample a negative label, example from here: https://github.com/pytorch/pytorch/issues/16897
                neg_idx = neg_labels_idx[torch.randint(0, neg_labels_idx.size(0), (1,))]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + input[i, neg_idx] - input[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(np.floor((Y - 1) / (num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1  # type: ignore

        loss = L * (
            1
            - torch.sum(positive_indices * input, dim=1)
            + torch.sum(negative_indices * input, dim=1)
        )

        ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)

        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(nn.Module):
    """ Implementation of 
    WARP (WEIGHTED APPROXIMATE RANK PAIRWISE LOSS)
    """

    def __init__(self, max_num_trials: Optional[int] = None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return WARP.apply(input_, target, self.max_num_trials)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:

        loss = torch.clamp(negative_score - positive_score + 1.0, 0.0)
        return loss.mean()


class LogisticLoss(nn.Module):
    """Logistic Loss"""

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_ = torch.clamp(input_, 0, 1)
        return F.binary_cross_entropy_with_logits(input_, target, size_average=True)


class PoissonLoss:
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (input_ - target * torch.log(input_)).mean()
