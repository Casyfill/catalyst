'''
'''
import torch
import torch.nn as nn


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

    def __init__(self, gamma:float=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score:torch.Tensor, neg_score:torch.Tensor)->torch.Tensor:
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss