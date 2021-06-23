# flake8: noqa
import pytest
import torch
# from catalyst.data import AllTripletsSampler

ratings = {
    0.6931: (torch.zeros(100), torch.zeros(100)),  # log(0.5 + loss.gamma)
    0.6931: (torch.tensor([1000,2000]), torch.tensor([1000,2000])),  # log(0.5 + loss.gamma)
    0.3133: (torch.ones(1000), torch.zeros(1000)),
    0.1269: (torch.ones(1000), -torch.ones(1000)),
}


@pytest.mark.parametrize("answer,args", ratings.items())
def test_BRPLoss(answer, args):
    """@TODO: Docs. Contribution is welcome."""
    from catalyst.contrib.recsys.loss import BPRLoss
    loss = BPRLoss()

    assert float(loss.forward(*args)) == pytest.approx(answer, 0.001)
    

@pytest.mark.parametrize("answer,args", ratings.items())
def test_WARPLoss(answer, args):
    from catalyst.contrib.recsys.loss import WARPLoss
    loss = WARPLoss()

    assert float(loss.forward(*args)) > 0

