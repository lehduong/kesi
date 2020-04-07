from torch import nn 
from torch.nn import functional as F

class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence loss between 2 tensor
    return the KL divergence between distributions
    :param temperature - float:
    input:
        inputs - torch.Tensor: the predictions of 1 model. The shape of this tensor should be batchsize x C x H x W
        targets - torch.Tensor: the target of
    """

    def __init__(self, temperature=1):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        p_s = F.log_softmax(inputs / self.temperature, dim=1)
        p_t = F.softmax(targets / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t) * (self.temperature ** 2)*targets.shape[1]
        return loss