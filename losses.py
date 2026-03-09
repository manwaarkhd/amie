import torch.nn.functional as F
import torch.nn as nn
import torch

class BCELoss(nn.Module):
    """ Binary Cross-Entropy Loss with optional class weighting. """
    default_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    
    def __init__(self, weights: list=None, reduction: str="mean"):
        super(BCELoss, self).__init__()
        if weights is None:
            weights = self.default_weights.clone()
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.reduction = reduction
    
    def forward(self, logits, targets):
        w0, w1 = self.weights.to(logits.device)

        loss = (w1 * targets * F.logsigmoid(logits)) + (w0 * (1. - targets) * F.logsigmoid(-logits))
        loss = (-1.) * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __str__(self):
        default_on_device = self.default_weights.to(self.weights.device)
        if torch.allclose(self.weights, default_on_device, atol=1e-6):
            return "BCELoss"
        else:
            return "WeightedBCELoss"