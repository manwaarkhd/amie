from typing import Union
import torch.nn as nn
import torch

class MaxPooling(nn.Module):
    def __init__(self, device: Union[str, torch.device]="auto"):
        super(MaxPooling, self).__init__()
        # device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.to(self.device)
    
    def forward(self, x: torch.tensor):
        # ensure input `x` is on correct device
        if x.device != self.device:
            x = x.to(self.device)

        Z, _ = torch.max(x, dim=0, keepdim=True)
        return Z

class MeanPooling(nn.Module):
    def __init__(self, device: Union[str, torch.device]="auto"):
        super(MeanPooling, self).__init__()
        # device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.to(self.device)
    
    def forward(self, x: torch.tensor):
        # ensure input `x` is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        Z = torch.mean(x, dim=0, keepdim=True)
        return Z

class GatedAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int=256, num_classes: int=1, device: Union[str, torch.device]="auto"):
        super(GatedAttentionPooling, self).__init__()
        # device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # gated attention module
        # `tanh` attention branch
        self.attention_tanh = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )

        # `sigmoid` attention branch
        self.attention_sigm = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )

        # attention projection
        self.attention = nn.Linear(attention_dim, num_classes)
        self.to(self.device)

    def forward(self, x: torch.tensor, return_scores: bool=False):
        # ensure input `x` is on correct device
        if x.device != self.device:
            x = x.to(self.device)

        # gated attention
        A1 = self.attention_sigm(x)
        A2 = self.attention_tanh(x)
        A = torch.mul(A1, A2)

        # attention pooling
        A = self.attention(A)
        A = torch.transpose(A, 1, 0)
        scores = torch.softmax(A, dim=1)
        Z = torch.matmul(scores, x)
        
        if return_scores:
            return Z, scores
        else:
            return Z