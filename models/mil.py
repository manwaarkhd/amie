import torch.nn as nn
import torch

from .pooling import MaxPooling, MeanPooling, GatedAttentionPooling

class MILNet(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        feature_dim: int=None, 
        pooling: str="attention", 
        attention_dim: int=256, 
        num_classes: int=1, 
        device: str="auto"
    ):
        super(MILNet, self).__init__()
        # device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # feature extractor
        self.encoder = encoder
        if self.encoder:
            feature_dim = self.encoder.feature_dim
        else:
            feature_dim = feature_dim
        
        # pooling
        pooling = pooling.lower()
        if pooling == "max":
            self.pooling = MaxPooling()
        elif pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "attention":
            self.pooling = GatedAttentionPooling(feature_dim, attention_dim, num_classes, device=self.device)
        else:
            raise ValueError(f"unsupported pooling type: `{pooling}`. Available options: 'max', 'mean', 'attention'")

        # classifier
        self.classifier = nn.Linear(feature_dim * num_classes, num_classes, device=self.device)
    
    def forward(self, input: torch.tensor, return_scores: bool=False, chunk_size: int=None):
        # feature extraction
        if self.encoder:
            if chunk_size:
                # store each feature separately
                num_instances = input.shape[0]
                features = []
                for index in range(0, num_instances, chunk_size):
                    x = input[index:index+chunk_size]
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        embeddings = self.encoder(x)
                    features.append(embeddings)
                features = torch.cat(features, dim=0)
            else:
                features = self.encoder(input)
        else:
            features = input

        # mil pooling
        if return_scores:
            Z, scores = self.pooling(features, return_scores=True)
        else:
            Z = self.pooling(features)
        
        # classification
        logits = self.classifier(Z)
        
        if return_scores:
            return logits, scores
        else:
            return logits