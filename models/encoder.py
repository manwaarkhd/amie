import torch.utils.checkpoint as cp
from collections import OrderedDict
from typing import Union, Optional
from torchvision import models
import torch.nn as nn
import torch
import os

class Encoder(nn.Module):
    
    def __init__(
        self,
        name: str, 
        weights: Optional[str] = None, 
        use_checkpoint: bool=False,
        device: Union[str, torch.device]="auto"
    ):
        super(Encoder, self).__init__()
        # device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # build encoder network
        name = name.lower()
        if name == "resnet18":
            model = models.resnet18(weights=None)
            self.encoder = torch.nn.Sequential(OrderedDict(list(model.named_children())[:-2]))
            self.name = "ResNet18"
            featmap_depth = 512
        elif name == "resnet34":
            model = models.resnet34(weights=None)
            self.encoder = torch.nn.Sequential(OrderedDict(list(model.named_children())[:-2]))
            self.name = "ResNet34"
            featmap_depth = 512
        elif name == "resnet50":
            model = models.resnet50(weights=None)
            self.encoder = torch.nn.Sequential(OrderedDict(list(model.named_children())[:-2]))
            self.name = "ResNet50"
            featmap_depth = 2048
        else:
            raise ValueError(f"unsupported encoder: `{name}`. Available options: 'vgg16', 'resnet34', 'resnet50'")
        
        # load weights
        if weights is not None:
            if isinstance(weights, str):
                if weights.lower() == "imagenet":
                    model_with_weights = getattr(models, name)(weights=getattr(models, f"{self.name}_Weights").DEFAULT)
                    self.encoder.load_state_dict(
                        torch.nn.Sequential(OrderedDict(list(model_with_weights.named_children())[:-2])).state_dict(),
                        strict=False
                    )
                elif os.path.isfile(weights):
                    checkpoint = torch.load(weights, weights_only=False, map_location=self.device)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.encoder.load_state_dict(state_dict, strict=False)
                else:
                    raise ValueError(f"invalid weights argument: '{weights}' (must be 'imagenet', path to checkpoint, or None)")
            else:
                raise TypeError(f"expected `weights` to be str or None, got {type(weights)}")

        # feature extraction head
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        
        # replace inplace ReLUs to make checkpointing safe
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.disable_inplace(self.encoder)
        self.feature_dim = 1 * 1 * featmap_depth
        self.to(self.device)
    
    def disable_inplace(self, module: nn.Module):
        """ Recursively set inplace=False for all layers that have an 'inplace' attribute. """
        for child_name, child in module.named_children():
            if hasattr(child, "inplace") and getattr(child, "inplace") is True:
                setattr(child, "inplace", False)
            else:
                self.disable_inplace(child)

    def forward(self, x: torch.tensor):
        # ensure input `x` is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # forward through encoder with optional checkpointing
        if self.use_checkpoint and self.training:
            for module in self.encoder:
                x = cp.checkpoint(module, x, use_reentrant=False)
        else:
            x = self.encoder(x)
        
        # extract features
        features = self.feature_head(x)
        return features
    
    def freeze(self, block_name: Optional[str]=None):
        """ Freeze encoder parameters up to (and including) the specified block name. """
        children = list(self.encoder.named_children())
        block_names = [name for name, _ in children]

        # freeze entire encoder
        if block_name is None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            return
        elif block_name in block_names:
            # freeze all blocks up to and including the specified one
            block_index = block_names.index(block_name)
            for index, (name, module) in enumerate(children):
                requires_grad = index > block_index
                for param in module.parameters():
                    param.requires_grad = requires_grad
        else:
            raise ValueError(f"invalid block_name '{block_name}'. Available blocks: {block_names}")

    def unfreeze(self):
        """ Unfreeze encoder parameters for fine-tuning. """
        for param in self.encoder.parameters():
            param.requires_grad = True