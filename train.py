from posixpath import curdir

from augmentations import RandomPatchMask, RandomFourierTransform, RandomColorDistortion
from utils import AverageMeter, clear_gpu_cache 
from metrics import Accuracy, compute_metrics
from datasets import TCGAKFold, TCGABags
from models import Encoder, MILNet
from losses import BCELoss
from paths import Paths
from config import cfg


from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any
import pandas as pd
import numpy as np
import argparse
import os

# hpcdir = r"\\essces\project\ag-jafra\Muhammad" # r"/project/ag-jafra/Muhammad"

class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.CrossEntropyLoss,
        ckpt_dir: str=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ckpt_dir = ckpt_dir

    def train_one_epoch(self, data: DataLoader):
        num_batches = len(data)
        self.model.train()

        loss_meter = AverageMeter(name="loss")
        acc_meter = AverageMeter(name="accuracy")
        metric = Accuracy(threshold=0.5)

        train_targets, train_logits = [], []
        for index, (patches, true_label) in enumerate(data):
            patches = torch.squeeze(patches, dim=0)
            targets = torch.unsqueeze(true_label, dim=0).to(dtype=torch.float32, device=self.model.device)

            self.optimizer.zero_grad()
            logits = self.model(patches, chunk_size=256)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            accuracy = metric(logits, targets)
            loss_meter.update(loss.item())
            acc_meter.update(accuracy)

            print(f"\r{index+1}/{num_batches} - {loss_meter} - {acc_meter}", end="")
            train_targets.append(targets.item())
            train_logits.append(logits.item())
            
            # cleanup
            del patches, targets, logits, loss
            clear_gpu_cache()
        
        return loss_meter.average, train_targets, train_logits
    
    def validate(self, data: DataLoader):
        loss_meter = AverageMeter(name="loss")
        self.model.eval()
    
        valid_targets, valid_logits = [], []
        with torch.no_grad():
            for index, (patches, true_label) in enumerate(data):
                patches = torch.squeeze(patches, dim=0)
                targets = torch.unsqueeze(true_label, dim=0).to(dtype=torch.float32, device=self.model.device)
                               
                logits = self.model(patches, chunk_size=256)
                loss = self.criterion(logits, targets)
                loss_meter.update(loss.item())
                
                valid_targets.append(targets.item())
                valid_logits.append(logits.item())
        
        return loss_meter.average, valid_targets, valid_logits
    
    def fit(self, train_data: DataLoader, valid_data: DataLoader, epochs: int=10, **kwargs):
        for epoch in range(epochs):
            num_digits = len(str(epochs))
            fmt = "{:" + str(num_digits) + "d}"
            print("Epoch: " + fmt.format(epoch+1) + "/" + fmt.format(epochs) + "")

            train_loss, train_targets, train_logits = self.train_one_epoch(train_data)
            metrics = self.compute_metrics(train_targets, train_logits)
            print(f"\rtrain_loss: {train_loss: .5f} - "
                f"accuracy: {metrics['accuracy']: .3f} - "
                f"precision: {metrics['precision']: .3f} - "
                f"recall: {metrics['sensitivity']: .3f} - "
                f"f1 score: {metrics['f1']: .3f} - "
                f"auc: {metrics['roc_auc']: .3f}", 
                end="\n"
            )

            valid_loss, valid_targets, valid_logits = self.validate(valid_data)
            metrics = compute_metrics(valid_targets, valid_logits)
            print(f"\rvalid_loss: {valid_loss: .5f} - "
                f"accuracy: {metrics['accuracy']: .3f} - "
                f"precision: {metrics['precision']: .3f} - "
                f"recall: {metrics['sensitivity']: .3f} - "
                f"f1 score: {metrics['f1']: .3f} - "
                f"auc: {metrics['roc_auc']: .3f}", 
                end="\n"
            )

            chkpt_score = sum(1.0 - np.array(list(metrics.values())))
            chkpt_name = f"{fold}.{epoch+1}-{chkpt_score:0.5f}-{valid_loss:0.5f}-{metrics['f1']:0.5f}-" \
                f"{kwargs['dataset']}.{kwargs['project']}-" \
                f"level.{kwargs['level']}-" \
                f"{kwargs['learning_rate']}-" \
                f"{self.model.encoder.name}-" \
                f"{self.criterion}.pth"
            
            checkpoint = {
                "data": {
                    "slide_name": [sample["slide_name"] for sample in valid_data.dataset.samples],
                    "targets": valid_targets,
                    "logits": valid_logits,
                },
                "attr": {
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "sensitivity": metrics['sensitivity'],
                    "specificity": metrics['specificity'], 
                    "f1": metrics['f1'],
                    "roc_auc": metrics['roc_auc'],
                    "pr_auc": metrics['pr_auc'],
                    "mcc": metrics['mcc']
                }
            }
            self.save_checkpoint(path=os.path.join(self.ckpt_dir, chkpt_name), record=checkpoint)

    def save_checkpoint(self, path: str, record: Dict[str, Any]=None):
        """ Saves a model checkpoint with optional metadata. """
        checkpoint = {"model_state_dict": self.model.state_dict()}
        if record:
            for key, data in record.items():
                if isinstance(data, dict):
                    checkpoint[key] = data.copy()
                else:
                    checkpoint[key] = data
        torch.save(checkpoint, path)


parser = argparse.ArgumentParser(description="Distributed training job")
parser.add_argument("--dataset", type=str, default="TCGA", help="Dataset cohort name (e.g., TCGA, CPTAC, Camelyon16).")
parser.add_argument("--project", type=str, help="Cancer type or project identifier (e.g., BRCA, LUAD).")
parser.add_argument("--level", type=int, default=0, help="Magnification level for feature extraction (0=highest resolution, 3=lowest resolution, default: 0)")
parser.add_argument("--lr", type=float, help="Initial learning rate.")
parser.add_argument("--fold_num", type=int, default=None, help="Specific fold number to train (1-based). If not provided, trains all folds.")
args = parser.parse_args()

if __name__ == "__main__":
    # training configurations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_folds = 3
    seed = 36

    # random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if args.project == "all":
        projects = cfg.CANCER_TYPES
    elif args.project in cfg.CANCER_TYPES:
        projects = [args.project]
    else:
        raise ValueError(f"Invalid project: {args.project}. Expected one of {cfg.CANCER_TYPES} or 'TCGA'.")

    path = Paths.get_dataset_path(args.dataset)
    annotations = pd.read_csv(os.path.join(path, "annotations (v1.1).csv"))
    tcga_folds = TCGAKFold(path, annotations, projects, num_folds=3)

    # determine folds to train
    if args.fold_num is None:
        folds_to_train = range(1, num_folds + 1)
        print(f"Training all {num_folds} folds...")
    else:
        if not (1 <= args.fold_num <= num_folds):
            raise ValueError(f"Invalid fold_num: {args.fold_num}. It must be between 1 and {num_folds}.")
        folds_to_train = [args.fold_num]
        print(f"Training only fold {args.fold_num}...")

    for fold in folds_to_train:
        folds_data = tcga_folds[fold-1]
        
        tcga_train = TCGABags(folds_data["train"], level=args.level, 
            augmentations=[
                RandomPatchMask(strength=(0.35, 0.65), value=128, mode="constant"),
                RandomPatchMask(strength=(0.35, 0.65), mode="mean"),
                RandomPatchMask(strength=(0.35, 0.65), value=128, mode="hybrid"),
                RandomFourierTransform(radius_range=(48, 128), filter_type="low"),
                RandomColorDistortion(
                    strength=(0.35, 0.65), 
                    brightness=0.4,  # Conservative brightness for tissue visibility
                    contrast=0.7,    # Moderate contrast adjustment
                    saturation=1.0,  # Higher saturation variation (stain intensity)
                    hue=0.1
                )
            ]
        )
        train_data = DataLoader(tcga_train, batch_size=1, shuffle=True)

        tcga_test = TCGABags(folds_data["test"], level=args.level, augmentations=None)
        test_data = DataLoader(tcga_test, batch_size=1, shuffle=False)

        print(f"Fold {fold}: Train samples = {len(tcga_train)}, Test samples = {len(tcga_test)}")

        encoder = Encoder(name="resnet18", weights="imagenet", use_checkpoint=False, device=device)
        model = MILNet(
            encoder, 
            feature_dim=encoder.feature_dim, 
            pooling="attention", 
            attention_dim=256, 
            num_classes=1,
            device=device
        )

        print(f"Model initialized with encoder '{encoder.name}' and attention pooling. Feature dimension: {encoder.feature_dim}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        labels = [sample["label"] for sample in train_data.dataset.samples]
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        criterion = BCELoss(weights)

        print(f"Fold {fold}/{num_folds}")
        kwargs = {
            "dataset": args.dataset,
            "project": args.project,
            "level": args.level,
            "learning_rate": args.lr,
        }
        trainer = Trainer(model, optimizer, criterion, ckpt_dir=os.path.join(Paths.get_ckpt_path(), args.project, "folds", f"{fold}"))
        trainer.fit(train_data, test_data, epochs=args.epochs, **kwargs)
        


