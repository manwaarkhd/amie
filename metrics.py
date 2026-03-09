from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, 
    auc, roc_auc_score, precision_recall_curve,
    confusion_matrix, matthews_corrcoef
)
import torch

class Accuracy:
    """ Classification accuracy metric. """

    def __init__(self, threshold: float=0.5):
        self.threshold = threshold

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        """ Computes accuracy from logits or probabilities using a given threshold. """
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # ensure same device, shape, and dtype
        targets = targets.to(device=logits.device).float()
        if preds.shape != targets.shape:
            targets = targets.view_as(preds)
        
        correct = (preds == targets).sum().item()
        total = targets.numel()
        return correct / total


def compute_metrics(self, targets, logits):
        """ Compute accuracy, precision, recall, F1, and AUC """
        probs = sigmoid(logits)
        preds = np.greater(probs, 0.5)

        # accuracy
        accuracy = accuracy_score(targets, preds)
        
        # precision
        precision = precision_score(targets, preds, average="binary", zero_division=0)

        # calculate confusion matrix and derived metrics
        TN, FP, FN, TP = confusion_matrix(targets, preds).ravel()

        # sensitivity or recall
        sensitivity = TP / (TP + FN)

        # specificity
        specificity = TN / (TN + FP)
        
        # f1 score
        f1 = f1_score(targets, preds, average="binary", zero_division=0)

        # area-under-curve
        roc_auc = roc_auc_score(targets, probs)

        # precision-recall-curve
        precision_curve, recall_curve, _ = precision_recall_curve(targets, probs)
        pr_auc = auc(recall_curve, precision_curve)

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(targets, preds)

        return {
            "accuracy": accuracy, 
            "precision": precision, 
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1, 
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mcc": mcc
        }