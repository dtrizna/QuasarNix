import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve

import torch
from torch.utils.data.dataloader import DataLoader
import lightning as L

from typing import Union
from .models import PyTorchLightningModel
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix


def sigmoid(x):
    x = np.clip(x, -500, 500) # to prevent overflows
    return 1 / (1 + np.exp(-x))


def predict(loader, trainer, lightning_model, decision_threshold=0.5, dump_logits=False):
    """Get scores out of a loader."""
    y_pred_logits = trainer.predict(model=lightning_model, dataloaders=loader)
    y_pred_logits = torch.cat(y_pred_logits, dim=0)
    y_pred_probs = torch.sigmoid(y_pred_logits).numpy()
    y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred_probs])
    if dump_logits:
        assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
        pickle.dump(y_pred_logits, open(dump_logits, "wb"))
    return y_pred, y_pred_logits.numpy()


def get_tpr_at_fpr(true_labels, predicted_labels, fprNeeded=1e-4, logits=True, include_thresholds=False):
    if logits:
        if isinstance(predicted_labels, torch.Tensor):
            predicted_labels = torch.sigmoid(predicted_labels).cpu().detach().numpy()
        else:
            predicted_labels = sigmoid(predicted_labels)
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    if all(np.isnan(fpr)):
        return (0, 0) if include_thresholds else 0
    else:
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return (tpr_at_fpr, threshold_at_fpr) if include_thresholds else tpr_at_fpr


def collect_scores(
        X_test: Union[DataLoader, np.ndarray, csr_matrix],
        y_test: np.ndarray,
        model: Union[PyTorchLightningModel, XGBClassifier],
        trainer: L.Trainer = None,
        scores: dict = None,
        score_suffix: str = "",
        run_name: str = "",
        verbose: bool = True) -> dict:
    
    assert isinstance(model, (PyTorchLightningModel, XGBClassifier)), "Please provide a PyTorchLightningModel or XGBClassifier"
    assert isinstance(X_test, (DataLoader, np.ndarray, csr_matrix)), "Please provide a DataLoader, np.ndarray or csr_matrix"
    assert isinstance(X_test, DataLoader) == isinstance(model, PyTorchLightningModel), "Please provide a DataLoader if you are using a PyTorchLightningModel"
    assert isinstance(X_test, (np.ndarray, csr_matrix)) == isinstance(model, XGBClassifier), "Please provide a np.ndarray if you are using a XGBClassifier"
    assert isinstance(model, PyTorchLightningModel) == (trainer is not None), "Please provide a trainer if you are using a PyTorchLightningModel"
    
    if isinstance(model, PyTorchLightningModel):
        y_test_pred, y_test_pred_logits = predict(X_test, trainer, model, decision_threshold=0.5)
    elif isinstance(model, XGBClassifier):
        y_test_pred = model.predict(X_test)
        y_test_pred_logits = model.predict_proba(X_test)[:,1]

    tpr_test_poisoned = get_tpr_at_fpr(y_test, y_test_pred_logits, logits=True)
    f1_test_poisoned = f1_score(y_test, y_test_pred)
    acc_test_poisoned = accuracy_score(y_test, y_test_pred)
    misclassified_test_poisoned_idxs = np.where(y_test != y_test_pred)[0]
    misclassified_test_poisoned = len(misclassified_test_poisoned_idxs)
    misclassified_test_poisoned_malicious = len(np.where(y_test[misclassified_test_poisoned_idxs] == 1)[0])
    misclassified_test_poisoned_benign = len(np.where(y_test[misclassified_test_poisoned_idxs] == 0)[0])
    # check if single class, if not -- then calculate auc
    auc_test_poisoned = roc_auc_score(y_test, y_test_pred) if len(np.unique(y_test)) > 1 else None
    local_scores = {
            f"tpr{score_suffix}": tpr_test_poisoned,
            f"f1{score_suffix}": f1_test_poisoned,
            f"acc{score_suffix}": acc_test_poisoned,
            f"auc{score_suffix}": auc_test_poisoned,
            f"misclassified{score_suffix}": misclassified_test_poisoned,
            f"misclassified_malicious{score_suffix}": misclassified_test_poisoned_malicious,
            f"misclassified_benign{score_suffix}": misclassified_test_poisoned_benign,
        }
    if scores is None:
        scores = local_scores
    else:
        scores.update(local_scores)
    
    if verbose:
        print(f"[!] Scores for '{run_name}' model ({score_suffix.replace('_','')}): f1={f1_test_poisoned:.4f}, acc={acc_test_poisoned:.4f}, tpr={tpr_test_poisoned:.4f}")
        print(f"               misclassified={misclassified_test_poisoned}: malicious={misclassified_test_poisoned_malicious} | benign={misclassified_test_poisoned_benign}")

    return scores
