
import os
import pickle
import numpy as np
import pandas as pd
from shutil import copyfile

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from .scoring import get_tpr_at_fpr

from typing import Union


def training_tabular(
        model: Union[LogisticRegression, RandomForestClassifier, XGBClassifier], 
        name: str, 
        X_train_encoded: np.ndarray,
        X_test_encoded: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray, 
        logs_folder: str,
        model_file: str = None
) -> Union[LogisticRegression, RandomForestClassifier, XGBClassifier]:
    print(f"[*] Training {name} model...")
    model.fit(X_train_encoded, y_train)

    # save trained model to LOGS_FOLDER/name
    os.makedirs(f"{logs_folder}/{name}", exist_ok=True)
    if isinstance(model, XGBClassifier):
        model.save_model(f"{logs_folder}/{name}/model.xgboost")
    else:
        with open(f"{logs_folder}/{name}/model.pkl", "wb") as f:
            pickle.dump(model, f)
    if model_file is not None:
        model_saved = [x for x in os.listdir(f"{logs_folder}/{name}") if x.startswith("model")][0]
        model_saved = os.path.join(f"{logs_folder}/{name}", model_saved)
        copyfile(model_saved, model_file)
    
    # calculate train metrics
    y_train_preds = model.predict_proba(X_train_encoded)[:,1]
    tpr = get_tpr_at_fpr(y_train, y_train_preds, fprNeeded=1e-4, logits=False)
    f1 = f1_score(y_train, y_train_preds.round())
    acc = accuracy_score(y_train, y_train_preds.round())
    auc = roc_auc_score(y_train, y_train_preds)
    print(f"[!] {name} model scores: train_tpr={tpr:.4f}, train_f1={f1:.4f}, train_acc={acc:.4f}, train_auc={auc:.4f}")
    metrics_csv = pd.DataFrame({"train_tpr": [tpr], "train_f1": [f1], "train_acc": [acc], "train_auc": [auc]})

    # calculate test metrics
    y_test_preds = model.predict_proba(X_test_encoded)[:,1]
    tpr = get_tpr_at_fpr(y_test, y_test_preds, fprNeeded=1e-4, logits=False)
    tpr_1e5 = get_tpr_at_fpr(y_test, y_test_preds, fprNeeded=1e-5, logits=False)
    tpr_1e6 = get_tpr_at_fpr(y_test, y_test_preds, fprNeeded=1e-6, logits=False)
    tpr_1e7 = get_tpr_at_fpr(y_test, y_test_preds, fprNeeded=1e-7, logits=False)
    f1 = f1_score(y_test, y_test_preds.round())
    acc = accuracy_score(y_test, y_test_preds.round())
    auc = roc_auc_score(y_test, y_test_preds)
    print(f"[!] {name} model scores: val_tpr={tpr:.4f}, val_f1={f1:.4f}, val_acc={acc:.4f}, val_auc={auc:.4f}")
    metrics_csv["val_tpr_1e4"] = tpr
    metrics_csv["val_tpr_1e5"] = tpr_1e5
    metrics_csv["val_tpr_1e6"] = tpr_1e6
    metrics_csv["val_tpr_1e7"] = tpr_1e7
    metrics_csv["val_f1"] = f1
    metrics_csv["val_acc"] = acc
    metrics_csv["val_auc"] = auc

    with open(f"{logs_folder}/{name}/metrics.csv", "w") as f:
        metrics_csv.to_csv(f, index=False)
    
    return model
