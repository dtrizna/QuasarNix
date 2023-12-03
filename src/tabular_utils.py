
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
    with open(f"{logs_folder}/{name}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    if model_file is not None:
        copyfile(f"{logs_folder}/{name}/model.pkl", model_file)
    
    y_test_preds = model.predict_proba(X_test_encoded)[:,1]

    tpr = get_tpr_at_fpr(y_test, y_test_preds, fprNeeded=1e-4, logits=False)
    f1 = f1_score(y_test, y_test_preds.round())
    acc = accuracy_score(y_test, y_test_preds.round())
    auc = roc_auc_score(y_test, y_test_preds)
    
    print(f"[!] {name} model scores: tpr={tpr:.4f}, f1={f1:.4f}, acc={acc:.4f}, auc={auc:.4f}")
    
    metrics_csv = pd.DataFrame({"tpr": [tpr], "f1": [f1], "acc": [acc], "auc": [auc]})
    with open(f"{logs_folder}/{name}/metrics.csv", "w") as f:
        metrics_csv.to_csv(f, index=False)
    
    return model
