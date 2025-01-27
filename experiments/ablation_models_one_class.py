import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc

# importing one class models -- oc svm and isolation forest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM

# tokenizers
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import load_data
from nltk.tokenize import wordpunct_tokenize


LIMIT = None
SEED = 33
ROOT = os.path.dirname(os.path.abspath('__file__'))
VOCAB_SIZE = 4096
MAX_LEN = 128
BASELINE = True # whether to train on baseline or malicious data

LOGS_FOLDER = "logs_models_one_class_v2"
os.makedirs(LOGS_FOLDER, exist_ok=True)

TOKENIZER = wordpunct_tokenize
tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    # return X_train_cmds, y_train, X_test_cmds, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_tpr_at_fpr(predicted_probs, true_labels, fprNeeded=1e-5):
    # if isinstance(predicted_logits, torch.Tensor):
    #     predicted_probs = torch.sigmoid(predicted_logits).cpu().detach().numpy()
    # else:
    #     predicted_probs = sigmoid(predicted_logits)
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    if all(np.isnan(fpr)):
        return np.nan#, np.nan
    else:
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return tpr_at_fpr#, threshold_at_fpr

X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(SEED, LIMIT)
if LIMIT is not None:
    X_train_cmds = X_train_cmds[:LIMIT]
    y_train = y_train[:LIMIT]
    X_test_cmds = X_test_cmds[:LIMIT]
    y_test = y_test[:LIMIT]
    
print('X_train_cmds: ', len(X_train_cmds))
print('X_test_cmds: ', len(X_test_cmds))

train_mask = y_train == 0 if BASELINE else y_train == 1
X_train_cmds_one_class = np.array(X_train_cmds)[train_mask]
y_train_one_class = np.array(y_train)[train_mask]

def fit_one_hot(X):
    oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
    print("[*] Fitting One-Hot encoder...")
    now = time.time()
    oh.fit(X)
    print(f"[!] Finished fitting One-Hot encoder in {time.time() - now:.2f} seconds")
    return oh

# building one-hot encoder
oh_vectorizer_file = os.path.join(LOGS_FOLDER, f"onehot_vectorizer_{VOCAB_SIZE}_lim_{LIMIT}.pkl")
if os.path.exists(oh_vectorizer_file):
    print("[*] Loading One-Hot vectorizer...")
    oh = pickle.load(open(oh_vectorizer_file, "rb"))
else:
    oh = fit_one_hot(X_train_cmds_one_class)
    with open(oh_vectorizer_file, "wb") as f:
        pickle.dump(oh, f)


print("[*] Transforming train and test sets...")

now = time.time()
X_train_onehot_one_class = oh.transform(X_train_cmds_one_class)
print(f"[!] Finished transforming train set in {time.time() - now:.2f} seconds")

now = time.time()
X_test_onehot = oh.transform(X_test_cmds)
print(f"[!] Finished transforming test set in {time.time() - now:.2f} seconds")

print(f"[!] Shapes: X_train_onehot_one_class: {X_train_onehot_one_class.shape}, X_test_onehot: {X_test_onehot.shape}")


def train_and_predict(model, X_train, X_test, y_test, name=""):
    # training
    print("[*] Training model...")
    now = time.time()
    model.fit(X_train)
    print(f"[*] Training model took {time.time() - now:.2f} seconds.")
    
    # dump model
    model_file = os.path.join(LOGS_FOLDER, f"model_{name}_{VOCAB_SIZE}_lim_{LIMIT}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # predicting
    model_preds = model.predict(X_test)
    model_preds = np.array([1 if x == -1 else 0 for x in model_preds])

    # calculating metrics
    model_tpr = get_tpr_at_fpr(model_preds, y_test, fprNeeded=1e-5)
    model_f1 = f1_score(y_test, model_preds)
    model_acc = accuracy_score(y_test, model_preds)
    model_auc = roc_auc_score(y_test, model_preds)

    print('TPR at FPR=1e-5: {:.4f}%'.format(model_tpr*100))
    print('F1 score: {:.4f}%'.format(model_f1*100))
    print('Accuracy: {:.4f}%'.format(model_acc*100))
    print('AUC: {:.4f}%'.format(model_auc*100))

    return model_tpr, model_f1, model_acc, model_auc

# ### OneClassSVM is really slow model:
#   - takes 30 sec (optimal speed, nu='0.1') / 2 mins (best scores, nu=0.5) per 50k samples 
#   - takes ~2h 30 min (optimal speed, nu='0.1') / ?? h (best scores, nu=0.5) per full dataset
# oc svm
# for nu in [0.1, 0.3, 0.5, 0.7, 0.9]:
#     print(f"\n[!] Nu: {nu}")
#     oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
#     _ = train_and_predict(oc_svm, X_train_onehot_malicious, X_test_onehot, y_test, name=f"oc_svm_nu_{nu}")

# ### SGD implementation of OneClassSVM is much faster
# oc svm sgd version
for nu in [0.5, 0.9]:
    for tol in [1e-3, 1e-4, 1e-5]:
        print(f"\n[!] Nu: {nu} | tol: {tol}")
        oc_svm_sgd = SGDOneClassSVM(nu=nu, random_state=SEED, learning_rate='constant', eta0=0.1, tol=tol, max_iter=1000)
        _ = train_and_predict(oc_svm_sgd, X_train_onehot_one_class, X_test_onehot, y_test, name=f"oc_svm_sgd_nu_{nu}_tol_{tol}")

### Isolation Forest
for contamination in ["auto", 0.1, 0.3, 0.5]:
    for max_samples in ['auto', 0.01, 0.05, 0.1, 0.3]:
        print(f"\n[!] Contamination: {contamination} | Max samples: {max_samples}")
        isolation_forest = IsolationForest(n_estimators=100, max_samples=0.1, contamination=contamination, 
                                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=SEED)
        _ = train_and_predict(isolation_forest, X_train_onehot_one_class, X_test_onehot, y_test, name=f"isolation_forest_cont_{contamination}_max_samples_{max_samples}")
