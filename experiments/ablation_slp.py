import os
import time
import json
import pickle
import numpy as np
from watermark import watermark
from nltk.tokenize import wordpunct_tokenize
from lightning.fabric.utilities.seed import seed_everything

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(ROOT)

from src.data_utils import load_data
from src.tabular_utils import training_tabular
from src.slp import ShellTokenizer, ShellEncoder
from src.augmentation import NixCommandAugmentation, read_template_file
from xgboost import XGBClassifier

SEED = 33
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128

# # PROD RUN CONFIG
DEVICE = "cpu"
EPOCHS = 10
LIT_SANITY_STEPS = 1
LIMIT = 100000
DATALOADER_WORKERS = 4
MODE = "augm" # "augmented" or "non_augmented"
LOGS_FOLDER = os.path.join(ROOT, "experiments", f"logs_slp_{MODE}_{int(time.time())}")


def train_slp(commands):
    tokenizer = ShellTokenizer(verbose=True)
    commands_tokenized_corpus, command_counter = tokenizer.tokenize(commands)
    return tokenizer, commands_tokenized_corpus, command_counter

if __name__ == "__main__":
    # ===========================================
    print(watermark(packages="torch,lightning,sklearn", python=True))
    print(f"[!] Script start time: {time.ctime()}")

    TOKENIZER = wordpunct_tokenize
    seed_everything(SEED)
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    # ============================================
    # LOADING DATA
    # ============================================
    print(f"[*] Loading data...")
    (
        X_train_cmds,
        y_train,
        X_test_cmds,
        y_test,
        X_train_malicious_cmd,
        X_train_baseline_cmd,
        X_test_malicious_cmd,
        X_test_baseline_cmd
    ) = load_data(root=ROOT, seed=SEED, limit=LIMIT)
    
    print(f"[!] Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # ============================================
    # DATA WITHOUT AUGMENTATION
    # ============================================
    if MODE == "augm":
        pass
    elif MODE == "non_augm":
        # generate non-augmented train set
        train_templates = read_template_file(os.path.join(ROOT, "data", "nix_shell", "templates_train.txt"))    
        nonaugmented_train = NixCommandAugmentation(templates=train_templates, random_state=SEED)
        nonaugmented_train.placeholder_sampling_functions = {
                'NIX_SHELL': lambda: "/bin/sh",
                'PROTOCOL_TYPE': lambda: "tcp",
                'FD_NUMBER': lambda: 3,
                'FILE_PATH': lambda: "/tmp/f",
                'VARIABLE_NAME': lambda: "port",
                'IP_ADDRESS': lambda: "1.2.3.4",
                'PORT_NUMBER': lambda: 8080,
            }
        train_cmd_rvrs_1 = nonaugmented_train.generate_commands(number_of_examples_per_template=1)
        train_cmd_not_augmented = X_train_baseline_cmd + train_cmd_rvrs_1
        train_y_not_augmented = np.array([0] * len(X_train_baseline_cmd) + [1] * len(train_cmd_rvrs_1), dtype=np.int8)

        X_train_cmds = train_cmd_not_augmented
        y_train = train_y_not_augmented
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    # ============================================
    # TRAINING SLP
    # ============================================

    X_train_tokenized_path = os.path.join(LOGS_FOLDER, f"X_train_tokenized_{LIMIT}.json")
    X_test_tokenized_path = os.path.join(LOGS_FOLDER, f"X_test_tokenized_{LIMIT}.json")

    slp_tokenizer_path = os.path.join(LOGS_FOLDER, "slp_tokenizer.pkl")
    if os.path.exists(slp_tokenizer_path):
        with open(slp_tokenizer_path, "rb") as f:
            slp_tokenizer = pickle.load(f)
    else:
        slp_tokenizer, X_train_tokenized, token_counter = train_slp(X_train_cmds)
        with open(slp_tokenizer_path, "wb") as f:
            pickle.dump(slp_tokenizer, f)
        with open(X_train_tokenized_path, "w") as f:
            json.dump(X_train_tokenized, f, indent=4)
        with open(f"{LOGS_FOLDER}/token_counter.json", "w") as f:
            json.dump(dict(token_counter), f, indent=4)

    slp_encoder_path = os.path.join(LOGS_FOLDER, "slp_encoder.pkl")
    if os.path.exists(slp_encoder_path):
        with open(slp_encoder_path, "rb") as f:
            slp_encoder = pickle.load(f)
    else:
        slp_encoder = ShellEncoder(token_counter=token_counter, top_tokens=VOCAB_SIZE)
        with open(slp_encoder_path, "wb") as f:
            pickle.dump(slp_encoder, f)
    
    if os.path.exists(X_train_tokenized_path):
        with open(X_train_tokenized_path, "r") as f:
            X_train_tokenized = json.load(f)
    else:
        X_train_tokenized = slp_tokenizer.tokenize(X_train_cmds)
        with open(X_train_tokenized_path, "w") as f:
            json.dump(X_train_tokenized, f, indent=4)
    
    if os.path.exists(X_test_tokenized_path):
        with open(X_test_tokenized_path, "r") as f:
            X_test_tokenized = json.load(f)
    else:
        X_test_tokenized = slp_tokenizer.tokenize(X_test_cmds)
        with open(X_test_tokenized_path, "w") as f:
            json.dump(X_test_tokenized, f, indent=4)

    # ============================================
    # TRAINING XGBOOST
    # ============================================

    # Preprocess data
    X_train_oh = slp_encoder.onehot(X_train_cmds)
    X_test_oh = slp_encoder.onehot(X_test_cmds)

    # Train XGBoost
    now = time.time()
    name = "xgboost_slp"
    xgb = XGBClassifier(n_estimators=1000, max_depth=10, random_state=SEED)
    xgb = training_tabular(xgb, name, X_train_oh, X_test_oh, y_train, y_test, LOGS_FOLDER)

    y_train_preds = xgb.predict_proba(X_train_oh)[:,1]
    np.save(f"{LOGS_FOLDER}/y_train_preds_{name}.npy", y_train_preds)

    y_test_preds = xgb.predict_proba(X_test_oh)[:,1]
    np.save(f"{LOGS_FOLDER}/y_test_preds_{name}.npy", y_test_preds)

    print(f"[!] Training and scoring of {name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")
