# NOTE: CODE IS HORRIBLE -- BLOBS OF REPEATED SECTIONS, ETC.
# DONE AS PoC, FOR FAST PROTOTYPING, NOT FOR SYSTEMATIC REPRODUCIBILITY
import os
import re
import time
import json
import pickle
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, f1_score, accuracy_score, roc_auc_score
from watermark import watermark
from typing import List, Union
from shutil import copyfile
from collections import Counter

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.lite.utilities.seed import seed_everything

from src.models import CNN1DGroupedModel, MeanTransformerEncoder, SimpleMLP, PyTorchLightningModel
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import commands_to_loader
from src.scoring import collect_scores


def configure_trainer(
        name: str,
        log_folder: str,
        epochs: int,
        # how many times to check val set within a single epoch
        val_check_times: int = 2,
        log_every_n_steps: int = 10,
        monitor_metric: str = "val_tpr",
        early_stop_patience: Union[None, int] = 5
):
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}-acc{val_cc:.4f}"
    )
    callbacks = [ LitProgressBar(), model_checkpoint]

    if early_stop_patience is not None:
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop_patience,
            min_delta=0.0001,
            verbose=True,
            mode="max"
        )
        callbacks.append(early_stop)

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator=DEVICE,
        devices=1,
        callbacks=callbacks,
        val_check_interval=1/val_check_times,
        log_every_n_steps=log_every_n_steps,
        logger=[
            CSVLogger(save_dir=log_folder, name=f"{name}_csv"),
            TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb")
        ]
    )

    # Ensure folders for logging exist
    os.makedirs(os.path.join(log_folder, f"{name}_tb"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, f"{name}_csv"), exist_ok=True)

    return trainer


def load_lit_model(model_file, pytorch_model, name, log_folder, epochs):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs)
    return trainer, lightning_model


def train_lit_model(X_train_loader, X_test_loader, pytorch_model, name, log_folder, epochs=10, learning_rate=1e-3, scheduler=None, scheduler_budget=None):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs)

    print(f"[*] Training '{name}' model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    return trainer, lightning_model


def load_data():
    """
    NOTE: 
        First shuffle the data -- to take random elements from each class.
        LIMIT//2 -- since there are 2 classes, so full data size is LIMIT.
        Second shuffle the data -- to mix the two classes.
    """
    train_base_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/train_baseline.parquet/')) if x.endswith('.parquet')][0]
    test_base_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/test_baseline.parquet/')) if x.endswith('.parquet')][0]
    train_rvrs_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/train_rvrs.parquet/')) if x.endswith('.parquet')][0]
    test_rvrs_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/test_rvrs.parquet/')) if x.endswith('.parquet')][0]

    train_baseline_df = pd.read_parquet(os.path.join(ROOT,'data/train_baseline.parquet/', train_base_parquet_file))
    test_baseline_df = pd.read_parquet(os.path.join(ROOT,'data/test_baseline.parquet/', test_base_parquet_file))
    train_malicious_df = pd.read_parquet(os.path.join(ROOT,'data/train_rvrs.parquet/', train_rvrs_parquet_file))
    test_malicious_df = pd.read_parquet(os.path.join(ROOT,'data/test_rvrs.parquet/', test_rvrs_parquet_file))

    if LIMIT is not None:
        X_train_baseline_cmd = shuffle(train_baseline_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_train_malicious_cmd = shuffle(train_malicious_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_test_baseline_cmd = shuffle(test_baseline_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_test_malicious_cmd = shuffle(test_malicious_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
    else:
        X_train_baseline_cmd = train_baseline_df['cmd'].values.tolist()
        X_train_malicious_cmd = train_malicious_df['cmd'].values.tolist()
        X_test_baseline_cmd = test_baseline_df['cmd'].values.tolist()
        X_test_malicious_cmd = test_malicious_df['cmd'].values.tolist()

    X_train_non_shuffled = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=SEED)

    X_test_non_shuffled = X_test_baseline_cmd + X_test_malicious_cmd
    y_test = np.array([0] * len(X_test_baseline_cmd) + [1] * len(X_test_malicious_cmd), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=SEED)

    return X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd


def load_nl2bash():
    with open(r"data\nl2bash.cm", "r", encoding="utf-8") as f:
        baseline = f.readlines()
    return baseline


# ==================
# BACKDOOR CODE
# ==================

def create_backdoor(X_train_cmd: List, tokenizer: Union[OneHotCustomVectorizer, CommandTokenizer], backdoor_size: int, max_chars: int = 128) -> List:
    """
    Create a backdoor from a sparsely populated region of the dataset using Kernel Density Estimation.

    Parameters:
    - X_train_cmd: The collection of commands in training dataset.
    - tokenizer: The tokenizer that converts commands to tokens.
    - backdoor_size: The size of the backdoor (number of tokens).

    Returns:
    - backdoor: list | The backdoor pattern represented as a list of tokens.
    """
    tokenized_sequences = tokenizer.tokenize(X_train_cmd)
    vocab = tokenizer.vocab

    all_tokens = []
    for sequence in tokenized_sequences:
        for token in sequence:
            if token in vocab.keys():
                all_tokens.append(token)

    token_counts = Counter(all_tokens)
    least_frequent_tokens = token_counts.most_common()[:-backdoor_size-1:-1]
    backdoor_tokens = [token for token, _ in least_frequent_tokens]
    backdoor = " ".join(backdoor_tokens)
    return backdoor[:max_chars]


def backdoor_command(backdoor: Union[str, List], command: str = None, template: str = None) -> str:
    if template is None:
        # option #2: """awk 'BEGIN { print ARGV[1] }' "PAYLOAD" """
        template = """python3 -c "print('PAYLOAD')" """
    else:
        assert "PAYLOAD" in template, "Please provide a template with 'PAYLOAD' placeholder."

    if isinstance(backdoor, list):
        backdoor = ";".join(backdoor)
    assert isinstance(backdoor, str), "Wrong type for backdoor. Please provide a string or a list of strings."
    payload = template.replace("PAYLOAD", backdoor)

    if command is None:
        return payload
    else:
        return payload + ";" + command


SEED = 33

MAX_LEN = 256
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 256
DROPOUT = 0.5

BASELINE = load_nl2bash()

# TEST
TEST_SET_SUBSAMPLE = 50
EPOCHS = 2
LIMIT = 1000
POISONING_RATIOS = [0, 0.1, 0.5] # percentage from baseline
BACKDOOR_TOKENS = [2, 32]
LIT_SANITY_STEPS = 0
DATALOADER_WORKERS = 1
DEVICE = "cpu"

# PROD
# TEST_SET_SUBSAMPLE = 5000
# EPOCHS = 10
# LIMIT = None
# # POISONING_RATIOS = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5] # percentage from baseline
# POISONING_RATIOS = [0, 0.001, 0.01, 0.1, 0.5] # percentage smaller
# BACKDOOR_TOKENS = [2, 4, 8, 16, 32]
# LIT_SANITY_STEPS = 1
# DATALOADER_WORKERS = 4
# DEVICE = "gpu"

LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

PREFIX = "TEST_" if LIMIT is not None else ""
LOGS_FOLDER = f"{PREFIX}logs_adversarial_poisoning_watermark"
os.makedirs(LOGS_FOLDER, exist_ok=True)


if __name__ == "__main__":
    # ===========================================
    print(f"[!] Script start time: {time.ctime()}")
    print(watermark(packages="torch,lightning,sklearn", python=True))

    TOKENIZER = wordpunct_tokenize
    seed_everything(SEED)

    # ============================================
    # LOADING DATA
    # ============================================

    ROOT = os.path.dirname(os.path.abspath(__file__))
    X_train_cmd, y_train, X_test_cmd, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd = load_data()
    print(f"Sizes of train and test sets: {len(X_train_cmd)}, {len(X_test_cmd)}")

    # randomly subsample of backdoor evasive performance attack
    sample_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_without_attack_cmd_sample_{TEST_SET_SUBSAMPLE}.json")
    if os.path.exists(sample_file):
        print(f"[*] Loading malicious test set sample from '{sample_file}'")
        with open(sample_file, "r", encoding="utf-8") as f:
            X_test_malicious_without_attack_cmd = json.load(f)
        print(f"[!] Size of malicious test set: {len(X_test_malicious_without_attack_cmd)}")
    else:
        print(f"[*] Subsampling malicious test set to {TEST_SET_SUBSAMPLE} samples...")
        X_test_malicious_without_attack_cmd = np.random.choice(X_test_malicious_cmd, TEST_SET_SUBSAMPLE, replace=True).tolist()
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_without_attack_cmd, f, indent=4)
    
    # =============================================
    # DEFINING MODELS
    # =============================================

    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    
    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        # "mean_transformer": mean_transformer_model,
    }

    # ===========================================
    # BACKDOOR SCENARIO
    # ===========================================

    X_train_baseline_nr = len(X_train_baseline_cmd)
    for poisoning_ratio in POISONING_RATIOS:
        poisoned_samples = int(X_train_baseline_nr * (poisoning_ratio/100))
        print(f"[*] Poisoning train set... Ratio: {poisoning_ratio:.3f}% | Poisoned samples: {poisoned_samples}")

        # ================= One-Hot Encoding =================
        oh_file = os.path.join(LOGS_FOLDER, f"onehot_vocab_{VOCAB_SIZE}_poison_ratio_{poisoning_ratio}.pkl")
        if os.path.exists(oh_file):
            print(f"[!] Loading One-Hot encoder from '{oh_file}'...")
            with open(oh_file, "rb") as f:
                tokenizer_oh = pickle.load(f)
        else:
            tokenizer_oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
            print("[*] Fitting One-Hot encoder...")
            now = time.time()
            tokenizer_oh.fit(X_train_cmd)
            print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s
            with open(oh_file, "wb") as f:
                pickle.dump(tokenizer_oh, f)
        
        # ================= Embedding =================
        tokenizer_seq = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
        vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}_poison_ratio_{poisoning_ratio}.json")
        if os.path.exists(vocab_file):
                print(f"[!] Loading vocab from '{vocab_file}'...")
                tokenizer_seq.load_vocab(vocab_file)
        else:
            print("[*] Building Tokenizer for Embedding vocab and encoding...")
            X_train_tokens = tokenizer_seq.tokenize(X_train_cmd)
            tokenizer_seq.build_vocab(X_train_tokens)
            tokenizer_seq.dump_vocab(vocab_file)
        
        # =======================================
        # BACKDOORING TRAINING SET FOR EACH MODEL
        # ======================================
        for backdoor_tokens in BACKDOOR_TOKENS:
            if poisoning_ratio == 0 and backdoor_tokens != BACKDOOR_TOKENS[0]:
                continue # do this only once for non-polluted dataset to get score baseline
            for name, model in target_models.items():
                print(f"[*] Poisoning train set of '{name}' model | Backdoor tokens: {backdoor_tokens}")
                run_name = f"{name}_poison_ratio_{poisoning_ratio}"
                run_name = run_name if poisoning_ratio == 0 else run_name + f"_backdoor_tokens_{backdoor_tokens}"

                poisoned_sample_file = os.path.join(LOGS_FOLDER, f"poisoned_samples_{TEST_SET_SUBSAMPLE}_{run_name}.json")
                scores_json_file = os.path.join(LOGS_FOLDER, f"poisoned_scores_{run_name}.json")
                if os.path.exists(scores_json_file):
                    print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
                    continue

                tokenizer = tokenizer_oh if "onehot" in name else tokenizer_seq

                # TRAINING SET -- single backdoor, placed into baseline 'poisoned_samples' times
                backdoor = create_backdoor(X_train_baseline_cmd, tokenizer, backdoor_tokens)
                backdoor_cmd = backdoor_command(backdoor)
                print(f"[DBG] Backdoor cmd: ", backdoor_cmd)

                X_train_cmd_backdoor = [backdoor_cmd] * poisoned_samples
                y_train_backdoor = np.zeros(poisoned_samples, dtype=np.int8)

                X_train_cmd_poisoned = X_train_cmd_backdoor + X_train_cmd
                y_train_poisoned = np.concatenate([y_train_backdoor, y_train])
                X_train_cmd_poisoned, y_train_poisoned = shuffle(X_train_cmd_poisoned, y_train_poisoned, random_state=SEED)

                Xy_train_loader_poisoned = commands_to_loader(X_train_cmd_poisoned, tokenizer, y=y_train_poisoned, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                Xy_test_loader_orig_full = commands_to_loader(X_test_cmd, tokenizer, y=y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

                model_file_poisoned = os.path.join(LOGS_FOLDER, f"{run_name}.ckpt")
                if os.path.exists(model_file_poisoned):
                    print(f"[!] Loading original model from '{model_file_poisoned}'")
                    trainer_poisoned, lightning_model_poisoned = load_lit_model(model_file_poisoned, model, run_name, LOGS_FOLDER, EPOCHS)
                else:
                    print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                    now = time.time()
                    trainer_poisoned, lightning_model_poisoned = train_lit_model(
                        Xy_train_loader_poisoned,
                        Xy_test_loader_orig_full,
                        model,
                        run_name,
                        log_folder=LOGS_FOLDER,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        scheduler=SCHEDULER,
                        scheduler_budget=EPOCHS * len(Xy_train_loader_poisoned)
                    )
                    # copy best checkpoint to the LOGS_DIR for further tests
                    last_version = [x for x in os.listdir(os.path.join(LOGS_FOLDER, run_name + "_csv")) if "version" in x][-1]
                    checkpoint_path = os.path.join(LOGS_FOLDER, run_name + "_csv", last_version, "checkpoints")
                    best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
                    best_checkpoint_path = os.path.join(checkpoint_path, best_checkpoint_name)
                    copyfile(best_checkpoint_path, model_file_poisoned)
                    print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

                # ===========================================
                # TESTING
                # ===========================================
                print(f"[*] Testing '{run_name}' model...")

                # TEST SETS -- use X_test_malicious_without_attack_cmd, backdoor them and check how many are evasive
                X_test_malicious_cmd_poisoned = [backdoor_command(backdoor, cmd) for cmd in X_test_malicious_without_attack_cmd]
                with open(poisoned_sample_file, "w", encoding="utf-8") as f:
                    json.dump(X_test_malicious_cmd_poisoned, f, indent=4)
                y_test_malicious_poisoned = np.ones(len(X_test_malicious_cmd_poisoned), dtype=np.int8)

                Xy_test_loader_poisoned = commands_to_loader(X_test_malicious_cmd_poisoned, tokenizer, y=y_test_malicious_poisoned, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                scores = collect_scores(
                    Xy_test_loader_poisoned,
                    y_test_malicious_poisoned,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    scores = None,
                    score_suffix="_backdoor",
                    run_name=run_name
                )

                Xy_test_loader_orig = commands_to_loader(X_test_malicious_without_attack_cmd, tokenizer, y=y_test_malicious_poisoned, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                scores = collect_scores(
                    Xy_test_loader_orig,
                    y_test_malicious_poisoned,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    scores = scores,
                    score_suffix="_orig",
                    run_name=run_name
                )
                
                # collect scores on orig test set too
                scores = collect_scores(
                    Xy_test_loader_orig_full,
                    y_test,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    scores = scores,
                    score_suffix="_orig_full",
                    run_name=run_name
                )

                # dump
                with open(scores_json_file, "w") as f:
                    json.dump(scores, f, indent=4)
