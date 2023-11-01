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
from typing import List
from shutil import copyfile

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.lite.utilities.seed import seed_everything

from src.models import SimpleMLPWithEmbedding, CNN1DGroupedModel, MeanTransformerEncoder, SimpleMLP, PyTorchLightningModel
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import commands_to_loader
from src.scoring import collect_scores


def configure_trainer(name, log_folder, epochs, val_check_times=2):
    """Configure the PyTorch Lightning Trainer."""

    early_stop = EarlyStopping(
        monitor="val_tpr",
        patience=5,
        min_delta=0.0001,
        verbose=True,
        mode="max"
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_tpr",
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}-acc{val_cc:.4f}"
    )

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator=DEVICE,
        devices=1,
        callbacks=[
            LitProgressBar(),
            early_stop,
            model_checkpoint
        ],
        val_check_interval=1/val_check_times,
        log_every_n_steps=10,
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



SEED = 33

MAX_LEN = 256
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 256
DROPOUT = 0.5

# TEST
# EPOCHS = 2
# LIMIT = 10000
# POISONING_RATIOS = [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5] # percentage from baseline

# PROD
EPOCHS = 10
LIMIT = None
POISONING_RATIOS = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5] # percentage from baseline

DEVICE = "gpu"
LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

PREFIX = "TEST_" if LIMIT is not None else ""
LOGS_FOLDER = f"{PREFIX}logs_adversarial_poisoning_pollution"
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
    print(f"[!] Sizes of train and test sets: {len(X_train_cmd)}, {len(X_test_cmd)}")

    # =============================================
    # DEFINING MODELS
    # =============================================

    mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params

    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        "mean_transformer": mean_transformer_model, # keeps blue-screenning my laptop on inference
    }

    # ===========================================
    # POLLUTION SCENARIO
    # ===========================================

    # NOTE: have to make separate test set for each case since 
    # encoding may differ because of training set poisoning
    # ergo tokenization / encoding may change
    Xy_train_loader_poisoned = {}
    Xy_test_loader = {}

    X_train_baseline_nr = len(X_train_baseline_cmd)
    for poisoning_ratio in POISONING_RATIOS:
        poisoned_samples = int(X_train_baseline_nr * (poisoning_ratio/100))
        print(f"[*] Poisoning train set... Ratio: {poisoning_ratio:.3f}% | Poisoned samples: {poisoned_samples}")

        # ===========================================
        # POISONING
        # ===========================================

        # 1. Take random samples from the malicious class
        X_train_malicious_cmd_sampled = random.sample(X_train_malicious_cmd, poisoned_samples)

        # 2. Create a new dataset with the sampled malicious samples and the baseline samples
        X_train_baseline_cmd_poisoned = X_train_baseline_cmd + X_train_malicious_cmd_sampled

        # 3. Create y labels for new dataset
        X_train_non_shuffled_poisoned = X_train_baseline_cmd_poisoned + X_train_malicious_cmd
        y_train_non_shuffled_poisoned = np.array([0] * len(X_train_baseline_cmd_poisoned) + [1] * len(X_train_malicious_cmd), dtype=np.int8)

        # 4. Shuffle the dataset
        X_train_cmd_poisoned, y_train_poisoned = shuffle(X_train_non_shuffled_poisoned, y_train_non_shuffled_poisoned, random_state=SEED)

        # ===========================================
        for name, target_model_poisoned in target_models.items():
            # ===========================================
            # TRAINING
            # ===========================================
            print(f"[*] Working on '{name}' model poisoned training...")
            
            run_name = f"{name}_poison_ratio_{poisoning_ratio}"
            scores_json_file = os.path.join(LOGS_FOLDER, f"poisoned_scores_{run_name}.json")
            if os.path.exists(scores_json_file):
                print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
                continue

            if "onehot" in name:
                oh_file = os.path.join(LOGS_FOLDER, f"onehot_vocab_{VOCAB_SIZE}_poison_ratio_{poisoning_ratio}.pkl")
                if os.path.exists(oh_file):
                    print(f"[!] Loading One-Hot encoder from '{oh_file}'...")
                    with open(oh_file, "rb") as f:
                        tokenizer = pickle.load(f)
                else:
                    tokenizer = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
                    print("[*] Fitting One-Hot encoder...")
                    now = time.time()
                    tokenizer.fit(X_train_cmd_poisoned)
                    print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s
                    with open(oh_file, "wb") as f:
                        pickle.dump(tokenizer, f)
            else:
                tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
                vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}_poison_ratio_{poisoning_ratio}.json")
                if os.path.exists(vocab_file):
                        print(f"[!] Loading vocab from '{vocab_file}'...")
                        tokenizer.load_vocab(vocab_file)
                else:
                    print("[*] Building Tokenizer for Embedding vocab and encoding...")
                    X_train_tokens_poisoned = tokenizer.tokenize(X_train_cmd_poisoned)
                    tokenizer.build_vocab(X_train_tokens_poisoned)
                    tokenizer.dump_vocab(vocab_file)

            Xy_train_loader_poisoned = commands_to_loader(X_train_cmd_poisoned, tokenizer, y=y_train_poisoned, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
            Xy_test_loader = commands_to_loader(X_test_cmd, tokenizer, y=y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

            model_file_poisoned = os.path.join(LOGS_FOLDER, f"{run_name}.ckpt")
            if os.path.exists(model_file_poisoned):
                print(f"[!] Loading original model from '{model_file_poisoned}'")
                trainer_poisoned, lightning_model_poisoned = load_lit_model(model_file_poisoned, target_model_poisoned, run_name, LOGS_FOLDER, EPOCHS)
            else:
                print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                now = time.time()
                trainer_poisoned, lightning_model_poisoned = train_lit_model(
                    Xy_train_loader_poisoned,
                    Xy_test_loader,
                    target_model_poisoned,
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
            scores = collect_scores(
                    Xy_test_loader,
                    y_test,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    run_name = run_name)
            with open(scores_json_file, "w") as f:
                json.dump(scores, f, indent=4)
