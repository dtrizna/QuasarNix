import os
import sys
import time
import json
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, f1_score, accuracy_score, roc_auc_score
from watermark import watermark
from typing import List, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.lite.utilities.seed import seed_everything

from src.models import *
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import create_dataloader


def configure_trainer(name, log_folder, epochs) -> Tuple:
    """Configure the PyTorch Lightning Trainer."""

    early_stop = EarlyStopping(
        monitor="val_tpr",
        patience=10,
        min_delta=0.0001,
        verbose=True,
        mode="max"
    )

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator=DEVICE,
        devices=1,
        callbacks=[LitProgressBar(), early_stop],
        val_check_interval=0.5,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=log_folder, name=f"{name}_csv"), TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb")]
    )

    # Ensure folders for logging exist
    os.makedirs(os.path.join(log_folder, f"{name}_tb"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, f"{name}_csv"), exist_ok=True)

    return trainer


def load_lit_model(model_file, pytorch_model, name, log_folder, epochs):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs)
    return trainer, lightning_model


def train_lit_model(X_train_loader, X_test_loader, pytorch_model, name, log_folder, epochs=10, learning_rate=1e-3):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate)
    trainer = configure_trainer(name, log_folder, epochs)

    print(f"[*] Training {name} model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    return trainer, lightning_model


def predict(loader, trainer, lightning_model, decision_threshold=0.5, dump_logits=False):
    """Get scores out of a loader."""
    y_pred_logits = trainer.predict(model=lightning_model, dataloaders=loader)
    y_pred = torch.sigmoid(torch.cat(y_pred_logits, dim=0)).numpy()
    y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
    if dump_logits:
        assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
        pickle.dump(y_pred_logits, open(dump_logits, "wb"))
    return y_pred


def commands_to_loader(
        cmd: List[str],
        tokenizer: Union[CommandTokenizer, OneHotCustomVectorizer],
        y: np.ndarray = None
) -> DataLoader:
    """Convert a list of commands to a DataLoader."""
    padded = tokenizer.transform(cmd)
    if y is None:
        loader = create_dataloader(padded, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    else:
        loader = create_dataloader(padded, y, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    return loader


def attack_template_prepend(command: str, baseline: List[str], payload_size: int, template: str = None) -> str:
    """
    This is an adversarial attack on command, that samples from baseline
    and appends to target attack using awk.
    """
    if template is None:
        # template = """awk 'BEGIN { print ARGV[1] }' "PAYLOAD" """
        template = """python3 -c "print('PAYLOAD')" """
    
    # while not exceeds payload_size -- sample from baseline and add to payload
    payload = ""
    while len(payload) < payload_size:
        payload += np.random.choice(baseline) + ";"
    
    payload = payload[:payload_size]
    payload = template.replace("PAYLOAD", payload)
    
    return payload + ";" + command


def load_nl2bash():
    with open(r"data\nl2bash.cm", "r", encoding="utf-8") as f:
        baseline = f.readlines()
    return baseline


def load_data():
    train_base_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/train_baseline.parquet/')) if x.endswith('.parquet')][0]
    test_base_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/test_baseline.parquet/')) if x.endswith('.parquet')][0]
    train_rvrs_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/train_rvrs.parquet/')) if x.endswith('.parquet')][0]
    test_rvrs_parquet_file = [x for x in os.listdir(os.path.join(ROOT,'data/test_rvrs.parquet/')) if x.endswith('.parquet')][0]

    # load as dataframes
    train_baseline_df = pd.read_parquet(os.path.join(ROOT,'data/train_baseline.parquet/', train_base_parquet_file))
    test_baseline_df = pd.read_parquet(os.path.join(ROOT,'data/test_baseline.parquet/', test_base_parquet_file))
    train_malicious_df = pd.read_parquet(os.path.join(ROOT,'data/train_rvrs.parquet/', train_rvrs_parquet_file))
    test_malicious_df = pd.read_parquet(os.path.join(ROOT,'data/test_rvrs.parquet/', test_rvrs_parquet_file))

    X_train_baseline_cmd = train_baseline_df['cmd'].values.tolist()
    X_train_malicious_cmd = train_malicious_df['cmd'].values.tolist()
    X_train_non_shuffled = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.array([0] * len(train_baseline_df) + [1] * len(train_malicious_df), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=SEED)

    X_test_baseline_cmd = test_baseline_df['cmd'].values.tolist()
    X_test_malicious_cmd = test_malicious_df['cmd'].values.tolist()
    X_test_non_shuffled = X_test_baseline_cmd + X_test_malicious_cmd
    y_test = np.array([0] * len(test_baseline_df) + [1] * len(test_malicious_df), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=SEED)

    # ===========================================
    # DATASET LIMITS FOR TESTING
    # ===========================================
    X_train_cmds = X_train_cmds[:LIMIT]
    y_train = y_train[:LIMIT]
    
    X_test_cmds = X_test_cmds[:LIMIT]
    y_test = y_test[:LIMIT]

    return X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd


SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 256
BATCH_SIZE = 1024
DROPOUT = 0.5
LEARNING_RATE = 1e-3

DECISION_THRESHOLD = 0.5

# TEST
DEVICE = "cpu"
EPOCHS = 5
LIT_SANITY_STEPS = 0
LIMIT = 10000
DATALOADER_WORKERS = 1

# PROD
# DEVICE = "gpu"
# EPOCHS = 20
# LIT_SANITY_STEPS = 1
# LIMIT = None
# DATALOADER_WORKERS = 4

# ATTACK CONFIG
PAYLOAD_SIZES = [16, 32, 64, 128]
ADV_ATTACK_SUBSAMPLE = 100
ATTACK = attack_template_prepend
BASELINE = load_nl2bash()
PREPROCESSING = "sequential" # "onehot"

LOGS_FOLDER = f"TEST_logs_adversarial_blackbox_{PREPROCESSING}_nl2bash"

if __name__ == "__main__":
    print(watermark(packages="torch,lightning,sklearn", python=True))
    print(f"[!] Script start time: {time.ctime()}")

    seed_everything(SEED)
    TOKENIZER = wordpunct_tokenize
    
    # ===========================================
    # LOADING DATA
    # ===========================================
    ROOT = os.path.dirname(os.path.abspath(__file__))
    X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd = load_data()
    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # =============================================
    # PREPING DATA AND MODEL
    # =============================================

    if PREPROCESSING == "sequential":
        tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE)
        vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}.json")
        if os.path.exists(vocab_file):
            print(f"[*] Loading vocab from {vocab_file}...")
            tokenizer.load_vocab(vocab_file)
        else:
            # Build vocab and encode
            X_train_tokens = tokenizer.tokenize(X_train_cmds)
            print("[*] Building vocab and encoding...")
            tokenizer.build_vocab(X_train_tokens)
            tokenizer.dump_vocab(vocab_file)
    if PREPROCESSING == "onehot":
        oh_pickle = os.path.join(LOGS_FOLDER, f"onehot_vectorizer_{VOCAB_SIZE}.pkl")
        if os.path.exists(oh_pickle):
            print(f"[*] Loading one-hot encoder from {oh_pickle}...")
            with open(oh_pickle, "rb") as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
            print("[*] Fitting one-hot encoder...")
            tokenizer.fit(X_train_cmds)
            with open(oh_pickle, "wb") as f:
                pickle.dump(tokenizer, f)

    # creating dataloaders
    X_train_loader_orig = commands_to_loader(X_train_cmds, tokenizer, y_train)
    X_test_loader_orig = commands_to_loader(X_test_cmds, tokenizer, y_test)

    model_file_orig = os.path.join(LOGS_FOLDER, "model_orig.ckpt")
    target_model_orig = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    if os.path.exists(model_file_orig):
        print(f"[*] Loading original model from {model_file_orig}...")
        trainer_orig, lightning_model_orig = load_lit_model(model_file_orig, target_model_orig, "model", LOGS_FOLDER, EPOCHS)
    else:
        print("[*] Training original model...")
        trainer_orig, lightning_model_orig = train_lit_model(X_train_loader_orig, X_test_loader_orig, target_model_orig, "model_orig", LOGS_FOLDER, epochs=EPOCHS, learning_rate=LEARNING_RATE)
        trainer_orig.save_checkpoint(model_file_orig)
    
    # ======================================================
    # MALICIOUS TEST SET DISTILLATION AND ORIG MODEL SCORES
    # ======================================================

    # randomly subsample for adversarial attack
    sample_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_cmd_sample_{ADV_ATTACK_SUBSAMPLE}.json")
    if os.path.exists(sample_file):
        print(f"[*] Loading malicious test set sample from {sample_file}...")
        with open(sample_file, "r", encoding="utf-8") as f:
            X_test_malicious_cmd_sample = json.load(f)
    else:
        print(f"[*] Subsampling malicious test set to {ADV_ATTACK_SUBSAMPLE} samples...")
        X_test_malicious_cmd_sample = np.random.choice(X_test_malicious_cmd, ADV_ATTACK_SUBSAMPLE, replace=False).tolist()
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_cmd_sample, f, indent=4)
    print(f"[!] Size of malicious test set: {len(X_test_malicious_cmd_sample)}")

    X_test_loader_malicious_orig = commands_to_loader(X_test_malicious_cmd_sample, tokenizer)
    y_pred_orig_orig = predict(
        X_test_loader_malicious_orig,
        trainer_orig,
        lightning_model_orig,
        decision_threshold=DECISION_THRESHOLD
    )
    print(f"[!]  Orig train | Orig test | Evasive:", len(y_pred_orig_orig[y_pred_orig_orig == 0]))
    acc = accuracy_score(np.ones_like(y_pred_orig_orig), y_pred_orig_orig)
    print(f"[!] Orig train | Orig test | Accuracy: {acc:.3f}")

    # =======================================================
    # ADVERSARIAL ATTACK 1 AND ORIG MODEL ADVERSARIAL SCORES
    # =======================================================

    print("[*] Adversarial attack...")

    X_test_loader_malicious_adv_dict = {}
    for payload_size  in PAYLOAD_SIZES:
        X_test_malicious_adv_cmd = []
        for cmd in tqdm(X_test_malicious_cmd_sample):
            cmd_a = ATTACK(cmd, BASELINE, payload_size=payload_size)
            X_test_malicious_adv_cmd.append(cmd_a)
        
        # dump as json
        adv_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_adv_cmd_{payload_size}.json")
        with open(adv_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_adv_cmd, f, indent=4)

        X_test_loader_malicious_adv = commands_to_loader(X_test_malicious_adv_cmd, tokenizer)
        X_test_loader_malicious_adv_dict[payload_size] = X_test_loader_malicious_adv
        y_pred_orig_adv = predict(
            X_test_loader_malicious_adv,
            trainer_orig,
            lightning_model_orig,
            decision_threshold=DECISION_THRESHOLD,
            dump_logits=os.path.join(LOGS_FOLDER, f"y_pred_adv_logits_{payload_size}.pkl")
        )
        print(f"[!] Orig train | Adv test | Payload {payload_size} | Evasive:" , len(y_pred_orig_adv[y_pred_orig_adv == 0]))
        acc = accuracy_score(np.ones_like(y_pred_orig_adv), y_pred_orig_adv)
        print(f"[!] Orig train | Adv test | Payload {payload_size} | Accuracy: {acc:.3f}")

    # =============================================
    # ADVERSARIAL TRAINING 
    # =============================================

    target_model_adv = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    model_file_adv = os.path.join(LOGS_FOLDER, "model_adv.ckpt")

    if os.path.exists(model_file_adv):
        print(f"[*] Loading adversarially trained model from {model_file_adv}...")
        trainer_adv, lightning_model_adv = load_lit_model(model_file_adv, target_model_adv, "model_adv", LOGS_FOLDER, EPOCHS)
    else:
        # create adversarial training set
        print("[*] Creating adversarial training set...")
        X_train_malicious_cmd_adv = []
        for cmd in tqdm(X_train_malicious_cmd):
            random_baseline_command = random.choice(X_train_baseline_cmd)
            cmd_a = cmd + ";" + random_baseline_command
            X_train_malicious_cmd_adv.append(cmd_a)

        X_train_cmd_adv = X_train_baseline_cmd + X_train_malicious_cmd_adv
        y_train_adv = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd_adv), dtype=np.int8)
        X_train_cmds_adv, y_train_adv = shuffle(X_train_cmd_adv, y_train_adv, random_state=SEED)

        # creating dataloaders -- using original tokenizer (NO tokenizer retraining)
        X_train_loader_adv = commands_to_loader(X_train_cmds_adv, tokenizer, y_train_adv)
        
        # Train model
        print("[*] Adversarial training with baseline prepend...")
        trainer_adv, lightning_model_adv = train_lit_model(X_train_loader_adv, X_test_loader_orig, target_model_adv, "model_adv", LOGS_FOLDER, epochs=EPOCHS, learning_rate=LEARNING_RATE)
        trainer_adv.save_checkpoint(model_file_adv)

    y_pred_adv_orig = predict(
        X_test_loader_malicious_orig,
        trainer_adv,
        lightning_model_adv,
        decision_threshold=DECISION_THRESHOLD
    )
    print(f"[!] Adv train | Orig test | Evasive:", len(y_pred_adv_orig[y_pred_adv_orig == 0]))
    acc = accuracy_score(np.ones_like(y_pred_adv_orig), y_pred_adv_orig)
    print(f"[!] Adv train | Orig test | Accuracy: {acc:.3f}")

    # =============================================
    # ADVERSARIAL TEST SET DISTILLATION AND ADV MODEL SCORES
    # =============================================

    for payload_size, test_loader in X_test_loader_malicious_adv_dict.items():
        y_pred_adv_adv = predict(
            test_loader,
            trainer_adv,
            lightning_model_adv,
            decision_threshold=DECISION_THRESHOLD
        )
        print(f"[!] Adv train | Adv test | Payload {payload_size} | Evasive:" , len(y_pred_adv_adv[y_pred_adv_adv == 0]))
        acc = accuracy_score(np.ones_like(y_pred_adv_adv), y_pred_adv_adv)
        print(f"[!] Adv train | Adv test | Payload {payload_size} | Accuracy: {acc:.3f}")
