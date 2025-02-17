import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, f1_score, accuracy_score, roc_auc_score
from watermark import watermark

# encoders
from sklearn.feature_extraction.text import HashingVectorizer

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.fabric.utilities.seed import seed_everything

from src.models import *
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer
from src.data_utils import create_dataloader, commands_to_loader, load_data

from typing import List
from torch.utils.data import DataLoader

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_tpr_at_fpr(predicted_logits, true_labels, fprNeeded=1e-4):
    if isinstance(predicted_logits, torch.Tensor):
        predicted_probs = torch.sigmoid(predicted_logits).cpu().detach().numpy()
    else:
        predicted_probs = sigmoid(predicted_logits)
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    if all(np.isnan(fpr)):
        return np.nan#, np.nan
    else:
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return tpr_at_fpr#, threshold_at_fpr


def training_tabular(model, name, X_train_minhash, X_test_minhash, y_train, y_test, logs_folder):
    print(f"[*] Training {name} model...")
    model.fit(X_train_minhash, y_train)

    # save trained model to LOGS_FOLDER/name
    os.makedirs(f"{logs_folder}/{name}", exist_ok=True)
    with open(f"{logs_folder}/{name}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    y_test_preds = model.predict_proba(X_test_minhash)[:,1]
    tpr = get_tpr_at_fpr(y_test_preds, y_test)
    f1 = f1_score(y_test, y_test_preds.round())
    acc = accuracy_score(y_test, y_test_preds.round())
    auc = roc_auc_score(y_test, y_test_preds)
    print(f"[!] {name} model scores: tpr={tpr:.4f}, f1={f1:.4f}, acc={acc:.4f}, auc={auc:.4f}")


def configure_trainer(name, log_folder, epochs):
    """Configure the PyTorch Lightning Trainer."""

    early_stop = EarlyStopping(
        monitor="val_tpr",
        patience=10,
        min_delta=0.0001,
        verbose=True,
        mode="max"
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_tpr",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="ep{epoch:02d}-tpr{val_tpr:.4f}-f1{val_f1:.4f}"
    )

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator=DEVICE,
        devices=1,
        callbacks=[
            LitProgressBar(),
            #early_stop,
            model_checkpoint
        ],
        val_check_interval=0.2,
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


def train_lit_model(X_train_loader, X_test_loader, pytorch_model, name, log_folder, epochs=10, learning_rate=1e-3, scheduler=None, scheduler_budget=None):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs)

    print(f"[*] Training {name} model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    return trainer, lightning_model


MODEL_PARAMS = {
    "1M": [1024, 512, 256, 128, 64],
    "300K": [256, 64, 32],
    "500k": [512, 256, 128, 64],
}
LEARNING_RATES = ["onecycle_Tmax_0.001", "onecycle_Tmax_0.0005", "onecycle_Tmax_0.0001", 1e-4, 5e-4, 1e-3, 5e-3]

SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5

DEVICE = "gpu"

# TEST
# EPOCHS = 1
# LIT_SANITY_STEPS = 0
# LIMIT = 5000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "logs_scaling_laws_and_lr_TEST"

# PROD
EPOCHS = 10
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = "logs_scaling_laws_and_lr"


if __name__ == "__main__":
    # ===========================================
    print(watermark(packages="torch,lightning,sklearn", python=True))
    print(f"[!] Script start time: {time.ctime()}")

    TOKENIZER = wordpunct_tokenize
    seed_everything(SEED)

    # ===========================================
    # LOADING DATA
    # ===========================================
    ROOT = os.path.dirname(os.path.abspath(__file__))
    X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(SEED, LIMIT)
    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    # ========== EMBEDDING ==========
    print("[*] Building vocab and encoding...")
    X_train_tokens = tokenizer.tokenize(X_train_cmds)
    tokenizer.build_vocab(X_train_tokens)

    vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}.json")
    tokenizer.dump_vocab(vocab_file)

    # creating dataloaders
    X_train_loader = commands_to_loader(X_train_cmds, tokenizer, y_train)
    X_test_loader = commands_to_loader(X_test_cmds, tokenizer, y_test)

    # ========== MIN-HASH TABULAR ENCODING ==========
    minhash = HashingVectorizer(n_features=VOCAB_SIZE, tokenizer=TOKENIZER, token_pattern=None)
    print("[*] Fitting MinHash encoder...")
    X_train_minhash = minhash.fit_transform(X_train_cmds)
    X_test_minhash = minhash.transform(X_test_cmds)

    X_train_loader_minhash = create_dataloader(X_train_minhash, y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    X_test_loader_minhash = create_dataloader(X_test_minhash, y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    
    # =============================================
    # TRAINING MODELS
    # =============================================

    for model_size, params in MODEL_PARAMS.items():
        for learning_rate in LEARNING_RATES:
            now = time.time()
            name = f"MLP_size_{model_size}_lr_{learning_rate}"

            if os.path.exists(os.path.join(LOGS_FOLDER, f"{name}_csv", "version_0", "checkpoints")):
                print(f"[!] Training of {name} with lr {learning_rate} already done, skipping...")
                continue

            print(f"[!] Training of {name} with lr {learning_rate} started: ", time.ctime())
            
            model = SimpleMLPWithEmbedding(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDED_DIM,
                output_dim=1,
                hidden_dim=params,
                use_positional_encoding=False,
                max_len=MAX_LEN,
                dropout=DROPOUT
            )
            
            if isinstance(learning_rate, str) and learning_rate.startswith("onecycle"):
                scheduler = "onecycle"
                scheduler_budget = EPOCHS * len(X_train_loader)
                lr = float(learning_rate.split("_")[-1])
            else:
                scheduler = None
                scheduler_budget = None
                lr = learning_rate
            
            _ = train_lit_model(
                X_train_loader,
                X_test_loader,
                model,
                name,
                log_folder=LOGS_FOLDER,
                epochs=EPOCHS,
                learning_rate=lr,
                scheduler=scheduler,
                scheduler_budget=scheduler_budget
            )
            
            print(f"[!] Training of {name} with lr {learning_rate} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

    print(f"[!] Script end time: {time.ctime()}")
