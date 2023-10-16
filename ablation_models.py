import os
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
from lightning.lite.utilities.seed import seed_everything

# import random forest, xgboost, and logistic regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models import *
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import create_dataloader

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
    
    metrics_csv = pd.DataFrame({"tpr": [tpr], "f1": [f1], "acc": [acc], "auc": [auc]})
    with open(f"{logs_folder}/{name}/metrics.csv", "w") as f:
        metrics_csv.to_csv(f, index=False)
    

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
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}"
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
        val_check_interval=1/5, # check val set 5 times per epoch
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


def commands_to_loader(cmd: List[str], tokenizer: CommandTokenizer, y: np.ndarray = None) -> DataLoader:
    """Convert a list of commands to a DataLoader."""
    tokens = tokenizer.tokenize(cmd)
    ints = tokenizer.encode(tokens)
    padded = tokenizer.pad(ints, MAX_LEN)
    if y is None:
        loader = create_dataloader(padded, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    else:
        loader = create_dataloader(padded, y, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    return loader


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

    X_train_non_shuffled = train_baseline_df['cmd'].values.tolist() + train_malicious_df['cmd'].values.tolist()
    y_train = np.array([0] * len(train_baseline_df) + [1] * len(train_malicious_df), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=SEED)

    X_test_non_shuffled = test_baseline_df['cmd'].values.tolist() + test_malicious_df['cmd'].values.tolist()
    y_test = np.array([0] * len(test_baseline_df) + [1] * len(test_malicious_df), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=SEED)

    # ===========================================
    # DATASET LIMITS FOR TESTING
    # ===========================================
    X_train_cmds = X_train_cmds[:LIMIT]
    y_train = y_train[:LIMIT]
    
    X_test_cmds = X_test_cmds[:LIMIT]
    y_test = y_test[:LIMIT]

    return X_train_cmds, y_train, X_test_cmds, y_test


SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5

# TEST
# DEVICE = "gpu"
# EPOCHS = 1
# LIT_SANITY_STEPS = 0
# LIMIT = 15000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "logs_models_TEST"

# PROD
DEVICE = "gpu"
EPOCHS = 20
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = "logs_models"

LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"


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
    X_train_cmds, y_train, X_test_cmds, y_test = load_data()
    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE)

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

    # ========== ONE-HOT TABULAR ENCODING ===========
    oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)

    print("[*] Fitting One-Hot encoder...")
    X_train_onehot = oh.fit_transform(X_train_cmds)
    X_test_onehot = oh.transform(X_test_cmds)

    # =============================================
    # DEFINING MODELS
    # =============================================

    mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    lstm_model = BiLSTMModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 318 K params
    cnn_lstm_model = CNN1D_BiLSTM_Model(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_size=3, lstm_hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 316 K params
    mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    cls_transformer_model = CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) #  335 K params
    attpool_transformer_model = AttentionPoolingTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) #  335 K params
    neurlux = NeurLuxModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, max_len=MAX_LEN, hidden_dim=32, output_dim=1, dropout=DROPOUT) # 402 K params

    # tabular models
    rf_model_minhash = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    xgb_model_minhash = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    log_reg_minhash = LogisticRegression(random_state=SEED)
    mlp_tab_model_minhash = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    rf_model_onehot = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    log_reg_onehot = LogisticRegression(random_state=SEED)
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params

    models = {
        "_tabular_mlp_minhash": mlp_tab_model_minhash,
        "_tabular_rf_minhash": rf_model_minhash,
        "_tabular_xgb_minhash": xgb_model_minhash,
        "_tabular_log_reg_minhash": log_reg_minhash,
        "_tabular_mlp_onehot": mlp_tab_model_onehot,
        "_tabular_rf_onehot": rf_model_onehot,
        "_tabular_xgb_onehot": xgb_model_onehot,
        "_tabular_log_reg_onehot": log_reg_onehot,
        "mlp_seq": mlp_seq_model,
        "attpool_transformer": attpool_transformer_model,
        "cls_transformer": cls_transformer_model,
        "mean_transformer": mean_transformer_model,
        "neurlux": neurlux,
        "cnn": cnn_model,
        "lstm": lstm_model,
        "cnn_lstm": cnn_lstm_model,
    }

    # =============================================
    # TRAINING MODELS
    # =============================================

    for name, model in models.items():
        if os.path.exists(os.path.join(LOGS_FOLDER, f"{name}_csv", "version_0", "checkpoints")):
            print(f"[!] Training of {name} already done, skipping...")
            continue

        now = time.time()
        print(f"[!] Training of {name} started: ", time.ctime())
        
        if name.startswith("_tabular"):
            x_train, x_test = None, None
            
            preprocessor = name.split("_")[-1]
            assert preprocessor in ["onehot", "minhash"]

            if preprocessor == "onehot":
                x_train = X_train_onehot
                x_test = X_test_onehot
            elif preprocessor == "minhash":
                x_train = X_train_minhash
                x_test = X_test_minhash
            
            if "_mlp_" in name:
                train_loader = create_dataloader(x_train, y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                test_loader = create_dataloader(x_test, y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                _ = train_lit_model(
                    train_loader,
                    test_loader,
                    model,
                    name,
                    log_folder=LOGS_FOLDER,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    scheduler_budget = EPOCHS * len(X_train_loader)
                )            
            else:
                training_tabular(
                    model,
                    name,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    logs_folder=LOGS_FOLDER
                )        
        else:
            _ = train_lit_model(
                X_train_loader,
                X_test_loader,
                model,
                name,
                log_folder=LOGS_FOLDER,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                scheduler=SCHEDULER,
                scheduler_budget= EPOCHS * len(X_train_loader)
            )
        
        print(f"[!] Training of {name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

    print(f"[!] Script end time: {time.ctime()}")
