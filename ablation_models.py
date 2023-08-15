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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.lite.utilities.seed import seed_everything

# import random forest and xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append("Linux/")
from src.models import *
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer
from src.data_utils import create_dataloader


def get_tpr_at_fpr(predicted_logits, true_labels, fprNeeded=1e-4):
    predicted_probs = torch.sigmoid(predicted_logits).cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    if all(np.isnan(fpr)):
        return np.nan#, np.nan
    else:
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return tpr_at_fpr#, threshold_at_fpr


def training(pytorch_model, X_train_loader, X_test_loader, name, log_folder, epochs=10):
    lightning_model = MLP_LightningModel(model=pytorch_model, learning_rate=1e-3)

    # ensure folders for logging exist
    os.makedirs(f"{log_folder}/{name}_csv", exist_ok=True)
    os.makedirs(f"{log_folder}/{name}_tb", exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_f1",
        patience=3,
        min_delta=0.001,
        verbose=True,
        mode="max"
    )

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[
            LitProgressBar(),
            #early_stop,
        ],
        val_check_interval=0.2, # log validation scores five times per epoch
        log_every_n_steps=10,
        logger=[
            CSVLogger(save_dir=log_folder, name=f"{name}_csv"),
            TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb"),
        ]
    )

    print(f"[*] Training {name} model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)
    # trainer.test(lightning_model, X_test_loader)

SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 2048
DROPOUT = 0.5

# TEST
EPOCHS = 2
LIT_SANITY_STEPS = 0
LIMIT = 15000
DATALOADER_WORKERS = 1
LOGS_FOLDER = "logs_models_TEST"

# PROD
# EPOCHS = 10
# LIT_SANITY_STEPS = 1
# LIMIT = None
# DATALOADER_WORKERS = 4
# LOGS_FOLDER = "logs_models"


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

    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE)

    # Tokenize
    X_train_tokens = tokenizer.tokenize(X_train_cmds)
    X_test_tokens = tokenizer.tokenize(X_test_cmds)

    # Build vocab and encode
    print("[*] Building vocab and encoding...")
    tokenizer.build_vocab(X_train_tokens)
    X_train_ints = tokenizer.encode(X_train_tokens)
    X_test_ints = tokenizer.encode(X_test_tokens)

    # Pad sequences
    X_train_padded = tokenizer.pad(X_train_ints, MAX_LEN)
    X_test_padded = tokenizer.pad(X_test_ints, MAX_LEN)

    # creating dataloaders
    X_train_loader = create_dataloader(X_train_padded, y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    X_test_loader = create_dataloader(X_test_padded, y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

    # MIN-HASH TABULAR ENCODING
    minhash = HashingVectorizer(n_features=VOCAB_SIZE, tokenizer=TOKENIZER)
    print("[*] Fitting MinHash encoder...")
    X_train_minhash = minhash.fit_transform(X_train_cmds)
    X_test_minhash = minhash.transform(X_test_cmds)

    X_train_loader_minhash = create_dataloader(X_train_minhash, y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
    X_test_loader_minhash = create_dataloader(X_test_minhash, y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

    # =============================================
    # DEFINING MODELS
    # =============================================

    # sequential models
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[3, 5, 7], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 299 328 params
    lstm_model = BiLSTMModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 334 656 params
    cnn_lstm_model = CNN1D_BiLSTM_Model(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_size=3, lstm_hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 324 416 params
    mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_seq_len=MAX_LEN, dropout=DROPOUT) # 297 345 params
    transformer_encoder_model = TransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, mlp_hidden_dims=[64,32], max_len=MAX_LEN, dropout=DROPOUT) # 333 953 params
    
    # tabular models
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    xgb_model = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    mlp_tab_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_seq_len=MAX_LEN, dropout=DROPOUT) # 297 345 params

    models = {
        "cnn": cnn_model,
        "lstm": lstm_model,
        "cnn_lstm": cnn_lstm_model,
        "mlp_seq": mlp_seq_model,
        "transformer": transformer_encoder_model,
        "mlp_tab": mlp_tab_model,
        "rf": rf_model,
        "xgb": xgb_model,
    }

    # =============================================
    # TRAINING MODELS
    # =============================================

    for name, model in models.items():
        if name in ["rf", "xgb"]:
            print(f"[*] Training {name} model...")
            model.fit(X_train_minhash, y_train)

            # save trained model to LOGS_FOLDER/name
            os.makedirs(f"{LOGS_FOLDER}/{name}", exist_ok=True)
            with open(f"{LOGS_FOLDER}/{name}/model.pkl", "wb") as f:
                pickle.dump(model, f)
            
            y_test_preds = model.predict_proba(X_test_minhash)[:,1]
            tpr = get_tpr_at_fpr(y_test_preds, y_test)
            f1 = f1_score(y_test, y_test_preds.round())
            acc = accuracy_score(y_test, y_test_preds.round())
            auc = roc_auc_score(y_test, y_test_preds)
            print(f"[!] {name} model scores: tpr={tpr:.4f}, f1={f1:.4f}, acc={acc:.4f}, auc={auc:.4f}")
        
        elif name == "mlp_tab":
            training(model, X_train_loader_minhash, X_test_loader_minhash, name, log_folder=LOGS_FOLDER, epochs=EPOCHS)

        else:
            training(model, X_train_loader, X_test_loader, name, log_folder=LOGS_FOLDER, epochs=EPOCHS)

    print(f"[!] Script end time: {time.ctime()}")
