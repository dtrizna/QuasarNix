import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from watermark import watermark

# encoders
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.lite.utilities.seed import seed_everything

sys.path.append("Linux/")
from src.models import SimpleMLPWithEmbedding, MLP_LightningModel
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer
from src.data_utils import create_dataloader


def embedding_training(X_train_loader, X_test_loader, name, embedding_dim, positional=False, epochs=10, log_folder="logs/"):
    pytorch_model = SimpleMLPWithEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=embedding_dim,
        output_dim=1,
        hidden_dim=HIDDEN,
        use_positional_encoding=positional,
        max_seq_len=MAX_LEN
    )
    training(pytorch_model, X_train_loader, X_test_loader, name, log_folder, epochs=epochs)


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
MAX_LEN = 128
HIDDEN = 32
BATCH_SIZE = 2048
EPOCHS = 10

# TEST
# LIT_SANITY_STEPS = 0
# LIMIT = 15000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "logs_dropout_TEST"

# PROD
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = "logs_dropout"

EMBEDDING_DIM = 64
#DROPOUTS = [0, 0.1, 0.3, 0.5, 0.7]
DROPOUTS = [0.7]

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
    # PREPING DATA FOR EMBEDDING ANALYSIS
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

    # ===========================================
    # EMBEDDED
    # ===========================================
    for dropout in DROPOUTS:
        embedding_training(X_train_loader, X_test_loader, name=f"embedded_{dropout}", embedding_dim=EMBEDDING_DIM, positional=False, log_folder=LOGS_FOLDER, epochs=EPOCHS)

    print(f"[!] Script end time: {time.ctime()}")
