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
from src.models import SimpleMLP, SimpleMLPWithEmbedding, PyTorchLightningModel
from src.lit_utils import LitProgressBar
from src.preprocessors import OneHotCustomVectorizer, CommandTokenizer
from src.data_utils import create_dataloader


def csr_training(X_train, y_train, X_test, y_test, name, vocab_size, epochs=10, workers=4, log_folder="logs/"):
    X_train_loader = create_dataloader(X_train, y_train, batch_size=BATCH_SIZE, workers=workers)
    X_test_loader = create_dataloader(X_test, y_test, batch_size=BATCH_SIZE, workers=workers)

    pytorch_model = SimpleMLP(input_dim=vocab_size, hidden_dim=HIDDEN, output_dim=1)
    training(pytorch_model, X_train_loader, X_test_loader, name, log_folder, epochs=epochs)


def embedding_training(X_train_loader, X_test_loader, name, positional, epochs=10, log_folder="logs/"):
    pytorch_model = SimpleMLPWithEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        output_dim=1,
        hidden_dim=HIDDEN,
        use_positional_encoding=positional,
        max_seq_len=MAX_LEN
    )
    training(pytorch_model, X_train_loader, X_test_loader, name, log_folder, epochs=epochs)


def training(pytorch_model, X_train_loader, X_test_loader, name, log_folder, epochs=10):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=1e-3)

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

VOCAB_SIZE = 1024
MAX_LEN = 128
HIDDEN = 32
BATCH_SIZE = 1024

EMBEDDING_DIM = 64
EPOCHS = 10

# TEST
# LIT_SANITY_STEPS = 0
# LIMIT = 15000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "logs_preprocessor_TEST"

# PROD
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = "logs_preprocessor"

RUNS = ['onehot', 'tfidf', 'minhash', 'embedded', 'embedded_positional']

# these fail with cuda errors -- might be default vocabs are too high, e.g. tf-idf gets 332893 features
# RUNS = ['tfidf_default', 'minhash_default']

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

    # ===========================================
    # ONE HOT
    # ===========================================
    if 'onehot' in RUNS:
        oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)

        print("[*] Fitting one-hot encoder...")
        X_train_onehot = oh.fit_transform(X_train_cmds)
        X_test_onehot = oh.transform(X_test_cmds)

        csr_training(X_train_onehot, y_train, X_test_onehot, y_test, name="onehot", vocab_size=VOCAB_SIZE, epochs=EPOCHS, workers=DATALOADER_WORKERS, log_folder=LOGS_FOLDER)

    # ===========================================
    # TF-IDF
    # ===========================================
    if 'tfidf' in RUNS:
        tfidf = TfidfVectorizer(max_features=VOCAB_SIZE, tokenizer=TOKENIZER)

        print("[*] Fitting TF-IDF encoder...")
        X_train_tfidf = tfidf.fit_transform(X_train_cmds)
        X_test_tfidf = tfidf.transform(X_test_cmds)

        csr_training(X_train_tfidf, y_train, X_test_tfidf, y_test, name="tfidf", vocab_size=VOCAB_SIZE, epochs=EPOCHS, workers=DATALOADER_WORKERS, log_folder=LOGS_FOLDER)

    if 'tfidf_default' in RUNS:
        tfidf = TfidfVectorizer(tokenizer=TOKENIZER)

        print("[*] Fitting TF-IDF encoder...")
        X_train_tfidf_novocab = tfidf.fit_transform(X_train_cmds)
        X_test_tfidf_novocab = tfidf.transform(X_test_cmds)

        # get tfidf default vocab size
        vocab_size_tfidf = len(tfidf.get_feature_names())
        print(f"[!] Default TF-IDF vocab size: ", vocab_size_tfidf)

        csr_training(X_train_tfidf_novocab, y_train, X_test_tfidf_novocab, y_test, name="tfidf_default", vocab_size=vocab_size_tfidf, epochs=EPOCHS, workers=DATALOADER_WORKERS, log_folder=LOGS_FOLDER)

    # ===========================================
    # MIN HASH
    # ===========================================
    if 'minhash' in RUNS:
        minhash = HashingVectorizer(n_features=VOCAB_SIZE, tokenizer=TOKENIZER)

        print("[*] Fitting MinHash encoder...")
        X_train_minhash = minhash.fit_transform(X_train_cmds)
        X_test_minhash = minhash.transform(X_test_cmds)

        csr_training(X_train_minhash, y_train, X_test_minhash, y_test, name="minhash", vocab_size=VOCAB_SIZE, epochs=EPOCHS, workers=DATALOADER_WORKERS, log_folder=LOGS_FOLDER)

    if 'minhash_default' in RUNS:
        minhash = HashingVectorizer(tokenizer=TOKENIZER)

        print("[*] Fitting MinHash encoder...")
        X_train_minhash_novocab = minhash.fit_transform(X_train_cmds)
        X_test_minhash_novocab = minhash.transform(X_test_cmds)

        # get minhash default vocab size
        vocab_size_minhash = len(minhash.get_feature_names())
        print(f"[!] Default MinHash vocab size: ", vocab_size_minhash)

        csr_training(X_train_minhash_novocab, y_train, X_test_minhash_novocab, y_test, name="minhash_default", vocab_size=vocab_size_minhash, epochs=EPOCHS, workers=DATALOADER_WORKERS, log_folder=LOGS_FOLDER)

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
    if 'embedded' in RUNS:
        embedding_training(X_train_loader, X_test_loader, name="embedded", positional=False, log_folder=LOGS_FOLDER, epochs=EPOCHS)

    # ===========================================
    # EMBEDDED + POSITIONAL ENCODINGS
    # ===========================================
    if 'embedded_positional' in RUNS:
        embedding_training(X_train_loader, X_test_loader, name="embedded_positional", positional=True, log_folder=LOGS_FOLDER, epochs=EPOCHS)

    print(f"[!] Script end time: {time.ctime()}")
