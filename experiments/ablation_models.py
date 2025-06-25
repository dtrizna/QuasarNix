import os
import pickle
import time
import sys
from pathlib import Path
from watermark import watermark

from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.append(str(ROOT))

from src.models import *
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import create_dataloader, commands_to_loader, load_data
from src.lit_utils import train_lit_model
from src.tabular_utils import training_tabular
from lightning.fabric.utilities.seed import seed_everything

SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

# # TEST RUN CONFIG
#DEVICE = "cpu"
# EPOCHS = 1
# LIT_SANITY_STEPS = 0
# LIMIT = 5000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "TEST_logs_models"

# PROD RUN CONFIG
DEVICE = "gpu"
EPOCHS = 20
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = Path("experiments/logs_models_ACM_TOPS_v3")
os.makedirs(LOGS_FOLDER, exist_ok=True)


if __name__ == "__main__":
    # ===========================================
    print(watermark(packages="torch,lightning,sklearn", python=True))
    print(f"[!] Script start time: {time.ctime()}")

    TOKENIZER = wordpunct_tokenize
    seed_everything(SEED)

    # ===========================================
    # LOADING DATA
    # ===========================================
    ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
    X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(root=ROOT, seed=SEED, limit=LIMIT)
    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")

    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    # ========== TOKENIZING ==========
    vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}.json")
    if os.path.exists(vocab_file):
        tokenizer.load_vocab(vocab_file)
        print(f"[!] Loaded vocab from {vocab_file}")
    else:
        print("[*] Building vocab and encoding...")
        X_train_tokens = tokenizer.tokenize(X_train_cmds)
        tokenizer.build_vocab(X_train_tokens)
        tokenizer.dump_vocab(vocab_file)
        print(f"Vocab size: {len(tokenizer.vocab)}")

    # creating dataloaders
    print("[*] Creating dataloaders...")
    X_train_loader_path = os.path.join(LOGS_FOLDER, f"X_train_loader_{VOCAB_SIZE}.pt")
    X_test_loader_path = os.path.join(LOGS_FOLDER, f"X_test_loader_{VOCAB_SIZE}.pt")
    if os.path.exists(X_train_loader_path) and os.path.exists(X_test_loader_path):
        X_train_loader = torch.load(X_train_loader_path, weights_only=False)
        X_test_loader = torch.load(X_test_loader_path, weights_only=False)
        print(f"[!] Loaded dataloaders from {X_train_loader_path} and {X_test_loader_path}")
    else:
        X_train_loader = commands_to_loader(X_train_cmds, tokenizer, y=y_train, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE, )
        torch.save(X_train_loader, X_train_loader_path)
        X_test_loader = commands_to_loader(X_test_cmds, tokenizer, y=y_test, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE)
        torch.save(X_test_loader, X_test_loader_path)

    # ========== MIN-HASH TABULAR ENCODING ==========
    # minhash = HashingVectorizer(n_features=VOCAB_SIZE, tokenizer=TOKENIZER, token_pattern=None)
    # print("[*] Fitting MinHash encoder...")
    # X_train_minhash = minhash.fit_transform(X_train_cmds)
    # X_test_minhash = minhash.transform(X_test_cmds)

    # ========== ONE-HOT TABULAR ENCODING ===========
    oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)

    X_train_onehot_path = os.path.join(LOGS_FOLDER, f"X_train_onehot_{VOCAB_SIZE}.pkl")
    X_test_onehot_path = os.path.join(LOGS_FOLDER, f"X_test_onehot_{VOCAB_SIZE}.pkl")
    if os.path.exists(X_train_onehot_path) and os.path.exists(X_test_onehot_path):
        with open(X_train_onehot_path, "rb") as f:
            X_train_onehot = pickle.load(f)
        with open(X_test_onehot_path, "rb") as f:
            X_test_onehot = pickle.load(f)
        print(f"[!] Loaded One-Hot encoded data from {X_train_onehot_path} and {X_test_onehot_path}")
    else:
        print("[*] Fitting One-Hot encoder...")
        X_train_onehot = oh.fit_transform(X_train_cmds)
        X_test_onehot = oh.transform(X_test_cmds)
        with open(X_train_onehot_path, "wb") as f:
            pickle.dump(X_train_onehot, f)
        with open(X_test_onehot_path, "wb") as f:
            pickle.dump(X_test_onehot, f)

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
        # "_tabular_mlp_minhash": mlp_tab_model_minhash,
        # "_tabular_rf_minhash": rf_model_minhash,
        # "_tabular_xgb_minhash": xgb_model_minhash,
        # "_tabular_log_reg_minhash": log_reg_minhash,
        "_tabular_mlp_onehot": mlp_tab_model_onehot,
        "_tabular_rf_onehot": rf_model_onehot,
        "_tabular_xgb_onehot": xgb_model_onehot,
        # "_tabular_log_reg_onehot": log_reg_onehot,
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
            # elif preprocessor == "minhash":
            #     x_train = X_train_minhash
            #     x_test = X_test_minhash
            
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
                    scheduler_budget = EPOCHS * len(train_loader)
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
