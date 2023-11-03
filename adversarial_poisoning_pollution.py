# NOTE: CODE IS HORRIBLE -- BLOBS OF REPEATED SECTIONS, ETC.
# DONE AS PoC, FOR FAST PROTOTYPING, NOT FOR SYSTEMATIC REPRODUCIBILITY
import os
import time
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from watermark import watermark

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
from lightning.lite.utilities.seed import seed_everything
from src.models import CNN1DGroupedModel, MeanTransformerEncoder, SimpleMLP
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.lit_utils import load_lit_model, train_lit_model
from src.data_utils import commands_to_loader
from src.scoring import collect_scores


def load_data(seed):
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
        X_train_baseline_cmd = shuffle(train_baseline_df['cmd'].values.tolist(), random_state=seed)[:LIMIT//2]
        X_train_malicious_cmd = shuffle(train_malicious_df['cmd'].values.tolist(), random_state=seed)[:LIMIT//2]
        X_test_baseline_cmd = shuffle(test_baseline_df['cmd'].values.tolist(), random_state=seed)[:LIMIT//2]
        X_test_malicious_cmd = shuffle(test_malicious_df['cmd'].values.tolist(), random_state=seed)[:LIMIT//2]
    else:
        X_train_baseline_cmd = train_baseline_df['cmd'].values.tolist()
        X_train_malicious_cmd = train_malicious_df['cmd'].values.tolist()
        X_test_baseline_cmd = test_baseline_df['cmd'].values.tolist()
        X_test_malicious_cmd = test_malicious_df['cmd'].values.tolist()

    X_train_non_shuffled = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=seed)

    X_test_non_shuffled = X_test_baseline_cmd + X_test_malicious_cmd
    y_test = np.array([0] * len(X_test_baseline_cmd) + [1] * len(X_test_malicious_cmd), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=seed)

    return X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd


def load_tokenizer(tokenizer_type, cmd_train, suffix="", logs_folder="./"):
    if "onehot" in tokenizer_type:
        oh_file = os.path.join(logs_folder, f"onehot_vocab_{VOCAB_SIZE}{suffix}.pkl")
        if os.path.exists(oh_file):
            print(f"[!] Loading One-Hot encoder from '{oh_file}'...")
            with open(oh_file, "rb") as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
            print("[*] Fitting One-Hot encoder...")
            now = time.time()
            tokenizer.fit(cmd_train)
            print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s
            with open(oh_file, "wb") as f:
                pickle.dump(tokenizer, f)
    else:
        tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
        vocab_file = os.path.join(logs_folder, f"wordpunct_vocab_{VOCAB_SIZE}{suffix}.json")
        if os.path.exists(vocab_file):
                print(f"[!] Loading vocab from '{vocab_file}'...")
                tokenizer.load_vocab(vocab_file)
        else:
            print("[*] Building Tokenizer for Embedding vocab and encoding...")
            X_train_tokens_poisoned = tokenizer.tokenize(cmd_train)
            tokenizer.build_vocab(X_train_tokens_poisoned)
            tokenizer.dump_vocab(vocab_file)
    
    return tokenizer


MAX_LEN = 128
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 2048 # 256/512 if training transformer
DROPOUT = 0.5
TOKENIZER = wordpunct_tokenize

# TEST
# EPOCHS = 2
# LIMIT = 10000
# POISONING_RATIOS = [0, 0.1, 1]

# PROD
EPOCHS = 10
LIMIT = None
# percentage from baseline
POISONING_RATIOS = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3] 

DEVICE = "gpu"
LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_LOG_FOLDER = os.path.join(ROOT, f"logs_adversarial_poisoning_pollution")


def main(seed):
    subfolder = f"seed_{seed}_lim_{LIMIT}"
    log_folder = os.path.join(ROOT_LOG_FOLDER, subfolder)
    os.makedirs(log_folder, exist_ok=True)

    # ===========================================
    print(f"[!] Script start time: {time.ctime()}")
    print(watermark(packages="torch,lightning,sklearn", python=True))

    seed_everything(seed)

    # ============================================
    # LOADING DATA
    # ============================================

    X_train_cmd, y_train, X_test_cmd, y_test, X_train_malicious_cmd, X_train_baseline_cmd, _, _ = load_data(seed)
    print(f"[!] Sizes of train and test sets: {len(X_train_cmd)}, {len(X_test_cmd)}")

    # =============================================
    # DEFINING MODELS
    # =============================================

    cnn_model = CNN1DGroupedModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDED_DIM,
        num_channels=32,
        kernel_sizes=[2, 3, 4, 5],
        mlp_hidden_dims=[64, 32],
        output_dim=1,
        dropout=DROPOUT
    ) # 301 K params

    mlp_tab_model_onehot = SimpleMLP(
        input_dim=VOCAB_SIZE,
        output_dim=1,
        hidden_dim=[64, 32],
        dropout=DROPOUT
    ) # 264 K params

    mean_transformer_model = MeanTransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=EMBEDDED_DIM,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        mlp_hidden_dims=[64,32],
        output_dim=1
    ) # 335 K params

    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        # "mean_transformer": mean_transformer_model, # keeps blue-screenning my laptop on inference
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
        nr_of_poisoned_samples = int(X_train_baseline_nr * (poisoning_ratio/100))
        print(f"[*] Poisoning train set... Ratio: {poisoning_ratio:.3f}% | Poisoned samples: {nr_of_poisoned_samples}")

        # ===========================================
        # POISONING
        # ===========================================

        # 1. Take random samples from the malicious class w/o replacement
        X_train_malicious_cmd_sampled = np.random.choice(X_train_malicious_cmd, nr_of_poisoned_samples, replace=False).tolist()

        # 2. Create a new dataset with the sampled malicious samples and the baseline samples
        X_train_baseline_cmd_poisoned = X_train_baseline_cmd + X_train_malicious_cmd_sampled

        # 3. Create y labels for new dataset
        X_train_non_shuffled_poisoned = X_train_baseline_cmd_poisoned + X_train_malicious_cmd
        y_train_non_shuffled_poisoned = np.array([0] * len(X_train_baseline_cmd_poisoned) + [1] * len(X_train_malicious_cmd), dtype=np.int8)

        # 4. Shuffle the dataset
        X_train_cmd_poisoned, y_train_poisoned = shuffle(X_train_non_shuffled_poisoned, y_train_non_shuffled_poisoned, random_state=seed)

        # ===========================================
        for name, target_model_poisoned in target_models.items():
            # ===========================================
            # TRAINING
            # ===========================================
            print(f"[*] Working on '{name}' model poisoned training...")
            run_name = f"{name}_poison_samples_{nr_of_poisoned_samples}"

            scores_json_file = os.path.join(log_folder, f"poisoned_scores_{run_name}.json")
            if os.path.exists(scores_json_file):
                print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
                continue

            tokenizer = load_tokenizer(
                tokenizer_type=name,
                cmd_train=X_train_cmd_poisoned,
                suffix=f"_poisoned_samples_{nr_of_poisoned_samples}_ratio_{poisoning_ratio}",
                logs_folder=log_folder
            )
            Xy_train_loader_poisoned = commands_to_loader(
                X_train_cmd_poisoned,
                tokenizer,
                y=y_train_poisoned,
                batch_size=BATCH_SIZE,
                workers=DATALOADER_WORKERS
            )
            Xy_test_loader = commands_to_loader(
                X_test_cmd,
                tokenizer,
                y=y_test,
                batch_size=BATCH_SIZE,
                workers=DATALOADER_WORKERS
            )

            model_file_poisoned = os.path.join(log_folder, f"{run_name}.ckpt")
            if os.path.exists(model_file_poisoned):
                print(f"[!] Loading original model from '{model_file_poisoned}'")
                trainer_poisoned, lightning_model_poisoned = load_lit_model(
                    model_file_poisoned,
                    target_model_poisoned,
                    run_name,
                    log_folder,
                    EPOCHS,
                    DEVICE,
                    LIT_SANITY_STEPS
                )
            else:
                print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                now = time.time()
                trainer_poisoned, lightning_model_poisoned = train_lit_model(
                    Xy_train_loader_poisoned,
                    Xy_test_loader,
                    target_model_poisoned,
                    run_name,
                    log_folder=log_folder,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    scheduler_budget=EPOCHS * len(Xy_train_loader_poisoned),
                    model_file = model_file_poisoned,
                    device=DEVICE,
                    lit_sanity_steps=LIT_SANITY_STEPS
                )
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
                    run_name=run_name
            )
            with open(scores_json_file, "w") as f:
                json.dump(scores, f, indent=4)
    
    print(f"[!] Script end time: {time.ctime()}")


if __name__ == "__main__":
    seeds = [0, 33, 42]
    for seed in seeds:
        main(seed)
