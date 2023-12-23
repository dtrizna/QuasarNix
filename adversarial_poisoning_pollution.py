import os
import time
import json
import pickle
import numpy as np
from sklearn.utils import shuffle
from watermark import watermark

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
from lightning.lite.utilities.seed import seed_everything
from src.models import CNN1DGroupedModel, SimpleMLP
from xgboost import XGBClassifier
from src.lit_utils import load_lit_model, train_lit_model
from src.data_utils import commands_to_loader, load_data, load_tokenizer
from src.tabular_utils import training_tabular
from src.scoring import collect_scores
from src.models import CLSTransformerEncoder


MAX_LEN = 128
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 1024 # 256/512 if training transformer
DROPOUT = 0.5
TOKENIZER = wordpunct_tokenize

# # TEST RUN CONFIG
# EPOCHS = 2
# LIMIT = 10000
# POISONING_RATIOS = [0, 0.1, 1]

# PROD RUN CONFIG
EPOCHS = 10
LIMIT = None
# NOTE: this is percentage from baseline, 
# e.g. 0.001 is 0.001% i.e. 0.00001 * len(X_train)
POISONING_RATIOS = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3] 

DEVICE = "gpu"
LIT_SANITY_STEPS = 1
EARLY_STOP_PATIENCE = 10
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

    _, _, X_test_cmd, y_test, X_train_malicious_cmd, X_train_baseline_cmd, _, _ = load_data(ROOT, seed, limit=LIMIT)
    
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

    cls_transformer_model = CLSTransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=EMBEDDED_DIM,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        mlp_hidden_dims=[64,32],
        output_dim=1
    ) #  335 K params

    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=seed)

    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        "xgb_onehot": xgb_model_onehot,
        "cls_transformer": cls_transformer_model
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
        nr_of_poisoned_samples = int(X_train_baseline_nr * (poisoning_ratio / 100))
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

        print(f"[!] Sizes of train and test sets: {len(X_train_cmd_poisoned)}, {len(X_test_cmd)}")

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
                train_cmds=X_train_cmd_poisoned,
                vocab_size=VOCAB_SIZE,
                max_len=MAX_LEN,
                tokenizer_fn=TOKENIZER,
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
                if "xgb" in name:
                    with open(model_file_poisoned, "rb") as f:
                        model_poisoned = pickle.load(f)
                    X_test = tokenizer.transform(X_test_cmd)
                    trainer_poisoned = None
                else:
                    trainer_poisoned, model_poisoned = load_lit_model(
                        model_file_poisoned,
                        target_model_poisoned,
                        run_name,
                        log_folder,
                        EPOCHS,
                        DEVICE,
                        LIT_SANITY_STEPS
                    )
                    X_test = Xy_test_loader
            else:
                print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                now = time.time()
                if "xgb" in name:
                    X_train_cmd_poisoned_onehot = tokenizer.transform(X_train_cmd_poisoned)
                    X_test = tokenizer.transform(X_test_cmd)
                    trainer_poisoned = None
                    model_poisoned = training_tabular(
                        model=target_model_poisoned,
                        name=run_name,
                        X_train_encoded=X_train_cmd_poisoned_onehot,
                        X_test_encoded=X_test,
                        y_train=y_train_poisoned,
                        y_test=y_test,
                        logs_folder=log_folder,
                        model_file = model_file_poisoned
                    )
                else:
                    X_test = Xy_test_loader
                    trainer_poisoned, model_poisoned = train_lit_model(
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
                        lit_sanity_steps=LIT_SANITY_STEPS,
                        early_stop_patience=EARLY_STOP_PATIENCE
                    )
                print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

            # ===========================================
            # TESTING
            # ===========================================
            print(f"[*] Testing '{run_name}' model...")
            scores = collect_scores(
                    X_test,
                    y_test,
                    model_poisoned,
                    trainer_poisoned,
                    run_name=run_name
            )
            with open(scores_json_file, "w") as f:
                json.dump(scores, f, indent=4)
    
    print(f"[!] Script end time: {time.ctime()}")


if __name__ == "__main__":
    seeds = [0, 33, 42]
    for seed in seeds:
        main(seed)
