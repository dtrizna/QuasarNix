import os
import time
import json
import pickle
import numpy as np
from sklearn.utils import shuffle
from watermark import watermark
from typing import List, Union
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(ROOT)

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
from lightning.fabric.utilities.seed import seed_everything
from src.models import CNN1DGroupedModel, SimpleMLP
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from xgboost import XGBClassifier
from src.lit_utils import load_lit_model, LitTrainerWrapper
from src.tabular_utils import training_tabular
from src.data_utils import commands_to_loader, load_tokenizer, read_powershell_data
from src.scoring import collect_scores
from src.models import CLSTransformerEncoder


# ==================
# BACKDOOR CODE
# ==================

def create_backdoor(
        X_train_cmd: List,
        tokenizer: Union[OneHotCustomVectorizer, CommandTokenizer],
        backdoor_size: int,
        max_chars: int = 128
) -> str:
    """
    Create a backdoor from a sparsely populated region of the dataset. Since our training data is discrete, 
    the heuristical shortcut for sparse region is to use the least frequent tokens. Other options possible.

    Parameters:
    - X_train_cmd: The collection of commands in training dataset.
    - tokenizer: The tokenizer that converts commands to tokens.
    - backdoor_size: The size of the backdoor (number of tokens).

    Returns:
    - backdoor: The backdoor pattern represented as a string of max_chars length.
    """
    tokenized_sequences = tokenizer.tokenize(X_train_cmd)
    vocab = tokenizer.vocab

    all_tokens = []
    for sequence in tokenized_sequences:
        for token in sequence:
            if token in vocab.keys():
                all_tokens.append(token)

    token_counts = Counter(all_tokens)
    least_frequent_tokens = token_counts.most_common()[:-backdoor_size-1:-1]
    backdoor_tokens = [token for token, _ in least_frequent_tokens]
    backdoor = " ".join(backdoor_tokens)
    backdoor = backdoor.replace('"', '\\"') # escape double quotes
    backdoor = backdoor.replace("'", "\\'") # escape single quotes
    return backdoor[:max_chars]


def backdoor_command(backdoor: Union[str, List], command: str = None, template: str = None) -> str:
    if template is None:
        template = """python3 -c "print('PAYLOAD')" """
        # valid option #2 but with more character overhead: 
        # """awk 'BEGIN { print ARGV[1] }' "PAYLOAD" """
    else:
        assert "PAYLOAD" in template, "Please provide a template with 'PAYLOAD' placeholder."

    if isinstance(backdoor, list):
        backdoor = ";".join(backdoor)
    assert isinstance(backdoor, str), "Wrong type for backdoor. Please provide a string or a list of strings."
    payload = template.replace("PAYLOAD", backdoor)

    if command is None:
        return payload
    else:
        return payload + ";" + command

# ==================
# RUN CONFIG
# ==================

MAX_LEN = 128
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 1024 # 256/512 if training transformer
DROPOUT = 0.5
TOKENIZER = wordpunct_tokenize

# TEST
# EPOCHS = 2
# LIMIT = 1000
# TEST_SET_SUBSAMPLE = 50
# POISONING_RATIOS = [0, 0.01, 0.1]
# BACKDOOR_TOKENS = [2, 32]
# LIT_SANITY_STEPS = 0
# DATALOADER_WORKERS = 1
# DEVICE = "cpu"

# PROD
EPOCHS = 30
LIMIT = 10000
TEST_SET_SUBSAMPLE = 5000
# run to loop over token
POISONING_RATIOS = [0, 0.1, 0.3, 1, 3, 10] # percentage of baseline samples to poison
BACKDOOR_TOKENS = [2, 4, 8, 16, 32]

LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
DEVICE = "gpu"

EARLY_STOP_PATIENCE = 30
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"
MONITOR_METRIC = "val_f1"

NIX_ROOT = os.path.join(ROOT, "..", "..", "..") # /home/me/QuasarNix
ROOT_LOG_FOLDER = os.path.join(ROOT, "experiments", "logs_adversarial_pwsh_poisoning_backdoor_v2")

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

    backdoor_template = """Write-Output "PAYLOAD" """

    X_train_cmd, X_test_cmd, y_train, y_test = read_powershell_data(NIX_ROOT, seed, limit=LIMIT)
    print(f"Sizes of train and test sets: {len(X_train_cmd)}, {len(X_test_cmd)}")
    
    print(f"[*] Splitting train set into baseline and malicious...")
    now = time.time()
    train_baseline_cmd_mask = y_train == 0
    X_train_baseline_cmd = [cmd for cmd, mask in zip(X_train_cmd, train_baseline_cmd_mask) if mask]
    X_train_malicious_cmd = [cmd for cmd, mask in zip(X_train_cmd, ~train_baseline_cmd_mask) if mask]

    # to 10% of malicious train set set, add template without payload, to account for template in backdoor_command
    X_train_malicious_with_template = np.random.choice(X_train_malicious_cmd, int(len(X_train_malicious_cmd)*0.1), replace=False).tolist()
    X_train_malicious_with_template = [backdoor_command("", cmd, backdoor_template) for cmd in X_train_malicious_with_template]
    X_train_malicious_cmd = X_train_malicious_cmd + X_train_malicious_with_template
    X_train_cmd = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.concatenate([np.zeros(len(X_train_baseline_cmd), dtype=np.int8), np.ones(len(X_train_malicious_cmd), dtype=np.int8)])
    
    test_malicious_cmd_mask = y_test == 1
    X_test_malicious_cmd = [cmd for cmd, mask in zip(X_test_cmd, test_malicious_cmd_mask) if mask]
    print(f"[!] Splitting took: {time.time() - now:.2f} seconds")

    print(f"[!] Sizes of baseline and malicious train sets: {len(X_train_baseline_cmd)}, {len(X_train_malicious_cmd)}")
    print(f"[!] Sizes of malicious test set: {len(X_test_malicious_cmd)}")
    baseline = X_train_baseline_cmd # in case of powershell we consider the same baseline for backdoor as in train set
    
    # randomly subsample of backdoor evasive performance attack
    sample_file = os.path.join(log_folder, f"test_set_subsample_non_poisoned_{TEST_SET_SUBSAMPLE}.json")
    if os.path.exists(sample_file):
        print(f"[*] Loading malicious test set sample from '{sample_file}'")
        with open(sample_file, "r", encoding="utf-8") as f:
            X_test_malicious_without_attack_cmd = json.load(f)
        print(f"[!] Size of malicious test set: {len(X_test_malicious_without_attack_cmd)}")
    else:
        print(f"[*] Subsampling malicious test set to {TEST_SET_SUBSAMPLE} samples...")
        X_test_malicious_without_attack_cmd = np.random.choice(X_test_malicious_cmd, TEST_SET_SUBSAMPLE, replace=True).tolist()

        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_without_attack_cmd, f, indent=4)
    
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

    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=seed)
    
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

    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        "xgb_onehot": xgb_model_onehot,
        "cls_transformer": cls_transformer_model
    }

    X_train_baseline_nr = len(X_train_baseline_cmd)
    for poisoning_ratio in POISONING_RATIOS:
        nr_of_poisoned_samples = int(X_train_baseline_nr * (poisoning_ratio/100))
        print(f"[*] Run for Poisoning Ratio: {poisoning_ratio:.3f}% | Poisoned samples: {nr_of_poisoned_samples}")

        # =======================================
        # BACKDOORING TRAINING SET FOR EACH MODEL
        # ======================================

        for nr_of_backdoor_tokens in BACKDOOR_TOKENS:
            if poisoning_ratio == 0 and nr_of_backdoor_tokens != BACKDOOR_TOKENS[0]:
                continue # do this only once for non-polluted dataset to get score baseline
            for name, model in target_models.items():
                print(f"[*] Poisoning train set of '{name}' model | Backdoor tokens: {nr_of_backdoor_tokens}")
                run_name = f"{name}_train_set_poison_samples_{nr_of_poisoned_samples}"
                run_name = run_name if poisoning_ratio == 0 else run_name + f"_backdoor_tokens_{nr_of_backdoor_tokens}"

                poisoned_sample_file = os.path.join(log_folder, f"test_set_subsample_poisoned_{TEST_SET_SUBSAMPLE}_{run_name}.json")
                scores_json_file = os.path.join(log_folder, f"poisoned_scores_{run_name}.json")
                
                if os.path.exists(scores_json_file):
                    print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
                    continue

                tokenizer = load_tokenizer(
                    tokenizer_type=name,
                    train_cmds=X_train_cmd,
                    vocab_size=VOCAB_SIZE,
                    max_len=MAX_LEN,
                    tokenizer_fn=TOKENIZER,
                    suffix=f"_train_set_poison_samples_{nr_of_poisoned_samples}_ratio_{poisoning_ratio}",
                    logs_folder=log_folder
                )
                
                # TRAINING SET -- single backdoor, placed into baseline 'nr_of_poisoned_samples' times
                # ATTACKER DOESN'T KNOW TRUE BASELINE, CAN INFER FROM MALICIOUS TRAIN SET + PUBLIC SOURCES
                backdoor_cmd_space = X_train_malicious_cmd + baseline
                backdoor = create_backdoor(backdoor_cmd_space, tokenizer, nr_of_backdoor_tokens)
                backdoor_cmd = backdoor_command(backdoor, template=backdoor_template)
                print(f"[DBG] Backdoor cmd: ", backdoor_cmd)

                X_train_cmd_backdoor = [backdoor_cmd] * nr_of_poisoned_samples
                y_train_backdoor = np.zeros(nr_of_poisoned_samples, dtype=np.int8)

                X_train_cmd_poisoned = X_train_cmd_backdoor + X_train_cmd
                y_train_poisoned = np.concatenate([y_train_backdoor, y_train])
                X_train_cmd_poisoned, y_train_poisoned = shuffle(X_train_cmd_poisoned, y_train_poisoned, random_state=seed)

                Xy_train_loader_poisoned = commands_to_loader(
                    X_train_cmd_poisoned,
                    tokenizer,
                    y=y_train_poisoned,
                    batch_size=BATCH_SIZE,
                    workers=DATALOADER_WORKERS
                )
                Xy_test_loader_orig_full = commands_to_loader(
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
                        X_test_orig_full = tokenizer.transform(X_test_cmd)
                        trainer_poisoned = None
                    else:
                        trainer_poisoned, model_poisoned = load_lit_model(
                            model_file_poisoned,
                            model,
                            run_name,
                            log_folder,
                            EPOCHS,
                            DEVICE,
                            LIT_SANITY_STEPS)
                        X_test_orig_full = Xy_test_loader_orig_full
                else:
                    print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                    now = time.time()
                    if "xgb" in name:
                        X_train_cmd_poisoned_onehot = tokenizer.transform(X_train_cmd_poisoned)
                        X_test_orig_full = tokenizer.transform(X_test_cmd)
                        trainer_poisoned = None
                        model_poisoned = training_tabular(
                            model=model,
                            name=run_name,
                            X_train_encoded=X_train_cmd_poisoned_onehot,
                            X_test_encoded=X_test_orig_full,
                            y_train=y_train_poisoned,
                            y_test=y_test,
                            logs_folder=log_folder,
                            model_file = model_file_poisoned
                        )
                    else:
                        lit_trainer = LitTrainerWrapper(
                            pytorch_model=model,
                            name=run_name,
                            log_folder=log_folder,
                            epochs=EPOCHS,
                            learning_rate=LEARNING_RATE,
                            scheduler=SCHEDULER,
                            device=DEVICE,
                            lit_sanity_steps=LIT_SANITY_STEPS,
                            early_stop_patience=EARLY_STOP_PATIENCE,
                            monitor_metric=MONITOR_METRIC,
                            verbose=True
                        )
                        lit_trainer.train_lit_model(Xy_train_loader_poisoned, Xy_test_loader_orig_full)
                        lit_trainer.save_lit_model(model_file_poisoned)

                        model_poisoned = lit_trainer.lit_model
                        trainer_poisoned = lit_trainer.trainer
                        
                        X_test_orig_full = Xy_test_loader_orig_full
                    print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

                # ===========================================
                # TESTING
                # ===========================================
                print(f"[*] Testing '{run_name}' model...")

                # TEST SETS -- use X_test_malicious_without_attack_cmd, backdoor them and check how many are evasive
                X_test_malicious_cmd_poisoned = [backdoor_command(backdoor, cmd, template=backdoor_template) for cmd in X_test_malicious_without_attack_cmd]
                with open(poisoned_sample_file, "w", encoding="utf-8") as f:
                    json.dump(X_test_malicious_cmd_poisoned, f, indent=4)
                y_test_malicious_poisoned = np.ones(len(X_test_malicious_cmd_poisoned), dtype=np.int8)

                if "xgb" in name:
                    X_test_poisoned = tokenizer.transform(X_test_malicious_cmd_poisoned)
                else:
                    X_test_poisoned = commands_to_loader(
                        X_test_malicious_cmd_poisoned,
                        tokenizer,
                        y=y_test_malicious_poisoned,
                        batch_size=BATCH_SIZE,
                        workers=DATALOADER_WORKERS
                    )
                scores = collect_scores(
                    X_test_poisoned,
                    y_test_malicious_poisoned,
                    model_poisoned,
                    trainer_poisoned,
                    scores=None,
                    score_suffix="_backdoor",
                    run_name=run_name
                )

                # collect scores on orig test -- subset of malicious test set without backdoor
                if "xgb" in name:
                    X_test_orig = tokenizer.transform(X_test_malicious_without_attack_cmd)
                else:
                    X_test_orig = commands_to_loader(
                        X_test_malicious_without_attack_cmd,
                        tokenizer,
                        y=y_test_malicious_poisoned,
                        batch_size=BATCH_SIZE,
                        workers=DATALOADER_WORKERS
                    )
                scores = collect_scores(
                    X_test_orig,
                    y_test_malicious_poisoned,
                    model_poisoned,
                    trainer_poisoned,
                    scores=scores,
                    score_suffix="_orig",
                    run_name=run_name
                )
                
                # collect scores on orig test set too -- full test set
                scores = collect_scores(
                    X_test_orig_full,
                    y_test,
                    model_poisoned,
                    trainer_poisoned,
                    scores=scores,
                    score_suffix="_orig_full",
                    run_name=run_name
                )

                # dump
                with open(scores_json_file, "w") as f:
                    json.dump(scores, f, indent=4)

if __name__ == "__main__":
    seeds = [0, 33]#, 42]
    for seed in seeds:
        main(seed)
