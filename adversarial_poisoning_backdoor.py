import os
import time
import json
import numpy as np
from sklearn.utils import shuffle
from watermark import watermark
from typing import List, Union
from collections import Counter

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
from lightning.lite.utilities.seed import seed_everything
from src.models import CNN1DGroupedModel, SimpleMLP
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from xgboost import XGBClassifier
from src.lit_utils import load_lit_model, train_lit_model
from src.data_utils import commands_to_loader, load_data, load_nl2bash, load_tokenizer
from src.scoring import collect_scores


# ==================
# BACKDOOR CODE
# ==================

def create_backdoor(X_train_cmd: List, tokenizer: Union[OneHotCustomVectorizer, CommandTokenizer], backdoor_size: int, max_chars: int = 128) -> List:
    """
    Create a backdoor from a sparsely populated region of the dataset using Kernel Density Estimation.

    Parameters:
    - X_train_cmd: The collection of commands in training dataset.
    - tokenizer: The tokenizer that converts commands to tokens.
    - backdoor_size: The size of the backdoor (number of tokens).

    Returns:
    - backdoor: list | The backdoor pattern represented as a list of tokens.
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
    return backdoor[:max_chars]


def backdoor_command(backdoor: Union[str, List], command: str = None, template: str = None) -> str:
    if template is None:
        # option #2: """awk 'BEGIN { print ARGV[1] }' "PAYLOAD" """
        template = """python3 -c "print('PAYLOAD')" """
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


MAX_LEN = 256
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 2048 # 256/512 if training transformer
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
EPOCHS = 10
LIMIT = 100000
TEST_SET_SUBSAMPLE = 5000
POISONING_RATIOS = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
BACKDOOR_TOKENS = [2, 4, 8, 16, 32]
LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
DEVICE = "gpu"

LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_LOG_FOLDER = os.path.join(ROOT, f"logs_adversarial_poisoning_backdoor")


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

    baseline = load_nl2bash(ROOT)
    X_train_cmd, y_train, X_test_cmd, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, _ = load_data(ROOT, seed, limit=LIMIT)
    print(f"Sizes of train and test sets: {len(X_train_cmd)}, {len(X_test_cmd)}")

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
    
    target_models = {
        "cnn": cnn_model,
        "mlp_onehot": mlp_tab_model_onehot,
        "xgb_onehot": xgb_model_onehot,
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
                backdoor_cmd = backdoor_command(backdoor)
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
                    trainer_poisoned, lightning_model_poisoned = load_lit_model(
                        model_file_poisoned,
                        model,
                        run_name,
                        log_folder,
                        EPOCHS,
                        DEVICE,
                        LIT_SANITY_STEPS)
                else:
                    print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                    now = time.time()
                    trainer_poisoned, lightning_model_poisoned = train_lit_model(
                        Xy_train_loader_poisoned,
                        Xy_test_loader_orig_full,
                        model,
                        run_name,
                        log_folder=log_folder,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        scheduler=SCHEDULER,
                        scheduler_budget=EPOCHS * len(Xy_train_loader_poisoned),
                        model_file = model_file_poisoned,
                        device=DEVICE,
                        lit_sanity_steps=LIT_SANITY_STEPS)
                    print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

                # ===========================================
                # TESTING
                # ===========================================
                print(f"[*] Testing '{run_name}' model...")

                # TEST SETS -- use X_test_malicious_without_attack_cmd, backdoor them and check how many are evasive
                X_test_malicious_cmd_poisoned = [backdoor_command(backdoor, cmd) for cmd in X_test_malicious_without_attack_cmd]
                with open(poisoned_sample_file, "w", encoding="utf-8") as f:
                    json.dump(X_test_malicious_cmd_poisoned, f, indent=4)
                y_test_malicious_poisoned = np.ones(len(X_test_malicious_cmd_poisoned), dtype=np.int8)

                Xy_test_loader_poisoned = commands_to_loader(
                    X_test_malicious_cmd_poisoned,
                    tokenizer,
                    y=y_test_malicious_poisoned,
                    batch_size=BATCH_SIZE,
                    workers=DATALOADER_WORKERS
                )
                scores = collect_scores(
                    Xy_test_loader_poisoned,
                    y_test_malicious_poisoned,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    scores=None,
                    score_suffix="_backdoor",
                    run_name=run_name
                )

                Xy_test_loader_orig = commands_to_loader(
                    X_test_malicious_without_attack_cmd,
                    tokenizer,
                    y=y_test_malicious_poisoned,
                    batch_size=BATCH_SIZE,
                    workers=DATALOADER_WORKERS
                )
                scores = collect_scores(
                    Xy_test_loader_orig,
                    y_test_malicious_poisoned,
                    trainer_poisoned,
                    lightning_model_poisoned,
                    scores=scores,
                    score_suffix="_orig",
                    run_name=run_name
                )
                
                # collect scores on orig test set too
                scores = collect_scores(
                    Xy_test_loader_orig_full,
                    y_test,
                    trainer_poisoned,
                    lightning_model_poisoned,
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
