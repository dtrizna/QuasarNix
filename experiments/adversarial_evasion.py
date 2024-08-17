import os
import re
import time
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from watermark import watermark
from typing import List

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(ROOT)

# modeling
from lightning.fabric.utilities.seed import seed_everything
from src.models import (
    SimpleMLPWithEmbedding,
    CNN1DGroupedModel,
    MeanTransformerEncoder,
    CLSTransformerEncoder,
    SimpleMLP,
)
from xgboost import XGBClassifier
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import commands_to_loader, load_data, load_nl2bash
from src.tabular_utils import training_tabular
from src.lit_utils import load_lit_model, train_lit_model, predict_lit_model

# =============================
# ATTACK FUNCTIONS
# =============================

def attack_hybrid(
        command: str,
        baseline: List[str],
        attack_parameter: int,
        template: str = None
) -> str:
    command_adv = attack_template_prepend(command, baseline, int(attack_parameter * 128), template)
    command_adv = attack_evasive_tricks(command_adv, baseline, attack_parameter)
    return command_adv


def attack_template_prepend(
        command: str,
        baseline: List[str],
        attack_parameter: int, # payload_size
        template: str = None
) -> str:
    """
    This is an adversarial attack on command, that samples from baseline
    and appends to target attack using awk.
    """
    if template is None:
        # template = """awk 'BEGIN { print ARGV[1] }' "PAYLOAD" """
        template = """python3 -c "print('PAYLOAD')" """
    
    # while not exceeds payload_size -- sample from baseline and add to payload
    payload = ""
    while len(payload) < attack_parameter:
        payload += np.random.choice(baseline) + ";"
    
    payload = payload[:attack_parameter]
    payload = template.replace("PAYLOAD", payload)
    
    return payload + ";" + command


def ip_to_decimal(ip):
    """
    Idea from: https://book.hacktricks.xyz/linux-hardening/bypass-bash-restrictions
    Implementation: ChatGPT.

    # Decimal IPs
    127.0.0.1 == 2130706433

    root@dmz:~# ping 2130706433
    PING 2130706433 (127.0.0.1) 56(84) bytes of data.
    64 bytes from 127.0.0.1: icmp_seq=1 ttl=64 time=0.030 ms
    64 bytes from 127.0.0.1: icmp_seq=2 ttl=64 time=0.036 ms
    ^C
    """
    parts = [int(part) for part in ip.split(".")]
    if len(parts) != 4:
        raise ValueError("The input does not seem to be a valid IPv4 address")    
    decimal_value = (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    return decimal_value


def replace_ip_with_decimal(command_adv):
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ip_match = re.search(ip_pattern, command_adv)
    if ip_match:
        ip_address = ip_match.group()
        try:
            decimal_ip = ip_to_decimal(ip_address)
            command_adv = command_adv.replace(ip_address, str(decimal_ip))
        except ValueError:
            print("A potential IP was found, but it's not valid. No replacement was made.")
    return command_adv


def attack_evasive_tricks(
        command: str,
        baseline: List[str], # keep for backward compatibility with attack_template_prepend
        attack_parameter: float = 0.7 # threshold how often to change, here: 70% of the time
) -> str:
    command_adv = command

    # replaces
    python_renames = [
            "cp /usr/bin/python /tmp/test; /tmp/test ",
            "cp /usr/bin/python /tmp/python; /tmp/python ",
        ]
    replace_maps = {
        # bash tricks
        "sh -i": [
            "sh -li",
            "sh -i -l",
            "sh -a -i",
            "sh -avi"
        ],
        # ";exec": [";id;exec", ";find /home 2>/dev/null;exec", ""], # not in templates
        ";cat": [";id;cat", ";readlink;cat", ";whoami;cat", ";find /home 2>/dev/null;cat"],

        # nc tricks
        "nc -e": ["nc -ne", "nc -v -e", "nc -env"],
        "nc ": ["ncat ", "nc.traditional "],
        
        # perl swaps
        "use Socket;": "use warnings; use Socket;",
        "perl -e": ["perl -S -e", "perl -t -e"],
        "perl ": [
            "cp /usr/bin/perl /tmp/test; /tmp/test ",
            "cp /usr/bin/perl /tmp/perl; /tmp/perl ",
        ],

        # php swaps
        "php -r": "php -e -r",
        "php ": "cp /usr/bin/php /tmp/test; /tmp/test ",

        # ruby swaps
        "ruby -rsocket": [
            "ruby -ruri -rsocket",
            "ruby -ryaml -rsocket"
        ],
        "-rsocket -e": "-rsocket -a -e",
        "-e'spawn": """-e'puts"test".inspect;spawn""",
        "ruby ": [
            "cp /usr/bin/ruby /tmp/test; /tmp/test ",
            "cp /usr/bin/ruby /tmp/ruby; /tmp/ruby "
        ],

        # python swaps
        "python -c": "python -b -c",
        "python3 -c": "python3 -b -c",
        "python ": "python2.7 ",
        "python ": python_renames,
        "python3 ": python_renames,
        "python2.7 ": python_renames,
        "import os": "import sys,os",
        "import socket": "import sys,socket",
        "os.system": "import os as bs;bs.system"
    }
    for replace_src, replace_dst in replace_maps.items():
        if replace_src in command_adv:
            chance = random.random()
            if chance <= attack_parameter:
                if isinstance(replace_dst, str):
                    command_adv = command_adv.replace(replace_src, replace_dst)
                elif isinstance(replace_dst, list):
                    command_adv = command_adv.replace(replace_src, random.choice(replace_dst))
    
    # ip manipulation
    chance = random.random()
    if chance <= attack_parameter:
        command_adv = replace_ip_with_decimal(command_adv)

    return command_adv


SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 512
DROPOUT = 0.5
DEVICE = "gpu"
LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

# RUN CONFIG
ADV_ATTACK_SUBSAMPLE = 50000
EPOCHS = 10
LIMIT = None

ROBUST_TRAINING_PARAM = 0.5
ROBUST_MANIPULATION_PROB = 0.5


# ATTACK NR.1:
# ATTACK = attack_template_prepend
# ATTACK_PARAMETERS = [16, 32, 48, 64, 80, 96, 112, 128] # PAYLOAD_SIZES
# # NOTE: Total size of injected characters: PAYLOAD_SIZE + 23
# # since len(template) = 23 (w/o PAYLOAD) when template = """python3 -c "print('PAYLOAD')" """
# LOGS_FOLDER = os.path.join(f"logs_adversarial_evasion", "nl2bash_prepend")

# ATTACK NR.2:
# ATTACK = attack_evasive_tricks
# ATTACK_PARAMETERS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
# LOGS_FOLDER = os.path.join(f"logs_adversarial_evasion", f"domain_knowledge_prob_{ROBUST_MANIPULATION_PROB}_attack_param_{ROBUST_TRAINING_PARAM}_limit_{LIMIT}")

# ATTACK NR.3:
ATTACK = attack_hybrid
ATTACK_PARAMETERS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
LOGS_FOLDER = os.path.join(f"logs_adversarial_evasion", f"hybrid_prob_{ROBUST_MANIPULATION_PROB}_attack_param_{ROBUST_TRAINING_PARAM}_limit_{LIMIT}")

os.makedirs(LOGS_FOLDER, exist_ok=True)

if __name__ == "__main__":
    # ===========================================
    print(f"[!] Script start time: {time.ctime()}")
    print(watermark(packages="torch,lightning,sklearn", python=True))

    TOKENIZER = wordpunct_tokenize
    seed_everything(SEED)

    # ============================================
    # LOADING DATA
    # ============================================
    BASELINE = load_nl2bash(ROOT)

    (
        X_train_cmds,
        y_train,
        X_test_cmds,
        y_test,
        X_train_malicious_cmd,
        X_train_baseline_cmd,
        X_test_malicious_cmd,
        X_test_baseline_cmd
    ) = load_data(ROOT, SEED, limit=LIMIT)
    
    print(f"Sizes of train and test sets: {len(X_train_cmds)}, {len(X_test_cmds)}")
    
    # =============================================
    # CONSTRUCTING ADVERSARIAL TEST SET
    # =============================================

    # randomly subsample for adversarial attack
    sample_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_without_attack_cmd_sample_{ADV_ATTACK_SUBSAMPLE}.json")
    if os.path.exists(sample_file):
        print(f"[*] Loading malicious test set sample from '{sample_file}'")
        with open(sample_file, "r", encoding="utf-8") as f:
            X_test_malicious_without_attack_cmd = json.load(f)
        print(f"[!] Size of malicious test set: {len(X_test_malicious_without_attack_cmd)}")
    else:
        print(f"[*] Subsampling malicious test set to {ADV_ATTACK_SUBSAMPLE} samples...")
        X_test_malicious_without_attack_cmd = np.random.choice(X_test_malicious_cmd, ADV_ATTACK_SUBSAMPLE, replace=False).tolist()
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_without_attack_cmd, f, indent=4)
    
    # =============================================
    # RUNNING ATTACK ON THIS SET
    # =============================================

    X_test_malicious_with_attack_cmd_dict = {}
    for attack_parameter in ATTACK_PARAMETERS:
        adv_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_with_attack_{attack_parameter}_cmd_sample_{ADV_ATTACK_SUBSAMPLE}.json")
        if os.path.exists(adv_file):
            print(f"[!] Loading adversarial test set for payload size {attack_parameter} from '{adv_file}'")
            with open(adv_file, "r", encoding="utf-8") as f:
                X_test_malicious_with_attack_cmd = json.load(f)
        else:
            print(f"[*] Constructing adversarial test set with attack's payload size {attack_parameter}...")
            X_test_malicious_with_attack_cmd = []
            for cmd in tqdm(X_test_malicious_without_attack_cmd):
                cmd_a = ATTACK(cmd, BASELINE, attack_parameter=attack_parameter)
                X_test_malicious_with_attack_cmd.append(cmd_a)
            # dump as json
            with open(adv_file, "w", encoding="utf-8") as f:
                json.dump(X_test_malicious_with_attack_cmd, f, indent=4)

        X_test_malicious_with_attack_cmd_dict[attack_parameter] = X_test_malicious_with_attack_cmd

    # =============================================
    # ADVERSARIAL TRAINING SET
    # =============================================

    # create adversarial training set
    X_train_malicious_cmd_adv_file = os.path.join(LOGS_FOLDER, f"X_train_malicious_cmd_adv.json")
    if os.path.exists(X_train_malicious_cmd_adv_file):
        print(f"[!] Loading adversarial training set from '{X_train_malicious_cmd_adv_file}'...")
        with open(X_train_malicious_cmd_adv_file, "r", encoding="utf-8") as f:
            X_train_malicious_cmd_adv = json.load(f)
    else:
        # NOTE: Initial Greedy Version
        # print("[*] Creating robust training set: greedy append of single baseline command to malicious one...")
        # X_train_malicious_cmd_adv = []
        # for cmd in tqdm(X_train_malicious_cmd):
        #     random_baseline_command = random.choice(X_train_baseline_cmd)
        #     cmd_a = cmd + ";" + random_baseline_command
        #     X_train_malicious_cmd_adv.append(cmd_a)

        # NOTE: Updated after discussion with Luca to include actual ATTACK in adversarial training
        print("[*] Creating robust training set: applying attack with custom parameter...")
        X_train_malicious_cmd_adv = []
        for cmd in tqdm(X_train_malicious_cmd):
            if random.random() <= ROBUST_MANIPULATION_PROB:
                cmd_a = ATTACK(cmd, BASELINE, attack_parameter=ROBUST_TRAINING_PARAM)
            else:
                cmd_a = cmd

            X_train_malicious_cmd_adv.append(cmd_a)
        
        with open(X_train_malicious_cmd_adv_file, "w", encoding="utf-8") as f:
            json.dump(X_train_malicious_cmd_adv, f, indent=4)

    X_train_cmd_adv = X_train_baseline_cmd + X_train_malicious_cmd_adv
    y_train_adv = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd_adv), dtype=np.int8)
    X_train_cmd_adv, y_train_adv = shuffle(X_train_cmd_adv, y_train_adv, random_state=SEED)

    # =============================================
    # DEFINING MODELS
    # =============================================

    # mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    # mlp_seq_model_adv = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    cnn_model_adv = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    
    # mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    # mean_transformer_model_adv = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    
    cls_transformer_model = CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    cls_transformer_model_adv = CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params

    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    mlp_tab_model_onehot_adv = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    
    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    xgb_model_onehot_adv = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)

    target_models = {
        "cnn": (cnn_model, cnn_model_adv),
        "mlp_onehot": (mlp_tab_model_onehot, mlp_tab_model_onehot_adv),
        # "mean_transformer": (mean_transformer_model, mean_transformer_model_adv),
        "cls_transformer": (cls_transformer_model, cls_transformer_model_adv),
        "xgb_onehot": (xgb_model_onehot, xgb_model_onehot_adv),
        #"mlp_seq": (mlp_seq_model, mlp_seq_model_adv),
    }

    for name, (target_model_orig, target_model_adv) in target_models.items():
        print(f"[!!!] Starting attack against '{name}' model...")

        # =============================================
        # PREPING DATA
        # =============================================

        if "onehot" in name:
            # # ========== ONE-HOT TABULAR ENCODING ===========
            oh_tokenizer_file_orig = os.path.join(LOGS_FOLDER, f"onehot_tokenizer_{VOCAB_SIZE}_orig.pkl")
            if os.path.exists(oh_tokenizer_file_orig):
                print(f"[!] Loading One-Hot tokenizer from '{oh_tokenizer_file_orig}'...")
                with open(oh_tokenizer_file_orig, "rb") as f:
                    tokenizer_orig = pickle.load(f)
            else:
                tokenizer_orig = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
                print("[*] Fitting One-Hot encoder...")
                now = time.time()
                tokenizer_orig.fit(X_train_cmds)
                print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s
                with open(oh_tokenizer_file_orig, "wb") as f:
                    pickle.dump(tokenizer_orig, f)
        else:
            # ========== EMBEDDING ==========
            tokenizer_orig = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
            vocab_file_orig = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}_orig.json")
            if os.path.exists(vocab_file_orig):
                print(f"[!] Loading vocab from '{vocab_file_orig}'...")
                tokenizer_orig.load_vocab(vocab_file_orig)
            else:
                print("[*] Building vocab and encoding...")
                X_train_tokens = tokenizer_orig.tokenize(X_train_cmds)
                tokenizer_orig.build_vocab(X_train_tokens)
                tokenizer_orig.dump_vocab(vocab_file_orig)

        # creating dataloaders -- creating here to do it once for both sequential models
        Xy_train_loader = commands_to_loader(X_train_cmds, tokenizer_orig, y=y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
        Xy_test_loader = commands_to_loader(X_test_cmds, tokenizer_orig, y=y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

        # =======================================================
        # ORIG MODEL ADVERSARIAL SCORES
        # =======================================================

        run_name = f"{name}_orig"
        scores_json_file = os.path.join(LOGS_FOLDER, f"adversarial_scores_{run_name}.json")
        if os.path.exists(scores_json_file):
            print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
        else:
            # ========== TRAINING =============
            model_file_orig = os.path.join(LOGS_FOLDER, f"{run_name}.ckpt")
            if "xgb" in name:
                X_train_onehot = tokenizer_orig.transform(X_train_cmds)
                X_test_onehot = tokenizer_orig.transform(X_test_cmds)
                model_orig = training_tabular(
                    target_model_orig,
                    run_name,
                    X_train_onehot,
                    X_test_onehot,
                    y_train,
                    y_test,
                    LOGS_FOLDER
                )
            else:
                if os.path.exists(model_file_orig):
                    print(f"[!] Loading original model from '{model_file_orig}'")
                    trainer_orig, lightning_model_orig = load_lit_model(
                        model_file_orig, 
                        target_model_orig, 
                        run_name, 
                        LOGS_FOLDER, 
                        EPOCHS,
                        DEVICE,
                        LIT_SANITY_STEPS)
                else:
                    print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
                    now = time.time()
                    trainer_orig, lightning_model_orig = train_lit_model(
                        Xy_train_loader,
                        Xy_test_loader,
                        target_model_orig,
                        run_name,
                        log_folder=LOGS_FOLDER,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        scheduler=SCHEDULER,
                        device=DEVICE,
                        scheduler_budget=EPOCHS * len(Xy_train_loader),
                        model_file=model_file_orig
                    )
                    print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

            # ========== SCORING =============
            accuracies_orig = {}
            evasive_orig = {}

            # ORIG MODEL SCORES ON TEST SET W/O ATTACK
            if "xgb" in name:
                X_test_malicious_without_attack_onehot = tokenizer_orig.transform(X_test_malicious_without_attack_cmd)
                y_pred_orig_orig = model_orig.predict(X_test_malicious_without_attack_onehot)
            else:
                X_test_malicious_without_attack_loader = commands_to_loader(X_test_malicious_without_attack_cmd, tokenizer_orig, batch_size=BATCH_SIZE, workers=1)
                y_pred_orig_orig = predict_lit_model(
                    X_test_malicious_without_attack_loader,
                    trainer_orig,
                    lightning_model_orig,
                    decision_threshold=0.5)
            evasive = len(y_pred_orig_orig[y_pred_orig_orig == 0])
            print(f"[!] Orig train | Orig test |  Evasive:", evasive)
            evasive_orig[0] = evasive

            acc = accuracy_score(np.ones_like(y_pred_orig_orig), y_pred_orig_orig)
            print(f"[!] Orig train | Orig test | Accuracy: {acc:.3f}")
            accuracies_orig[0] = acc

            # ORIG MODEL SCORES ON TEST SETS WITH ATTACK
            for attack_parameter, X_test_malicious_with_attack_cmd in X_test_malicious_with_attack_cmd_dict.items():
                if "xgb" in name:
                    X_test_malicious_adv_onehot = tokenizer_orig.transform(X_test_malicious_with_attack_cmd)
                    y_pred_orig_adv = model_orig.predict(X_test_malicious_adv_onehot)
                else:
                    X_test_malicious_adv_loader = commands_to_loader(X_test_malicious_with_attack_cmd, tokenizer_orig, batch_size=BATCH_SIZE, workers=1)
                    y_pred_orig_adv = predict_lit_model(
                        X_test_malicious_adv_loader,
                        trainer_orig,
                        lightning_model_orig,
                        decision_threshold=0.5,
                        # dump_logits=os.path.join(LOGS_FOLDER, f"{run_name}_y_pred_with_attack_logits_payload_{attack_parameter}_sample_{ADV_ATTACK_SUBSAMPLE}.pkl")
                    )
                evasive = len(y_pred_orig_adv[y_pred_orig_adv == 0])
                print(f"[!] Orig train | Adv test | Payload {attack_parameter} |  Evasive:" , evasive)
                evasive_orig[attack_parameter] = evasive

                acc = accuracy_score(np.ones_like(y_pred_orig_adv), y_pred_orig_adv)
                print(f"[!] Orig train | Adv test | Payload {attack_parameter} | Accuracy: {acc:.3f}")
                accuracies_orig[attack_parameter] = acc
            
            accuracies_orig = {float(key): value for key, value in accuracies_orig.items()}
            evasive_orig = {float(key): value for key, value in evasive_orig.items()}
            results_dict = {'Accuracy': accuracies_orig, 'Evasive Samples': evasive_orig}
            results_json = json.dumps(results_dict, indent=4)
            with open(scores_json_file, 'w') as f:
                f.write(results_json)

        # =======================================================
        # ADVERSARIAL MODEL ADVERSARIAL SCORES
        # =======================================================

        run_name = f"{name}_adv"
        scores_json_file = os.path.join(LOGS_FOLDER, f"adversarial_scores_{run_name}.json")
        if os.path.exists(scores_json_file):
            print(f"[!] Scores already calculated for '{run_name}'! Skipping...")
        else:
            if "xgb" in name:
                oh_tokenizer_adv_path = os.path.join(LOGS_FOLDER, f"onehot_tokenizer_{VOCAB_SIZE}_adv.pkl")
                if os.path.exists(oh_tokenizer_adv_path):
                    print(f"[!] Loading One-Hot tokenizer from '{oh_tokenizer_adv_path}'...")
                    with open(oh_tokenizer_adv_path, "rb") as f:
                        tokenizer_adv = pickle.load(f)
                else:
                    print("[*] Fitting One-Hot encoder for adversarial model...")
                    tokenizer_adv = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
                    tokenizer_adv.fit(X_train_cmd_adv)
                    with open(oh_tokenizer_adv_path, "wb") as f:
                        pickle.dump(tokenizer_adv, f)

                X_train_onehot_adv = tokenizer_adv.transform(X_train_cmd_adv)
                X_test_onehot = tokenizer_adv.transform(X_test_cmds)
                model_adv = training_tabular(
                    target_model_adv,
                    run_name,
                    X_train_onehot_adv,
                    X_test_onehot,
                    y_train_adv,
                    y_test,
                    LOGS_FOLDER
                )
            else:
                vocab_file_adv = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}_adv.json")
                tokenizer_adv = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
                if os.path.exists(vocab_file_adv):
                    print(f"[!] Loading vocab from '{vocab_file_adv}'...")
                    tokenizer_adv.load_vocab(vocab_file_adv)
                else:
                    print("[*] Building vocab and encoding for adversarial model...")
                    X_train_tokens_adv = tokenizer_adv.tokenize(X_train_cmd_adv)
                    tokenizer_adv.build_vocab(X_train_tokens_adv)
                    tokenizer_adv.dump_vocab(vocab_file_adv)
                
                model_file_adv = os.path.join(LOGS_FOLDER, f"{run_name}.ckpt")
                # ========== TRAINING =============
                if os.path.exists(model_file_adv):
                    print(f"[*] Loading adversarially trained model from {model_file_adv}...")
                    trainer_adv, lightning_model_adv = load_lit_model(
                        model_file_adv,
                        target_model_adv,
                        run_name,
                        LOGS_FOLDER,
                        EPOCHS,
                        DEVICE,
                        LIT_SANITY_STEPS)
                else:
                    Xy_train_loader_adv = commands_to_loader(X_train_cmd_adv, tokenizer_adv, y=y_train_adv, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                    # Train adversarial model
                    print(f"[!] Training of adversarial model '{run_name}' started: {time.ctime()}")
                    now = time.time()
                    trainer_adv, lightning_model_adv = train_lit_model(
                        Xy_train_loader_adv,
                        Xy_test_loader, # NOTE: using unmodified test set!
                        target_model_adv,
                        run_name,
                        LOGS_FOLDER,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        device=DEVICE,
                        scheduler_budget=EPOCHS * len(Xy_train_loader_adv),
                        model_file=model_file_adv
                    )
                    print(f"[!] Training of '{run_name}' ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

            # ========== SCORING =============
            accuracies_adv = {}
            evasive_adv = {}

            if "xgb" in name:
                X_test_malicious_without_attack_onehot = tokenizer_adv.transform(X_test_malicious_without_attack_cmd)
                y_pred_adv_orig = model_adv.predict(X_test_malicious_without_attack_onehot)
            else:
                # ADVERSARIAL MODEL SCORES ON TEST SET W/O ATTACK
                X_test_malicious_without_attack_loader = commands_to_loader(X_test_malicious_without_attack_cmd, tokenizer_adv, batch_size=BATCH_SIZE, workers=1)
                y_pred_adv_orig = predict_lit_model(
                    X_test_malicious_without_attack_loader,
                    trainer_adv,
                    lightning_model_adv,
                    decision_threshold=0.5
                )
            evasive = len(y_pred_adv_orig[y_pred_adv_orig == 0])
            print(f"[!] Adv train | Orig test |  Evasive:", evasive)
            evasive_adv[0] = evasive

            acc = accuracy_score(np.ones_like(y_pred_adv_orig), y_pred_adv_orig)
            print(f"[!] Adv train | Orig test | Accuracy: {acc:.3f}")
            accuracies_adv[0] = acc

            # ADVERSARIAL MODEL SCORES ON TEST SETS WITH ATTACK
            for attack_parameter, X_test_malicious_with_attack_cmd in X_test_malicious_with_attack_cmd_dict.items():
                if "xgb" in name:
                    X_test_malicious_adv_onehot = tokenizer_adv.transform(X_test_malicious_with_attack_cmd)
                    y_pred_adv_adv = model_adv.predict(X_test_malicious_adv_onehot)
                else:
                    X_test_malicious_adv_loader = commands_to_loader(X_test_malicious_with_attack_cmd, tokenizer_adv, batch_size=BATCH_SIZE, workers=1)
                    y_pred_adv_adv = predict_lit_model(
                        X_test_malicious_adv_loader,
                        trainer_adv,
                        lightning_model_adv,
                        decision_threshold=0.5,
                        # dump_logits=os.path.join(LOGS_FOLDER, f"{run_name}_y_pred_with_attack_logits_payload_{attack_parameter}_sample_{ADV_ATTACK_SUBSAMPLE}.pkl")
                    )
                evasive = len(y_pred_adv_adv[y_pred_adv_adv == 0])
                print(f"[!] Adv train | Adv test | Payload {attack_parameter} |  Evasive:" , evasive)
                evasive_adv[attack_parameter] = evasive

                acc = accuracy_score(np.ones_like(y_pred_adv_adv), y_pred_adv_adv)
                print(f"[!] Adv train | Adv test | Payload {attack_parameter} | Accuracy: {acc:.3f}")
                accuracies_adv[attack_parameter] = acc
            
            accuracies_adv = {float(key): value for key, value in accuracies_adv.items()}
            evasive_adv = {float(key): value for key, value in evasive_adv.items()}
            results_dict = {'Accuracy': accuracies_adv, 'Evasive Samples': evasive_adv}
            results_json = json.dumps(results_dict, indent=4)
            with open(scores_json_file, 'w') as f:
                f.write(results_json)

    # ===== END LOOP ======
    print(f"[!] Script end time: {time.ctime()}")
