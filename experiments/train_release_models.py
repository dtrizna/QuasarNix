import os
import re
import time
import json
import random
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from watermark import watermark
from typing import List

# from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# importing root of repository
import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT)

from src.models import *
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import create_dataloader, commands_to_loader, load_data, load_nl2bash
from src.lit_utils import LitTrainerWrapper
from src.tabular_utils import training_tabular
from lightning.fabric.utilities.seed import seed_everything

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


ATTACK = attack_hybrid
ROBUST_TRAINING_PARAM = 0.5
ROBUST_MANIPULATION_PROB = 0.5
ATTACK_PARAMETER = 1

SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

# # TEST RUN CONFIG
# DEVICE = "cpu"
# EPOCHS = 1
# LIT_SANITY_STEPS = 0
# LIMIT = 5000
# DATALOADER_WORKERS = 1
# LOGS_FOLDER = "TEST_logs_adv_train_full"

# PROD RUN CONFIG
DEVICE = "gpu"
EPOCHS = 10
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = "logs_adv_train_full"

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
    print(f"[!] X_train_malicious_cmd: {len(X_train_malicious_cmd)} | X_test_malicious_cmd: {len(X_test_malicious_cmd)}")
    print(f"[!] X_train_baseline_cmd: {len(X_train_baseline_cmd)} | X_test_baseline_cmd: {len(X_test_baseline_cmd)}")

    X_train_malicious_cmd_file = os.path.join(LOGS_FOLDER, f"X_train_malicious_cmd.json")
    with open(X_train_malicious_cmd_file, "w", encoding="utf-8") as f:
        json.dump(X_train_malicious_cmd, f, indent=4)

    X_test_malicious_cmd_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_cmd.json")
    with open(X_test_malicious_cmd_file, "w", encoding="utf-8") as f:
        json.dump(X_test_malicious_cmd, f, indent=4)

    # =============================================
    # ADVERSARIAL DATA SET
    # =============================================

    # create adversarial data sets
    X_train_malicious_cmd_adv_file = os.path.join(LOGS_FOLDER, f"X_train_malicious_cmd_adv.json")
    if os.path.exists(X_train_malicious_cmd_adv_file):
        print(f"[*] Loading adversarial training set from:\n\t'{X_train_malicious_cmd_adv_file}'")
        with open(X_train_malicious_cmd_adv_file, "r", encoding="utf-8") as f:
            X_train_malicious_cmd_adv = json.load(f)
    else:
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

    X_test_malicious_cmd_adv_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_cmd_adv.json")
    if os.path.exists(X_test_malicious_cmd_adv_file):
        print(f"[*] Loading adversarial test set from:\n\t'{X_test_malicious_cmd_adv_file}'")
        with open(X_test_malicious_cmd_adv_file, "r", encoding="utf-8") as f:
            X_test_malicious_cmd_adv = json.load(f)
    else:
        print("[*] Creating robust test set: applying attack with custom parameter...")
        X_test_malicious_cmd_adv = []
        for cmd in tqdm(X_test_malicious_cmd):
            if random.random() <= ROBUST_MANIPULATION_PROB:
                cmd_a = ATTACK(cmd, BASELINE, attack_parameter=ROBUST_TRAINING_PARAM)
            else:
                cmd_a = cmd
            X_test_malicious_cmd_adv.append(cmd_a)

        with open(X_test_malicious_cmd_adv_file, "w", encoding="utf-8") as f:
            json.dump(X_test_malicious_cmd_adv, f, indent=4)

    X_full_malicious_cmd_adv = X_train_malicious_cmd_adv + X_test_malicious_cmd_adv
    X_full_cmd_adv = X_full_malicious_cmd_adv + X_train_baseline_cmd + X_test_baseline_cmd
    y_full_adv = np.array(
        [1] * len(X_full_malicious_cmd_adv) + \
        [0] * len(X_train_baseline_cmd + X_test_baseline_cmd)
    )
    X_full_cmd_adv, y_full_adv = shuffle(X_full_cmd_adv, y_full_adv, random_state=SEED)

    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    # ========== EMBEDDING ==========
    vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}.json")
    if os.path.exists(vocab_file):
        print(f"[*] Loading vocab from:\n\t'{vocab_file}'")
        tokenizer.load_vocab(vocab_file)
    else:
        print("[*] Building vocab and encoding...")
        X_full_tokens = tokenizer.tokenize(X_full_cmd_adv)
        tokenizer.build_vocab(X_full_tokens)
        tokenizer.dump_vocab(vocab_file)

    # creating dataloaders
    print("[*] Creating dataloaders from commands...")
    X_train_loader = commands_to_loader(X_full_cmd_adv, tokenizer, y=y_full_adv, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE)
    X_test_loader = commands_to_loader(X_full_cmd_adv, tokenizer, y=y_full_adv, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE)

    # ========== MIN-HASH TABULAR ENCODING ==========
    # minhash = HashingVectorizer(n_features=VOCAB_SIZE, tokenizer=TOKENIZER, token_pattern=None)
    # print("[*] Fitting MinHash encoder...")
    # X_train_minhash = minhash.fit_transform(X_full_cmd_adv)

    # ========== ONE-HOT TABULAR ENCODING ===========
    oh_pickle = os.path.join(LOGS_FOLDER, f"onehot_vectorizer_{VOCAB_SIZE}.pkl")
    if os.path.exists(oh_pickle):
        print(f"[*] Loading One-Hot encoder from:\n\t'{oh_pickle}'")
        with open(oh_pickle, "rb") as f:
            oh = pickle.load(f)
        X_train_onehot = oh.transform(X_full_cmd_adv)
    else:
        print("[*] Fitting One-Hot encoder...")
        oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
        X_train_onehot = oh.fit_transform(X_full_cmd_adv)
        with open(oh_pickle, "wb") as f:
            pickle.dump(oh, f)

    # =============================================
    # DEFINING MODELS
    # =============================================

    # mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    # lstm_model = BiLSTMModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 318 K params
    # cnn_lstm_model = CNN1D_BiLSTM_Model(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_size=3, lstm_hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 316 K params
    # mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    # attpool_transformer_model = AttentionPoolingTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) #  335 K params
    # neurlux = NeurLuxModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, max_len=MAX_LEN, hidden_dim=32, output_dim=1, dropout=DROPOUT) # 402 K params
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    cls_transformer_model = CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) #  335 K params

    # tabular models
    # rf_model_minhash = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    # xgb_model_minhash = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    # log_reg_minhash = LogisticRegression(random_state=SEED)
    # mlp_tab_model_minhash = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    # rf_model_onehot = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    # log_reg_onehot = LogisticRegression(random_state=SEED)
    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params

    models = {
        # "_tabular_mlp_minhash": mlp_tab_model_minhash,
        # "_tabular_rf_minhash": rf_model_minhash,
        # "_tabular_xgb_minhash": xgb_model_minhash,
        # "_tabular_log_reg_minhash": log_reg_minhash,
        # "_tabular_rf_onehot": rf_model_onehot,
        # "_tabular_log_reg_onehot": log_reg_onehot,
        "_tabular_mlp_onehot": mlp_tab_model_onehot,
        "_tabular_xgb_onehot": xgb_model_onehot,
        # "mlp_seq": mlp_seq_model,
        # "attpool_transformer": attpool_transformer_model,
        # "mean_transformer": mean_transformer_model,
        # "neurlux": neurlux,
        # "lstm": lstm_model,
        # "cnn_lstm": cnn_lstm_model,
        "cnn": cnn_model,
        "cls_transformer": cls_transformer_model,
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
            x_train_full = None
            
            preprocessor = name.split("_")[-1]
            assert preprocessor in ["onehot", "minhash"]

            if preprocessor == "onehot":
                x_train_full = X_train_onehot
            # elif preprocessor == "minhash":
            #     x_train = X_train_minhash
                
            if "_mlp_" in name:
                # DEPRECATED
                # train_loader = create_dataloader(x_train_full, y_full_adv, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                # test_loader = create_dataloader(x_train_full, y_full_adv, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                # _ = train_lit_model(
                #     train_loader,
                #     test_loader,
                #     model,
                #     name,
                #     log_folder=LOGS_FOLDER,
                #     epochs=EPOCHS,
                #     learning_rate=LEARNING_RATE,
                #     scheduler=SCHEDULER,
                #     scheduler_budget = EPOCHS * len(X_train_loader),
                #     device=DEVICE
                # )
                
                # NEW CLASS
                trainer = LitTrainerWrapper(
                    pytorch_model=model,
                    name=name,
                    log_folder=LOGS_FOLDER,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    device=DEVICE,
                    precision=16,
                )
                train_loader = trainer.create_dataloader(x_train_full, y_full_adv, batch_size=BATCH_SIZE, dataloader_workers=DATALOADER_WORKERS)
                test_loader = trainer.create_dataloader(x_train_full, y_full_adv, batch_size=BATCH_SIZE, dataloader_workers=DATALOADER_WORKERS)
                trainer.train_lit_model(train_loader, test_loader)

            else:
                training_tabular(
                    model=model,
                    name=name,
                    X_train_encoded=x_train_full,
                    X_test_encoded=x_train_full,
                    y_train=y_full_adv,
                    y_test=y_full_adv,
                    logs_folder=LOGS_FOLDER
                )
        else:
            # DEPRECATED
            # _ = train_lit_model(
            #     X_train_loader,
            #     X_test_loader,
            #     model,
            #     name,
            #     log_folder=LOGS_FOLDER,
            #     epochs=EPOCHS,
            #     learning_rate=LEARNING_RATE,
            #     scheduler=SCHEDULER,
            #     scheduler_budget= EPOCHS * len(X_train_loader),
            #     device=DEVICE
            # )

            # NEW CLASS
            trainer = LitTrainerWrapper(
                pytorch_model=model,
                name=name,
                log_folder=LOGS_FOLDER,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                scheduler=SCHEDULER,
                device=DEVICE,
                precision=16,
            )
            trainer.train_lit_model(X_train_loader, X_test_loader)
        
        print(f"[!] Training of {name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

    print(f"[!] Script end time: {time.ctime()}")
