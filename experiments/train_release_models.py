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

from nltk.tokenize import wordpunct_tokenize
from xgboost import XGBClassifier

# importing root of repository
import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT)

from src.models import *
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import commands_to_loader, load_data, load_nl2bash
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
    command_adv = attack_domain_knowledge(command_adv, baseline, attack_parameter)
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


def attack_domain_knowledge(
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


# =============================
# DATA SET WRAPPERS
# =============================

def create_adversarial_set(malicious_cmd: List[str], dataset_name: str) -> List[str]:
    assert dataset_name in ["train", "test"]

    adv_set_file = os.path.join(LOGS_FOLDER, f"X_{dataset_name}_malicious_cmd_adv.json")
    if os.path.exists(adv_set_file):
        print(f"[*] Loading adversarial {dataset_name} set from:\n\t'{adv_set_file}'")
        with open(adv_set_file, "r", encoding="utf-8") as f:
            x_malicious_cmd_adv = json.load(f)
    else:
        print(f"[*] Creating robust {dataset_name} set: applying attack with probability {ROBUST_MANIPULATION_PROB} and strength parameter {ROBUST_TRAINING_PARAM}...")
        x_malicious_cmd_adv = []
        for cmd in tqdm(malicious_cmd):
            if random.random() <= ROBUST_MANIPULATION_PROB:
                cmd_a = ATTACK(cmd, BASELINE, attack_parameter=ROBUST_TRAINING_PARAM)
            else:
                cmd_a = cmd
            x_malicious_cmd_adv.append(cmd_a)
        
        with open(adv_set_file, "w", encoding="utf-8") as f:
            json.dump(x_malicious_cmd_adv, f, indent=4)
    
    return x_malicious_cmd_adv


def get_embedded_tokenizer(x_cmds: List[str], tokenizer_type: str) -> CommandTokenizer:
    assert tokenizer_type in ["orig", "adv"]

    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}_{tokenizer_type}.json")
    
    if os.path.exists(vocab_file):
        print(f"[*] Loading vocab from:\n\t'{vocab_file}'")
        tokenizer.load_vocab(vocab_file)
    else:
        print(f"[*] Building {tokenizer_type} vocab and encoding...")
        x_tokens = tokenizer.tokenize(x_cmds)
        tokenizer.build_vocab(x_tokens)
        tokenizer.dump_vocab(vocab_file)

    return tokenizer


def get_onehot_tokenizer(x_cmds: List[str], tokenizer_type: str) -> OneHotCustomVectorizer:
    assert tokenizer_type in ["orig", "adv"]

    oh_pickle = os.path.join(LOGS_FOLDER, f"onehot_vectorizer_{VOCAB_SIZE}_{tokenizer_type}.pkl")
    if os.path.exists(oh_pickle):
        print(f"[*] Loading One-Hot encoder from:\n\t'{oh_pickle}'")
        with open(oh_pickle, "rb") as f:
            oh_tokenizer = pickle.load(f)
    else:
        print("[*] Fitting One-Hot encoder...")
        oh_tokenizer = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
        oh_tokenizer.fit(x_cmds)
        with open(oh_pickle, "wb") as f:
            pickle.dump(oh_tokenizer, f)
    
    return oh_tokenizer


def find_lit_checkpoint_folder(train_name):
    train_folder = os.path.join(LOGS_FOLDER, f"{train_name}_csv")
    version = sorted(os.listdir(train_folder))[-1]
    return os.path.join(train_folder, version, "checkpoints")


# =============================
# RUN CONFIGURATION
# =============================

ATTACK = attack_hybrid
ROBUST_TRAINING_PARAM = 0.7
ROBUST_MANIPULATION_PROB = 0.5

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
        X_train_cmds_orig,
        y_train_orig,
        X_test_cmds_orig,
        y_test_orig,
        X_train_malicious_cmds,
        X_train_baseline_cmds,
        X_test_malicious_cmds,
        X_test_baseline_cmds
    ) = load_data(ROOT, SEED, limit=LIMIT)

    print(f"[!] X_train_malicious_cmd: {len(X_train_malicious_cmds)} | X_test_malicious_cmd: {len(X_test_malicious_cmds)}")
    print(f"[!] X_train_baseline_cmd: {len(X_train_baseline_cmds)} | X_test_baseline_cmd: {len(X_test_baseline_cmds)}")

    X_train_malicious_cmd_file = os.path.join(LOGS_FOLDER, f"X_train_malicious_cmd.json")
    with open(X_train_malicious_cmd_file, "w", encoding="utf-8") as f:
        json.dump(X_train_malicious_cmds, f, indent=4)

    X_test_malicious_cmd_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_cmd.json")
    with open(X_test_malicious_cmd_file, "w", encoding="utf-8") as f:
        json.dump(X_test_malicious_cmds, f, indent=4)

    # =============================================
    # ADVERSARIAL DATA SET
    # =============================================

    X_train_malicious_cmds_adv = create_adversarial_set(X_train_malicious_cmds, "train")
    X_test_malicious_cmds_adv = create_adversarial_set(X_test_malicious_cmds, "test")

    X_train_cmds_adv = X_train_malicious_cmds_adv + X_train_baseline_cmds
    y_train_adv = np.array([1] * len(X_train_malicious_cmds_adv) + [0] * len(X_train_baseline_cmds))
    X_train_cmds_adv, y_train_adv = shuffle(X_train_cmds_adv, y_train_adv, random_state=SEED)

    X_test_cmds_adv = X_test_malicious_cmds_adv + X_test_baseline_cmds
    y_test_adv = np.array([1] * len(X_test_malicious_cmds_adv) + [0] * len(X_test_baseline_cmds))
    X_test_cmds_adv, y_test_adv = shuffle(X_test_cmds_adv, y_test_adv, random_state=SEED)

    X_full_cmds_adv = X_train_malicious_cmds_adv + X_test_malicious_cmds_adv + \
                        X_train_baseline_cmds + X_test_baseline_cmds
    y_full_adv = np.array([1] * len(X_train_malicious_cmds_adv) + [1] * len(X_test_malicious_cmds_adv) + \
                          [0] * len(X_train_baseline_cmds + X_test_baseline_cmds))
    X_full_cmds_adv, y_full_adv = shuffle(X_full_cmds_adv, y_full_adv, random_state=SEED)

    # =============================================
    # TOKENIZERS
    # =============================================

    # ========== EMBEDDING ==========
    
    # non-robust original model trained on training set
    tokenizer_train_orig = get_embedded_tokenizer(X_train_cmds_orig, "train_orig")
    
    # adversarially trained model on training set for further research
    tokenizer_train_adv = get_embedded_tokenizer(X_train_cmds_adv, "train_adv")
    
    # production release model trained on all data
    tokenizer_full_adv = get_embedded_tokenizer(X_full_cmds_adv, "full_adv")

    print("[*] Creating dataloaders from commands...")

    X_train_loader_orig = commands_to_loader(
        X_train_cmds_orig,
        tokenizer_train_orig,
        y=y_train_orig,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE
    )
    X_test_loader_orig = commands_to_loader(
        X_test_cmds_orig,
        tokenizer_train_orig,
        y=y_test_orig,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE
    )

    X_train_loader_adv = commands_to_loader(
        X_train_cmds_adv,
        tokenizer_train_adv,
        y=y_train_adv,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE
    )
    X_test_loader_adv = commands_to_loader(
        X_test_cmds_adv,
        tokenizer_train_adv,
        y=y_test_adv,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE
    )

    X_full_loader_adv = commands_to_loader(
        X_full_cmds_adv,
        tokenizer_full_adv,
        y=y_full_adv,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE
    )

    # ========== ONE-HOT TABULAR ENCODING ===========
    
    # non-robust original model trained on training set
    oh_tokenizer_train_orig = get_onehot_tokenizer(X_train_cmds_orig, "train_orig")
    
    # adversarially trained model on training set for further research
    oh_tokenizer_train_adv = get_onehot_tokenizer(X_train_cmds_adv, "train_adv")
    
    # production release model trained on all data
    oh_tokenizer_full_adv = get_onehot_tokenizer(X_full_cmds_adv, "full_adv")

    print("[*] Transforming commands to One-Hot encoding...")

    X_train_onehot_orig = oh_tokenizer_train_orig.transform(X_train_cmds_orig)
    X_test_onehot_orig = oh_tokenizer_train_orig.transform(X_test_cmds_orig)

    X_train_onehot_adv = oh_tokenizer_train_adv.transform(X_train_cmds_adv)
    X_test_onehot_adv = oh_tokenizer_train_adv.transform(X_test_cmds_adv)

    X_full_onehot_adv = oh_tokenizer_full_adv.transform(X_full_cmds_adv)

    # =============================================
    # DEFINING MODELS
    # =============================================

    models = {
        # xgb
        "xgb_train_orig": XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        "xgb_train_adv": XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        "xgb_full_adv": XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        # MLP models
        "mlp_train_orig": SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT), # 264 K params
        "mlp_train_adv": SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT), # 264 K params
        "mlp_full_adv": SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT), # 264 K params
        # CNN models
        "cnn_train_orig": CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT), # 301 K params
        "cnn_train_adv": CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT), # 301 K params
        "cnn_full_adv": CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT), # 301 K params
        # transformers
        "transformer_train_orig": CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1), #  335 K params
        "transformer_train_adv": CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1), #  335 K params
        "transformer_full_adv": CLSTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1), #  335 K params

    }

    # =============================================
    # TRAINING MODELS
    # =============================================

    for name, model in models.items():
        x_train, y_train, x_test, y_test, X_train_loader, X_test_loader = None, None, None, None, None, None
        
        if ("xgb" not in name and os.path.exists(find_lit_checkpoint_folder(name))) or \
            ("xgb" in name and os.path.exists(os.path.join(LOGS_FOLDER, name, "model.xgboost"))):
            print(f"[!] Training of {name} already done, skipping...")
            continue

        now = time.time()
        print(f"[!] Training of {name} started: ", time.ctime())
        
        # selectin train and test sets for this model
        if name.startswith("xgb") or name.startswith("mlp"):
            if "train_orig" in name:
                x_train, y_train, x_test, y_test = X_train_onehot_orig, y_train_orig, X_test_onehot_orig, y_test_orig
            elif "train_adv" in name:
                x_train, y_train, x_test, y_test = X_train_onehot_adv, y_train_adv, X_test_onehot_adv, y_test_adv
            else:
                x_train, y_train, x_test, y_test = X_full_onehot_adv, y_full_adv, X_full_onehot_adv, y_full_adv
        else:
            if "train_orig" in name:
                X_train_loader, X_test_loader = X_train_loader_orig, X_test_loader_orig
            elif "train_adv" in name:
                X_train_loader, X_test_loader = X_train_loader_adv, X_test_loader_adv
            else:
                X_train_loader, X_test_loader = X_full_loader_adv, X_full_loader_adv

        # actual training
        if "xgb" in name:
            training_tabular(
                    model=model,
                    name=name,
                    X_train_encoded=x_train,
                    X_test_encoded=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    logs_folder=LOGS_FOLDER
                )
        else:
            trainer = LitTrainerWrapper(
                    pytorch_model=model,
                    name=name,
                    log_folder=LOGS_FOLDER,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    device=DEVICE,
                    precision=32
                )
            if "mlp" in name:
                X_train_loader = trainer.create_dataloader(x_train, y_train, batch_size=BATCH_SIZE, dataloader_workers=DATALOADER_WORKERS)
                X_test_loader = trainer.create_dataloader(x_test, y_test, batch_size=BATCH_SIZE, dataloader_workers=DATALOADER_WORKERS)
            trainer.train_lit_model(X_train_loader, X_test_loader)
        
        print(f"[!] Training of {name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

    print(f"[!] Script end time: {time.ctime()}")
