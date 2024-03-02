import os
import time
import pickle
import numpy as np
from watermark import watermark
from nltk.tokenize import wordpunct_tokenize
from lightning.lite.utilities.seed import seed_everything

from src.augmentation import REVERSE_SHELL_TEMPLATES, NixCommandAugmentation
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import commands_to_loader, load_nl2bash, load_data, create_dataloader
from src.lit_utils import LitTrainerWrapper
from src.tabular_utils import training_tabular
from src.models import (
    CNN1DGroupedModel,
    MeanTransformerEncoder,
    SimpleMLP,
)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler

# ignore UndefinedMetricWarning because of
# use: python3 -W ignore ablation_augm_no_augm.py
# to ignore all warnings

RANDOM_SEED = 33
TEST_SIZE = 0.3
TRAIN_TEMPLATE_RATIO = 0.7

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
# LOGS_FOLDER = "TEST_logs_augm_no_augm"

# # PROD RUN CONFIG
DEVICE = "gpu"
EPOCHS = 10
LIT_SANITY_STEPS = 1
LIMIT = None
DATALOADER_WORKERS = 4
LOGS_FOLDER = os.path.join("logs_augm_no_augm", f"real_{int(time.time())}")


def generate_sets(random_state, log_folder=None):
    templates = REVERSE_SHELL_TEMPLATES

    # baselines
    # 1. NL2Bash
    # nl2bash = load_nl2bash(ROOT)
    # train_cmd_base, test_cmd_base = train_test_split(nl2bash, test_size=TEST_SIZE, random_state=random_state)
    # 2. real environment
    *_, train_cmd_base, _, test_cmd_base = load_data(ROOT, RANDOM_SEED, limit=LIMIT)
    
    # split templates to train and test
    train_templates, test_templates = train_test_split(templates, train_size=TRAIN_TEMPLATE_RATIO, random_state=random_state)

    # generate augmented train set
    train_entry_count_per_template = len(train_cmd_base) // len(templates)
    augmented_train = NixCommandAugmentation(templates=train_templates, random_state=random_state)
    train_cmd_rvrs = augmented_train.generate_commands(number_of_examples_per_template=train_entry_count_per_template)
    train_cmd_augmented = train_cmd_base + train_cmd_rvrs
    train_y_augmented = np.array([0] * len(train_cmd_base) + [1] * len(train_cmd_rvrs), dtype=np.int8)

    # generate non-augmented train set
    nonaugmented_train = NixCommandAugmentation(templates=train_templates, random_state=random_state)
    nonaugmented_train.placeholder_sampling_functions = {
            'NIX_SHELL': lambda: "/bin/sh",
            'PROTOCOL_TYPE': lambda: "tcp",
            'FD_NUMBER': lambda: 3,
            'FILE_PATH': lambda: "/tmp/f",
            'VARIABLE_NAME': lambda: "port",
            'IP_ADDRESS': lambda: "1.2.3.4",
            'PORT_NUMBER': lambda: 8080,
        }
    train_cmd_rvrs_1 = nonaugmented_train.generate_commands(number_of_examples_per_template=1)
    train_cmd_not_augmented = train_cmd_base + train_cmd_rvrs_1
    train_y_not_augmented = np.array([0] * len(train_cmd_base) + [1] * len(train_cmd_rvrs_1), dtype=np.int8)

    # generate test set
    augmented_test = NixCommandAugmentation(templates=test_templates, random_state=random_state)
    test_cmd_rvrs = augmented_test.generate_commands(number_of_examples_per_template=1)
    test_cmd = test_cmd_base + test_cmd_rvrs
    test_y = np.array([0] * len(test_cmd_base) + [1] * len(test_cmd_rvrs), dtype=np.int8)

    print(f"[!] Generated {len(train_cmd_augmented)} augmented train commands (balanced).")
    print(f"[!] Generated {len(train_cmd_not_augmented)} non-augmented train commands (imbalanced).")
    print(f"[!] Generated {len(test_cmd)} test commands (imbalanced).")
    print(f"    Size of train set: {len(train_cmd_base)}")
    print(f"    Size of test set: {len(test_cmd_base)}")

    if log_folder is not None:
        # dump
        with open(os.path.join(log_folder, "train_cmd_augm.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(train_cmd_augmented))
        with open(os.path.join(log_folder, "train_cmd_not_augm.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(train_cmd_not_augmented))
        with open(os.path.join(log_folder, "test_cmd.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(test_cmd))
        np.save(os.path.join(log_folder, "train_y_augm.npy"), train_y_augmented)
        np.save(os.path.join(log_folder, "train_y_not_augm.npy"), train_y_not_augmented)
        np.save(os.path.join(log_folder, "test_y.npy"), test_y)

    return train_cmd_augmented, train_cmd_not_augmented, test_cmd, train_y_augmented, train_y_not_augmented, test_y


def data_prep(name, X_train_cmds, y_train, X_test_cmds, y_test):
    # =============================================
    # PREPING DATA
    # =============================================
    tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    # ========== EMBEDDING ==========
    print("[*] Building vocab and encoding...")
    X_train_tokens = tokenizer.tokenize(X_train_cmds)
    tokenizer.build_vocab(X_train_tokens)

    vocab_file = os.path.join(LOGS_FOLDER, f"{name}_wordpunct_vocab_{VOCAB_SIZE}.json")
    tokenizer.dump_vocab(vocab_file)

    # creating dataloaders
    X_train_loader = commands_to_loader(X_train_cmds, tokenizer, y=y_train, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE)
    X_test_loader = commands_to_loader(X_test_cmds, tokenizer, y=y_test, workers=DATALOADER_WORKERS, batch_size=BATCH_SIZE)

    # ========== ONE-HOT TABULAR ENCODING ===========
    oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)

    print("[*] Fitting One-Hot encoder...")
    X_train_onehot = oh.fit_transform(X_train_cmds)
    X_test_onehot = oh.transform(X_test_cmds)
    
    # drop oh as pickle object
    oh_file = os.path.join(LOGS_FOLDER, f"{name}_onehot_{VOCAB_SIZE}.pkl")
    with open(oh_file, "wb") as f:
        pickle.dump(oh, f)

    return X_train_loader, X_test_loader, X_train_onehot, X_test_onehot


def train_models(run_name, X_train_onehot, X_test_onehot, X_train_loader, X_test_loader, y_train, y_test):
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    xgb_model_onehot = XGBClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)

    models = {
        "xgb_onehot": xgb_model_onehot,
        # "mlp_onehot": mlp_tab_model_onehot,
        # "mean_transformer": mean_transformer_model,
        # "cnn": cnn_model,
    }

    for name in models:
        model = models[name]
        name = f"{run_name}_{name}"
        
        now = time.time()
        print(f"[*] Training {name}... Started:", time.ctime())
        
        if "xgb_onehot" in name:
            model = training_tabular(
                model,
                name,
                X_train_onehot,
                X_test_onehot,
                y_train,
                y_test,
                logs_folder=LOGS_FOLDER
            )
            y_train_preds = model.predict_proba(X_train_onehot)[:,1]
            y_test_preds = model.predict_proba(X_test_onehot)[:,1]
            
        else:
            if "mlp_onehot" in name:
                train_loader = create_dataloader(X_train_onehot, y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
                test_loader = create_dataloader(X_test_onehot, y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
            else:
                train_loader = X_train_loader
                test_loader = X_test_loader
            lit_trainer = LitTrainerWrapper(
                pytorch_model=model,
                name=name,
                log_folder=LOGS_FOLDER,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                scheduler=SCHEDULER,
                early_stop_patience=50,
                monitor_metric="val_auc",
                lit_sanity_steps=LIT_SANITY_STEPS,
                log_every_n_steps=1
            )
            lit_trainer.train_lit_model(train_loader, test_loader)
            y_train_preds = lit_trainer.predict_proba(train_loader)
            y_test_preds = lit_trainer.predict_proba(test_loader)
        
        # dump train and test preds
        np.save(os.path.join(LOGS_FOLDER, f"y_train_preds_{name}.npy"), y_train_preds)
        np.save(os.path.join(LOGS_FOLDER, f"y_test_preds_{name}.npy"), y_test_preds)

        print(f"[!] Training of {name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")
        

if __name__ == "__main__":
    # ===========================================
    print(watermark(packages="torch,lightning,sklearn", python=True))
    print(f"[!] Script start time: {time.ctime()}")

    TOKENIZER = wordpunct_tokenize
    seed_everything(RANDOM_SEED)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    # ===========================================
    # LOADING DATA
    # ===========================================
    train_cmd_augm, train_cmd_not_augm, test_cmd, train_y_augm, train_y_not_augm, test_y = generate_sets(RANDOM_SEED, log_folder=LOGS_FOLDER)
    train_cmd_augm, train_y_augm = shuffle(train_cmd_augm, train_y_augm, random_state=RANDOM_SEED)
    train_cmd_not_augm, train_y_not_augm = shuffle(train_cmd_not_augm, train_y_not_augm, random_state=RANDOM_SEED)
    test_cmd, test_y = shuffle(test_cmd, test_y, random_state=RANDOM_SEED)
    
    # =============================================
    # PREPING DATA
    # =============================================
    X_train_loader_augm, X_test_loader_augm, X_train_onehot_augm, X_test_onehot_augm = data_prep("augm", train_cmd_augm, train_y_augm, test_cmd, test_y)
    X_train_loader_not_augm, X_test_loader_not_augm, X_train_onehot_not_augm, X_test_onehot_not_augm = data_prep("not_augm", train_cmd_not_augm, train_y_not_augm, test_cmd, test_y)

    # over-sampling of non-augmented train set
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_train_onehot_not_augm_balanced, train_y_not_augm_balanced = ros.fit_resample(X_train_onehot_not_augm, train_y_not_augm)
    X_train_loader_not_augm_balanced = create_dataloader(X_train_onehot_not_augm_balanced, train_y_not_augm_balanced, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

    # =============================================
    # TRAINING
    # =============================================
    train_models("augm", X_train_onehot_augm, X_test_onehot_augm, X_train_loader_augm, X_test_loader_augm, train_y_augm, test_y)
    train_models("not_augm", X_train_onehot_not_augm, X_test_onehot_not_augm, X_train_loader_not_augm, X_test_loader_not_augm, train_y_not_augm, test_y)
    train_models("not_augm_balanced", X_train_onehot_not_augm_balanced, X_test_onehot_not_augm, X_train_loader_not_augm_balanced, X_test_loader_not_augm, train_y_not_augm_balanced, test_y)

    print(f"[!] Script end time: {time.ctime()}")
