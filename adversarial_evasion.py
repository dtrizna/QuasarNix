import os
import time
import json
import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, f1_score, accuracy_score, roc_auc_score
from watermark import watermark
from typing import List
from shutil import copyfile

# tokenizers
from nltk.tokenize import wordpunct_tokenize, WhitespaceTokenizer
whitespace_tokenize = WhitespaceTokenizer().tokenize

# modeling
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.lite.utilities.seed import seed_everything

from src.models import SimpleMLPWithEmbedding, CNN1DGroupedModel, MeanTransformerEncoder, SimpleMLP, PyTorchLightningModel
from src.lit_utils import LitProgressBar
from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.data_utils import create_dataloader, commands_to_loader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_tpr_at_fpr(predicted_logits, true_labels, fprNeeded=1e-4):
    if isinstance(predicted_logits, torch.Tensor):
        predicted_probs = torch.sigmoid(predicted_logits).cpu().detach().numpy()
    else:
        predicted_probs = sigmoid(predicted_logits)
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    if all(np.isnan(fpr)):
        return np.nan#, np.nan
    else:
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return tpr_at_fpr#, threshold_at_fpr


def training_tabular(model, name, X_train_minhash, X_test_minhash, y_train, y_test, logs_folder):
    print(f"[*] Training {name} model...")
    model.fit(X_train_minhash, y_train)

    # save trained model to LOGS_FOLDER/name
    os.makedirs(f"{logs_folder}/{name}", exist_ok=True)
    with open(f"{logs_folder}/{name}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    y_test_preds = model.predict_proba(X_test_minhash)[:,1]

    tpr = get_tpr_at_fpr(y_test_preds, y_test)
    f1 = f1_score(y_test, y_test_preds.round())
    acc = accuracy_score(y_test, y_test_preds.round())
    auc = roc_auc_score(y_test, y_test_preds)
    
    print(f"[!] {name} model scores: tpr={tpr:.4f}, f1={f1:.4f}, acc={acc:.4f}, auc={auc:.4f}")
    
    metrics_csv = pd.DataFrame({"tpr": [tpr], "f1": [f1], "acc": [acc], "auc": [auc]})
    with open(f"{logs_folder}/{name}/metrics.csv", "w") as f:
        metrics_csv.to_csv(f, index=False)
    

def configure_trainer(name, log_folder, epochs, val_check_times=2):
    """Configure the PyTorch Lightning Trainer."""

    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=10,
        min_delta=0.0001,
        verbose=True,
        mode="max"
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_tpr",
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}"
    )

    trainer = L.Trainer(
        num_sanity_val_steps=LIT_SANITY_STEPS,
        max_epochs=epochs,
        accelerator=DEVICE,
        devices=1,
        callbacks=[
            LitProgressBar(),
            early_stop,
            model_checkpoint
        ],
        val_check_interval=1/val_check_times,
        log_every_n_steps=10,
        logger=[
            CSVLogger(save_dir=log_folder, name=f"{name}_csv"),
            TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb")
        ]
    )

    # Ensure folders for logging exist
    os.makedirs(os.path.join(log_folder, f"{name}_tb"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, f"{name}_csv"), exist_ok=True)

    return trainer


def load_lit_model(model_file, pytorch_model, name, log_folder, epochs):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs)
    return trainer, lightning_model


def train_lit_model(X_train_loader, X_test_loader, pytorch_model, name, log_folder, epochs=10, learning_rate=1e-3, scheduler=None, scheduler_budget=None):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs)

    print(f"[*] Training '{name}' model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    return trainer, lightning_model


def predict(loader, trainer, lightning_model, decision_threshold=0.5, dump_logits=False):
    """Get scores out of a loader."""
    y_pred_logits = trainer.predict(model=lightning_model, dataloaders=loader)
    y_pred = torch.sigmoid(torch.cat(y_pred_logits, dim=0)).numpy()
    y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
    if dump_logits:
        assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
        pickle.dump(y_pred_logits, open(dump_logits, "wb"))
    return y_pred


def load_data():
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
        X_train_baseline_cmd = shuffle(train_baseline_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_train_malicious_cmd = shuffle(train_malicious_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_test_baseline_cmd = shuffle(test_baseline_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
        X_test_malicious_cmd = shuffle(test_malicious_df['cmd'].values.tolist(), random_state=SEED)[:LIMIT//2]
    else:
        X_train_baseline_cmd = train_baseline_df['cmd'].values.tolist()
        X_train_malicious_cmd = train_malicious_df['cmd'].values.tolist()
        X_test_baseline_cmd = test_baseline_df['cmd'].values.tolist()
        X_test_malicious_cmd = test_malicious_df['cmd'].values.tolist()

    X_train_non_shuffled = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=SEED)

    X_test_non_shuffled = X_test_baseline_cmd + X_test_malicious_cmd
    y_test = np.array([0] * len(X_test_baseline_cmd) + [1] * len(X_test_malicious_cmd), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=SEED)

    return X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd


def load_nl2bash():
    with open(r"data\nl2bash.cm", "r", encoding="utf-8") as f:
        baseline = f.readlines()
    return baseline


def attack_template_prepend(
        command: str,
        baseline: List[str],
        payload_size: int,
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
    while len(payload) < payload_size:
        payload += np.random.choice(baseline) + ";"
    
    payload = payload[:payload_size]
    payload = template.replace("PAYLOAD", payload)
    
    return payload + ";" + command


BASELINE = load_nl2bash()
ATTACK = attack_template_prepend

MAX_LEN = 256 
# NOTE: increased max len 128 -> 256 if compared to model architecture tests
# Therefore, needed to reduce batch size to 512 so transformer fits on GPU

PAYLOAD_SIZES = [16, 32, 48, 64, 80, 96, 112, 128]
# NOTE: Total size of injected characters: PAYLOAD_SIZE + 23
# since len(template) = 23 (w/o PAYLOAD) when template = """python3 -c "print('PAYLOAD')" """

SEED = 33

VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
BATCH_SIZE = 256
DROPOUT = 0.5

# TEST
# ADV_ATTACK_SUBSAMPLE = 50
# EPOCHS = 2
# LIMIT = None

# PROD
ADV_ATTACK_SUBSAMPLE = 5000
EPOCHS = 20
LIMIT = None

DEVICE = "gpu"
LIT_SANITY_STEPS = 1
DATALOADER_WORKERS = 4
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"

PREFIX = "TEST_" if LIMIT is not None else ""
LOGS_FOLDER = f"{PREFIX}logs_adversarial_evasion_nl2bash"
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

    ROOT = os.path.dirname(os.path.abspath(__file__))
    X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd = load_data()
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
    for payload_size  in PAYLOAD_SIZES:
        adv_file = os.path.join(LOGS_FOLDER, f"X_test_malicious_with_attack_{payload_size}_cmd_sample_{ADV_ATTACK_SUBSAMPLE}.json")
        if os.path.exists(adv_file):
            print(f"[!] Loading adversarial test set for payload size {payload_size} from '{adv_file}'")
            with open(adv_file, "r", encoding="utf-8") as f:
                X_test_malicious_with_attack_cmd = json.load(f)
        else:
            print(f"[*] Constructing adversarial test set with attack's payload size {payload_size}...")
            X_test_malicious_with_attack_cmd = []
            for cmd in tqdm(X_test_malicious_without_attack_cmd):
                cmd_a = ATTACK(cmd, BASELINE, payload_size=payload_size)
                X_test_malicious_with_attack_cmd.append(cmd_a)
            # dump as json
            with open(adv_file, "w", encoding="utf-8") as f:
                json.dump(X_test_malicious_with_attack_cmd, f, indent=4)

        X_test_malicious_with_attack_cmd_dict[payload_size] = X_test_malicious_with_attack_cmd

    # =============================================
    # DEFINING MODELS
    # =============================================

    mlp_seq_model = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    mlp_seq_model_adv = SimpleMLPWithEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDED_DIM, output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False, max_len=MAX_LEN, dropout=DROPOUT) # 297 K params
    
    cnn_model = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    cnn_model_adv = CNN1DGroupedModel(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDED_DIM, num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32], output_dim=1, dropout=DROPOUT) # 301 K params
    
    mean_transformer_model = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    mean_transformer_model_adv = MeanTransformerEncoder(vocab_size=VOCAB_SIZE, d_model=EMBEDDED_DIM, nhead=4, num_layers=2, dim_feedforward=128, max_len=MAX_LEN, dropout=DROPOUT, mlp_hidden_dims=[64,32], output_dim=1) # 335 K params
    
    mlp_tab_model_onehot = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    mlp_tab_model_onehot_adv = SimpleMLP(input_dim=VOCAB_SIZE, output_dim=1, hidden_dim=[64, 32], dropout=DROPOUT) # 264 K params
    
    target_models = {
        # "cnn": (cnn_model, cnn_model_adv),
        # "mlp_onehot": (mlp_tab_model_onehot, mlp_tab_model_onehot_adv),
        "mean_transformer": (mean_transformer_model, mean_transformer_model_adv),
        #"mlp_seq": (mlp_seq_model, mlp_seq_model_adv),
    }

    for name, (target_model_orig, target_model_adv) in target_models.items():
        print(f"[*] Working on attack against '{name}' model...")

        # =============================================
        # PREPING DATA
        # =============================================

        if name == "mlp_onehot":
            # # ========== ONE-HOT TABULAR ENCODING ===========
            oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
            print("[*] Fitting one-hot encoder...")
            now = time.time()
            X_train_onehot = oh.fit_transform(X_train_cmds)
            X_test_onehot = oh.transform(X_test_cmds)
            print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s

            X_train_loader = create_dataloader(X_train_onehot, y=y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)
            X_test_loader = create_dataloader(X_test_onehot, y=y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS)

        else:
            # ========== EMBEDDING ==========
            tokenizer = CommandTokenizer(tokenizer_fn=TOKENIZER, vocab_size=VOCAB_SIZE)
            vocab_file = os.path.join(LOGS_FOLDER, f"wordpunct_vocab_{VOCAB_SIZE}.json")
            if os.path.exists(vocab_file):
                tokenizer.load_vocab(vocab_file)
            else:
                print("[*] Building vocab and encoding...")
                X_train_tokens = tokenizer.tokenize(X_train_cmds)
                tokenizer.build_vocab(X_train_tokens)
                tokenizer.dump_vocab(vocab_file)

            # creating dataloaders
            X_train_loader = commands_to_loader(X_train_cmds, tokenizer, y=y_train, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS, max_len=MAX_LEN)
            X_test_loader = commands_to_loader(X_test_cmds, tokenizer, y=y_test, batch_size=BATCH_SIZE, workers=DATALOADER_WORKERS, max_len=MAX_LEN)

        # =============================================
        # ORIGINAL MODEL TRAINING
        # =============================================

        run_name = f"{name}_orig"
        model_file_orig = os.path.join(LOGS_FOLDER, f"{run_name}.ckpt")
        if os.path.exists(model_file_orig):
            print(f"[*] Loading original model from {model_file_orig}...")
            trainer_orig, lightning_model_orig = load_lit_model(model_file_orig, target_model_orig, run_name, LOGS_FOLDER, EPOCHS)
        else:
            print(f"[!] Training original model '{run_name}' started: {time.ctime()}")
            now = time.time()
            trainer_orig, lightning_model_orig = train_lit_model(
                X_train_loader,
                X_test_loader,
                target_model_orig,
                run_name,
                log_folder=LOGS_FOLDER,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                scheduler=SCHEDULER,
                scheduler_budget= EPOCHS * len(X_train_loader)
            )
            # copy best checkpoint to the LOGS_DIR for further tests
            checkpoint_path = os.path.join(LOGS_FOLDER, run_name, "version_0", "checkpoints")
            best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
            best_checkpoint_path = os.path.join(checkpoint_path, best_checkpoint_name)
            copyfile(best_checkpoint_path, model_file_orig)

            print(f"[!] Training of {run_name} ended: ", time.ctime(), f" | Took: {time.time() - now:.2f} seconds")

        # ======================================================
        # ORIG MODEL SCORES ON TEST SET W/O ATTACK
        # ======================================================

        if name == "mlp_onehot":
            X_test_malicious_without_attack_onehot = oh.transform(X_test_malicious_without_attack_cmd)
            X_test_malicious_without_attack_loader = create_dataloader(X_test_malicious_without_attack_onehot, batch_size=BATCH_SIZE, workers=1)
        else:
            X_test_malicious_without_attack_loader = commands_to_loader(X_test_malicious_without_attack_cmd, tokenizer, batch_size=BATCH_SIZE, workers=1, max_len=MAX_LEN)
        
        # =======================================================
        # ORIG MODEL ADVERSARIAL SCORES
        # =======================================================

        accuracies = {}
        evasives = {}

        y_pred_orig_orig = predict(
            X_test_malicious_without_attack_loader,
            trainer_orig,
            lightning_model_orig,
            decision_threshold=0.5
        )
        evasive = len(y_pred_orig_orig[y_pred_orig_orig == 0])
        print(f"[!] Orig train | Orig test |  Evasive:", evasive)
        evasives[0] = evasive

        acc = accuracy_score(np.ones_like(y_pred_orig_orig), y_pred_orig_orig)
        print(f"[!] Orig train | Orig test | Accuracy: {acc:.3f}")
        accuracies[0] = acc

        for payload_size, X_test_malicious_with_attack_cmd in X_test_malicious_with_attack_cmd_dict.items():
            if name == "mlp_onehot":
                X_test_onehot_malicious_adv = oh.transform(X_test_malicious_with_attack_cmd)
                X_test_loader_malicious_adv = create_dataloader(X_test_onehot_malicious_adv, batch_size=BATCH_SIZE, workers=1)
            else:
                X_test_loader_malicious_adv = commands_to_loader(X_test_malicious_with_attack_cmd, tokenizer, batch_size=BATCH_SIZE, workers=1, max_len=MAX_LEN)

            y_pred_orig_adv = predict(
                X_test_loader_malicious_adv,
                trainer_orig,
                lightning_model_orig,
                decision_threshold=0.5,
                dump_logits=os.path.join(LOGS_FOLDER, f"{run_name}_y_pred_with_attack_logits_payload_{payload_size}_sample_{ADV_ATTACK_SUBSAMPLE}.pkl")
            )
            evasive = len(y_pred_orig_adv[y_pred_orig_adv == 0])
            print(f"[!] Orig train | Adv test | Payload {payload_size} |  Evasive:" , evasive)
            evasives[payload_size] = evasive

            acc = accuracy_score(np.ones_like(y_pred_orig_adv), y_pred_orig_adv)
            print(f"[!] Orig train | Adv test | Payload {payload_size} | Accuracy: {acc:.3f}")
            accuracies[payload_size] = acc
        
        accuracies = {int(key): value for key, value in accuracies.items()}
        evasives = {int(key): value for key, value in evasives.items()}
        results_dict = {'Accuracy': accuracies, 'Evasive Samples': evasives}
        results_json = json.dumps(results_dict, indent=4)
        with open(os.path.join(LOGS_FOLDER, f"adversarial_scores_{run_name}.json"), 'w') as f:
            f.write(results_json)

    # ===== END LOOP ======
    print(f"[!] Script end time: {time.ctime()}")