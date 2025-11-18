import os

# Configure threading / multiprocessing behaviour early to avoid segfaults on some
# platforms (e.g. Apple Silicon + MPS) without requiring shell-level env vars.
# This mirrors:
#   JOBLIB_MULTIPROCESSING=0 OMP_NUM_THREADS=1 python experiments/mc_repeated_training_gpt5_1.py
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import wordpunct_tokenize
import torch

# Ensure project root is on sys.path so that internal modules using `src.*` imports resolve correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import load_data, load_tokenizer, create_dataloader, commands_to_loader
from src.preprocessors import OneHotCustomVectorizer, CommandTokenizer
from src.tabular_utils import training_tabular
from src.augmentation import (
    NixCommandAugmentationConfig,
    NixCommandAugmentationWithBaseline,
    read_template_file,
)
from src.scoring import get_tpr_at_fpr
from src.models import (
    SimpleMLPWithEmbedding,
    CNN1DGroupedModel,
    BiLSTMModel,
    CNN1D_BiLSTM_Model,
    MeanTransformerEncoder,
    CLSTransformerEncoder,
    AttentionPoolingTransformerEncoder,
    NeurLuxModel,
    SimpleMLP,
)
from src.lit_utils import train_lit_model

NUM_RUNS = 10 # 10 for production run
TIMESTAMP = int(time.time())

# Mirror core hyperparameters from ablation_models.py for consistency
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5
LEARNING_RATE = 1e-3
SCHEDULER = "onecycle"
EPOCHS = 20
DATALOADER_WORKERS = 4


def detect_device() -> str:
    """
    Detect the best available Torch device.

    Priority:
      1. Apple Silicon GPU via MPS
      2. CUDA GPU
      3. CPU
    """
    # Prefer Apple Silicon GPU if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # Fall back to CUDA GPUs
    if torch.cuda.is_available():
        return "gpu"
    # Default to CPU
    return "cpu"


DEVICE = detect_device()

# Avoid spamming device info when Lightning spawns worker processes
if os.getenv("MC_DEVICE_PRINTED", "0") != "1":
    print(f"[MC] Using Torch device: {DEVICE}")
    os.environ["MC_DEVICE_PRINTED"] = "1"


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo repeated training."""

    num_runs: int = NUM_RUNS
    seeds: List[int] = None
    vocab_size: int = 4096
    max_len: int = 128
    logs_folder: Path = Path(f"experiments/logs_mc_repeated_training_{TIMESTAMP}")
    baseline: str = "real"

    def __post_init__(self) -> None:
        if self.seeds is None:
            # Default deterministic seeds for reproducibility
            self.seeds = [33 + i for i in range(self.num_runs)]
        self.logs_folder.mkdir(parents=True, exist_ok=True)


def build_augmentation_presets() -> List[NixCommandAugmentationConfig]:
    """
    Define a small set of sensible augmentation configs close to the default domain knowledge.
    """
    default = NixCommandAugmentationConfig()

    presets: List[NixCommandAugmentationConfig] = [
        # Baseline: exactly the default config
        default,
        # Slightly richer filesystem and IP diversity
        NixCommandAugmentationConfig(
            nix_shells=default.nix_shells,
            nix_shell_folders=default.nix_shell_folders,
            default_filepaths=default.default_filepaths,
            path_roots=["/tmp/", "/home/app/", "/var/www/", "/var/tmp/"],
            folder_lengths=[1, 4, 8],
            nr_of_random_filepaths=8,
            default_variable_names=default.default_variable_names,
            nr_of_random_variables=8,
            default_ips=default.default_ips,
            nr_of_random_ips=8,
            default_ports=default.default_ports,
            nr_of_random_ports=8,
            default_fd_number=default.default_fd_number,
            nr_of_random_fd_numbers=default.nr_of_random_fd_numbers,
        ),
        # Heavier reuse of enterprise-like patterns via more random sampling
        NixCommandAugmentationConfig(
            nix_shells=["sh", "bash", "dash", "zsh"],
            nix_shell_folders=["/bin/", "/usr/bin/", "/usr/local/bin/"],
            default_filepaths=default.default_filepaths,
            path_roots=["/tmp/", "/home/user/", "/srv/", "/opt/"],
            folder_lengths=[1, 3, 8],
            nr_of_random_filepaths=10,
            default_variable_names=default.default_variable_names + ["payload", "session"],
            nr_of_random_variables=10,
            default_ips=default.default_ips,
            nr_of_random_ips=10,
            default_ports=default.default_ports + [445, 3389],
            nr_of_random_ports=10,
            default_fd_number=default.default_fd_number,
            nr_of_random_fd_numbers=default.nr_of_random_fd_numbers,
        ),
    ]
    return presets


def generate_malicious_with_config(
    templates: List[str],
    baseline_commands: List[str],
    config: NixCommandAugmentationConfig,
    seed: int,
    examples_per_template: int,
) -> List[str]:
    """
    Generate malicious commands using NixCommandAugmentationWithBaseline
    under a specific config and seed.
    """
    generator = NixCommandAugmentationWithBaseline(
        templates=templates,
        legitimate_baseline=baseline_commands,
        random_state=seed,
        config=config,
    )
    return generator.generate_commands(examples_per_template)


def monte_carlo_run(mc_cfg: MonteCarloConfig) -> None:
    """
    Run Monte Carlo repeated training with different augmentation presets and seeds.
    """
    root = Path(__file__).parent.parent

    # ===== Fixed components across runs =====
    # 1) Load base train/test splits created from enterprise baselines + one synthetic pass
    X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(
        root=root,
        seed=mc_cfg.seeds[0],
        limit=None,
        baseline=mc_cfg.baseline,
    )

    # 2) Keep test set fixed for all MC runs
    X_test_fixed, y_test_fixed = X_test_cmds, y_test

    # 3) Load templates and enterprise baselines (respecting train/test split)
    data_root = root / "data" / "nix_shell"
    train_template_path = data_root / "templates_train.txt"
    test_template_path = data_root / "templates_test.txt"

    train_templates = read_template_file(train_template_path)
    test_templates = read_template_file(test_template_path)

    enterprise_train = (data_root / "enterprise_baseline_train.parquet")
    enterprise_test = (data_root / "enterprise_baseline_test.parquet")

    import pandas as pd

    train_baseline_df = pd.read_parquet(enterprise_train)
    test_baseline_df = pd.read_parquet(enterprise_test)
    baseline_train_cmds = train_baseline_df["cmd"].tolist()
    baseline_test_cmds = test_baseline_df["cmd"].tolist()

    # 4) One-hot encoder fitted once on an initial training view
    onehot_vectorizer: OneHotCustomVectorizer = load_tokenizer(
        tokenizer_type="onehot",
        train_cmds=X_train_cmds,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        tokenizer_fn=wordpunct_tokenize,
        suffix="_mc_run",
        logs_folder=str(mc_cfg.logs_folder),
    )

    # Sequence tokenizer (for embedding-based architectures)
    seq_tokenizer: CommandTokenizer = load_tokenizer(
        tokenizer_type="seq",
        train_cmds=X_train_cmds,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        tokenizer_fn=wordpunct_tokenize,
        suffix="_mc_seq",
        logs_folder=str(mc_cfg.logs_folder),
    )

    # Fixed test loader for sequence models
    X_test_seq_loader = commands_to_loader(
        X_test_fixed,
        seq_tokenizer,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE,
        y=y_test_fixed,
        shuffle=False,
    )

    presets = build_augmentation_presets()

    results: List[Dict[str, Any]] = []

    for run_idx in range(mc_cfg.num_runs):
        seed = mc_cfg.seeds[run_idx]
        preset = presets[run_idx % len(presets)]

        print(f"\n[MC-RUN] Run {run_idx + 1}/{mc_cfg.num_runs} | seed={seed}")
        print(f"[MC-XGB] Using augmentation preset index {run_idx % len(presets)}")

        # ===== Variable components per run: malicious training synthesis + model seed =====
        # Determine how many synthetic examples per template to generate to roughly
        # match the original malignant class size.
        original_malicious_count = int(y_train.sum())
        templates_count = len(train_templates)
        per_template = max(1, original_malicious_count // templates_count)

        print(
            f"[MC-RUN] Generating malicious training data: "
            f"{per_template} examples/template for {templates_count} templates "
            f"(~{per_template * templates_count} total)."
        )

        malicious_train_cmds = generate_malicious_with_config(
            templates=train_templates,
            baseline_commands=baseline_train_cmds,
            config=preset,
            seed=seed,
            examples_per_template=per_template,
        )

        # Build new training set: fixed enterprise baseline + new synthetic malicious
        X_train_run = baseline_train_cmds + malicious_train_cmds
        y_train_run = np.array(
            [0] * len(baseline_train_cmds) + [1] * len(malicious_train_cmds),
            dtype=np.int8,
        )
        X_train_run, y_train_run = shuffle(X_train_run, y_train_run, random_state=seed)

        # Encode train/test with the fixed one-hot tokenizer
        X_train_onehot = onehot_vectorizer.transform(X_train_run)
        X_test_onehot = onehot_vectorizer.transform(X_test_fixed)

        # Sequence loaders for this run
        X_train_seq_loader = commands_to_loader(
            X_train_run,
            seq_tokenizer,
            workers=DATALOADER_WORKERS,
            batch_size=BATCH_SIZE,
            y=y_train_run,
            shuffle=True,
        )

        # Tabular loaders for MLP (One-Hot)
        X_train_tab_loader = create_dataloader(
            X_train_onehot,
            y_train_run,
            batch_size=BATCH_SIZE,
            shuffle=True,
            workers=DATALOADER_WORKERS,
        )
        X_test_tab_loader = create_dataloader(
            X_test_onehot,
            y_test_fixed,
            batch_size=BATCH_SIZE,
            shuffle=False,
            workers=DATALOADER_WORKERS,
        )

        # =============================================
        # DEFINING MODELS (mirroring ablation_models.py)
        # =============================================
        mlp_seq_model = SimpleMLPWithEmbedding(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDED_DIM,
            output_dim=1,
            hidden_dim=[256, 64, 32],
            use_positional_encoding=False,
            max_len=MAX_LEN,
            dropout=DROPOUT,
        )
        cnn_model = CNN1DGroupedModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            num_channels=32,
            kernel_sizes=[2, 3, 4, 5],
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
        lstm_model = BiLSTMModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            hidden_dim=32,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
        cnn_lstm_model = CNN1D_BiLSTM_Model(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            num_channels=32,
            kernel_size=3,
            lstm_hidden_dim=32,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
        mean_transformer_model = MeanTransformerEncoder(
            vocab_size=VOCAB_SIZE,
            d_model=EMBEDDED_DIM,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_len=MAX_LEN,
            dropout=DROPOUT,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
        )
        cls_transformer_model = CLSTransformerEncoder(
            vocab_size=VOCAB_SIZE,
            d_model=EMBEDDED_DIM,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_len=MAX_LEN,
            dropout=DROPOUT,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
        )
        attpool_transformer_model = AttentionPoolingTransformerEncoder(
            vocab_size=VOCAB_SIZE,
            d_model=EMBEDDED_DIM,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_len=MAX_LEN,
            dropout=DROPOUT,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
        )
        neurlux_model = NeurLuxModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            max_len=MAX_LEN,
            hidden_dim=32,
            output_dim=1,
            dropout=DROPOUT,
        )

        rf_model_onehot = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
        )
        xgb_model_onehot = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            tree_method="hist",
        )
        mlp_tab_model_onehot = SimpleMLP(
            input_dim=VOCAB_SIZE,
            output_dim=1,
            hidden_dim=[64, 32],
            dropout=DROPOUT,
        )

        models = {
            "_tabular_rf_onehot": rf_model_onehot,
            "_tabular_xgb_onehot": xgb_model_onehot,
            "_tabular_mlp_onehot": mlp_tab_model_onehot,
            "mlp_seq": mlp_seq_model,
            "attpool_transformer": attpool_transformer_model,
            "cls_transformer": cls_transformer_model,
            "mean_transformer": mean_transformer_model,
            "neurlux": neurlux_model,
            "cnn": cnn_model,
            "lstm": lstm_model,
            "cnn_lstm": cnn_lstm_model,
        }

        # Train and evaluate all models for this run
        for model_name, model in models.items():
            full_name = f"{model_name}_mc_run_{run_idx}_seed_{seed}"
            print(f"\n[MC] Training '{full_name}'...")
            start = time.time()

            if model_name.startswith("_tabular") and "mlp" not in model_name:
                # Tabular RF/XGB: use training_tabular on one-hot features
                trained_model = training_tabular(
                    model=model,
                    name=full_name,
                    X_train_encoded=X_train_onehot,
                    X_test_encoded=X_test_onehot,
                    y_train=y_train_run,
                    y_test=y_test_fixed,
                    logs_folder=str(mc_cfg.logs_folder),
                )
                elapsed = time.time() - start
                y_test_scores = trained_model.predict_proba(X_test_onehot)[:, 1]
            elif model_name.startswith("_tabular") and "mlp" in model_name:
                # Tabular MLP: Lightning over one-hot loaders
                _, lightning_model = train_lit_model(
                    X_train_tab_loader,
                    X_test_tab_loader,
                    model,
                    full_name,
                    log_folder=str(mc_cfg.logs_folder),
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    scheduler_budget=EPOCHS * len(X_train_tab_loader),
                    device=DEVICE,
                    lit_sanity_steps=1,
                    early_stop_patience=5,
                    val_check_times=2,
                )
                elapsed = time.time() - start
                # Predict with Lightning
                import lightning as L
                if DEVICE == "gpu":
                    accelerator = "gpu"
                elif DEVICE == "mps":
                    accelerator = "mps"
                else:
                    accelerator = "cpu"
                trainer = L.Trainer(
                    accelerator=accelerator,
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                )
                preds = trainer.predict(lightning_model, X_test_tab_loader)
                import torch

                y_test_scores = torch.sigmoid(torch.vstack(preds).squeeze()).cpu().numpy()
            else:
                # Sequence models: Lightning over sequence loaders
                _, lightning_model = train_lit_model(
                    X_train_seq_loader,
                    X_test_seq_loader,
                    model,
                    full_name,
                    log_folder=str(mc_cfg.logs_folder),
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    scheduler=SCHEDULER,
                    scheduler_budget=EPOCHS * len(X_train_seq_loader),
                    device=DEVICE,
                    lit_sanity_steps=1,
                    early_stop_patience=5,
                    val_check_times=2,
                )
                elapsed = time.time() - start
                import lightning as L
                import torch
                if DEVICE == "gpu":
                    accelerator = "gpu"
                elif DEVICE == "mps":
                    accelerator = "mps"
                else:
                    accelerator = "cpu"
                trainer = L.Trainer(
                    accelerator=accelerator,
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                )
                preds = trainer.predict(lightning_model, X_test_seq_loader)
                y_test_scores = torch.sigmoid(torch.vstack(preds).squeeze()).cpu().numpy()

            y_test_pred = (y_test_scores >= 0.5).astype(int)

            tpr_at_1e6 = get_tpr_at_fpr(
                y_test_fixed,
                y_test_scores,
                fprNeeded=1e-6,
                logits=False,
            )

            run_result = {
                "run_idx": run_idx,
                "seed": seed,
                "preset_index": run_idx % len(presets),
                "model_name": model_name,
                "test_auc": roc_auc_score(y_test_fixed, y_test_scores),
                "test_f1": f1_score(y_test_fixed, y_test_pred),
                "test_acc": accuracy_score(y_test_fixed, y_test_pred),
                "tpr_at_1e6": tpr_at_1e6,
                "elapsed_sec": elapsed,
            }
            results.append(run_result)

    # Persist MC summary for quick inspection
    import pandas as pd

    results_df = pd.DataFrame(results)
    summary_path = mc_cfg.logs_folder / "mc_models_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"[MC-RUN] Monte Carlo summary saved to '{summary_path}'")

    # Print aggregate metrics (mean ± std) across runs in a paper-ready style.
    print("\n[MC-RUN] Aggregate metrics across runs (values in % where applicable):")
    metric_columns = ["test_auc", "test_f1", "test_acc", "tpr_at_1e6"]
    for model_name in sorted(results_df["model_name"].unique()):
        print(f"\n  Model: {model_name}")
        model_df = results_df[results_df["model_name"] == model_name]
        for col in metric_columns:
            mean_val = 100.0 * model_df[col].mean()
            std_val = 100.0 * model_df[col].std()
            print(f"    {col}: {mean_val:.2f} ± {std_val:.2f}")

        if "elapsed_sec" in model_df.columns:
            mean_time = model_df["elapsed_sec"].mean()
            std_time = model_df["elapsed_sec"].std()
            print(f"    elapsed_sec: {mean_time:.2f} ± {std_time:.2f} s")


if __name__ == "__main__":
    cfg = MonteCarloConfig()
    monte_carlo_run(cfg)
