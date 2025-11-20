import os

# Configure threading / multiprocessing behaviour early to avoid segfaults on some
# platforms (e.g. Apple Silicon + MPS) without requiring shell-level env vars.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from nltk.tokenize import wordpunct_tokenize
import torch
from xgboost import XGBClassifier

# Ensure project root is on sys.path so that internal modules using `src.*` imports resolve correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import load_data, load_tokenizer, commands_to_loader, create_dataloader
from src.preprocessors import OneHotCustomVectorizer, CommandTokenizer
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
from src.lit_utils import PyTorchLightningModel


# Mirror core hyperparameters from mc_repeated_training.py for consistency
VOCAB_SIZE = 4096
EMBEDDED_DIM = 64
MAX_LEN = 128
BATCH_SIZE = 1024
DROPOUT = 0.5
DATALOADER_WORKERS = 4

LOGS_FOLDER = ROOT / "experiments" / "logs_mc_repeated_training_1763474762"


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


def _get_scores_path(run_idx: int, seed: int, model_name: str) -> Path:
    """
    Location for cached prediction scores for a given (run, seed, model).

    We cache the model outputs *before thresholding* (probabilities or
    probability-like scores) so we can recompute metrics at new FPRs without
    re-running inference.
    """
    filename = f"scores_{model_name}_mc_run_{run_idx}_seed_{seed}.npy"
    return LOGS_FOLDER / filename


def _find_lightning_checkpoint(full_name: str) -> Path | None:
    """
    Locate a Lightning checkpoint (prefer 'last.ckpt') for a given model/run.
    """
    csv_root = LOGS_FOLDER / f"{full_name}_csv"
    if not csv_root.is_dir():
        return None

    version_dirs = [
        d for d in csv_root.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    if not version_dirs:
        return None

    try:
        latest = max(version_dirs, key=lambda d: int(d.name.split("_")[-1]))
    except Exception:
        latest = sorted(version_dirs)[-1]

    ckpt_dir = latest / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    ckpts = list(ckpt_dir.glob("*.ckpt"))
    return ckpts[0] if ckpts else None


def _build_base_model(model_name: str) -> torch.nn.Module:
    """Recreate base PyTorch model architectures used in mc_repeated_training.py."""
    if model_name == "mlp_seq":
        return SimpleMLPWithEmbedding(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDED_DIM,
            output_dim=1,
            hidden_dim=[256, 64, 32],
            use_positional_encoding=False,
            max_len=MAX_LEN,
            dropout=DROPOUT,
        )
    if model_name == "cnn":
        return CNN1DGroupedModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            num_channels=32,
            kernel_sizes=[2, 3, 4, 5],
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
    if model_name == "lstm":
        return BiLSTMModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            hidden_dim=32,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
    if model_name == "cnn_lstm":
        return CNN1D_BiLSTM_Model(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            num_channels=32,
            kernel_size=3,
            lstm_hidden_dim=32,
            mlp_hidden_dims=[64, 32],
            output_dim=1,
            dropout=DROPOUT,
        )
    if model_name == "mean_transformer":
        return MeanTransformerEncoder(
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
    if model_name == "cls_transformer":
        return CLSTransformerEncoder(
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
    if model_name == "attpool_transformer":
        return AttentionPoolingTransformerEncoder(
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
    if model_name == "neurlux":
        return NeurLuxModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDED_DIM,
            max_len=MAX_LEN,
            hidden_dim=32,
            output_dim=1,
            dropout=DROPOUT,
        )
    if model_name == "_tabular_mlp_onehot":
        return SimpleMLP(
            input_dim=VOCAB_SIZE,
            output_dim=1,
            hidden_dim=[64, 32],
            dropout=DROPOUT,
        )
    raise ValueError(f"Unknown model_name '{model_name}' for Lightning model reconstruction.")


def _evaluate_tabular_model(
    full_name: str,
    X_test_onehot,
) -> np.ndarray:
    """
    Load a previously trained tabular model (RF/XGB) from disk and return its test scores.
    """
    model_dir = LOGS_FOLDER / full_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Tabular model directory '{model_dir}' not found.")

    pkl_path = model_dir / "model.pkl"
    json_path = model_dir / "model.json"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            trained_model = pickle.load(f)
    elif json_path.exists():
        trained_model = XGBClassifier()
        trained_model.load_model(str(json_path))
    else:
        raise FileNotFoundError(f"No model artifact found in '{model_dir}'.")

    y_test_scores = trained_model.predict_proba(X_test_onehot)[:, 1]
    return y_test_scores


def _evaluate_lightning_model(
    full_name: str,
    model_name: str,
    test_loader,
) -> np.ndarray:
    """
    Load a previously trained Lightning model from checkpoint and return its test scores.
    """
    ckpt_path = _find_lightning_checkpoint(full_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for '{full_name}'.")

    try:
        import lightning as L  # imported lazily to avoid side effects at module import time
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Lightning import failed while evaluating '{full_name}': {exc}") from exc

    base_model = _build_base_model(model_name)
    lightning_model = PyTorchLightningModel.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=base_model,
    )

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
    preds = trainer.predict(lightning_model, test_loader)
    y_test_scores = torch.sigmoid(torch.vstack(preds).squeeze()).cpu().numpy()
    return y_test_scores


def recompute_mc_summary() -> None:
    """
    Recompute Monte Carlo summary metrics (including extra TPRs at 1e-4 and 1e-5)
    for all models logged in `logs_mc_repeated_training_1763474762`.
    """
    summary_path = LOGS_FOLDER / "mc_models_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV '{summary_path}' not found.")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        print(f"[MC-RES] Existing summary '{summary_path}' is empty; nothing to recompute.")
        return

    # Recreate the same test split used during mc_repeated_training.py
    base_seed = int(summary_df["seed"].min())
    X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(
        root=ROOT,
        seed=base_seed,
        limit=None,
        baseline="real",
    )

    # Recreate tokenizers using the same logs folder and suffixes
    onehot_vectorizer: OneHotCustomVectorizer = load_tokenizer(
        tokenizer_type="onehot",
        train_cmds=X_train_cmds,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        tokenizer_fn=wordpunct_tokenize,
        suffix="_mc_run",
        logs_folder=str(LOGS_FOLDER),
    )

    seq_tokenizer: CommandTokenizer = load_tokenizer(
        tokenizer_type="seq",
        train_cmds=X_train_cmds,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        tokenizer_fn=wordpunct_tokenize,
        suffix="_mc_seq",
        logs_folder=str(LOGS_FOLDER),
    )

    # Prepare test encodings/loaders
    X_test_onehot = onehot_vectorizer.transform(X_test_cmds)

    X_test_tab_loader = create_dataloader(
        X_test_onehot,
        y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=DATALOADER_WORKERS,
    )

    X_test_seq_loader = commands_to_loader(
        X_test_cmds,
        seq_tokenizer,
        workers=DATALOADER_WORKERS,
        batch_size=BATCH_SIZE,
        y=y_test,
        shuffle=False,
    )

    results: List[dict] = []

    for _, row in summary_df.iterrows():
        run_idx = int(row["run_idx"])
        seed = int(row["seed"])
        preset_index = int(row.get("preset_index", run_idx % 3))
        model_name = str(row["model_name"])
        elapsed_sec = float(row.get("elapsed_sec", np.nan))

        full_name = f"{model_name}_mc_run_{run_idx}_seed_{seed}"
        scores_path = _get_scores_path(run_idx, seed, model_name)

        if scores_path.exists():
            print(f"[MC-RES] Loading cached logits for '{full_name}' from '{scores_path.name}'...")
            y_test_scores = np.load(scores_path)
        else:
            print(f"[MC-RES] Evaluating '{full_name}'...")

            if model_name.startswith("_tabular") and "mlp" not in model_name:
                # RF / XGB on one-hot features
                y_test_scores = _evaluate_tabular_model(full_name, X_test_onehot)
            elif model_name.startswith("_tabular") and "mlp" in model_name:
                # Tabular MLP via Lightning on one-hot loader
                y_test_scores = _evaluate_lightning_model(
                    full_name,
                    model_name,
                    X_test_tab_loader,
                )
            else:
                # Sequence models via Lightning on sequence loader
                y_test_scores = _evaluate_lightning_model(
                    full_name,
                    model_name,
                    X_test_seq_loader,
                )

            np.save(scores_path, y_test_scores)

        y_test_pred = (y_test_scores >= 0.5).astype(int)

        test_auc = roc_auc_score(y_test, y_test_scores)
        test_f1 = f1_score(y_test, y_test_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        tpr_at_1e4 = get_tpr_at_fpr(
            y_test,
            y_test_scores,
            fprNeeded=1e-4,
            logits=False,
        )
        tpr_at_1e5 = get_tpr_at_fpr(
            y_test,
            y_test_scores,
            fprNeeded=1e-5,
            logits=False,
        )
        tpr_at_1e6 = get_tpr_at_fpr(
            y_test,
            y_test_scores,
            fprNeeded=1e-6,
            logits=False,
        )

        run_result = {
            "run_idx": run_idx,
            "seed": seed,
            "preset_index": preset_index,
            "model_name": model_name,
            "test_auc": test_auc,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "tpr_at_1e4": tpr_at_1e4,
            "tpr_at_1e5": tpr_at_1e5,
            "tpr_at_1e6": tpr_at_1e6,
            "elapsed_sec": elapsed_sec,
        }
        results.append(run_result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(summary_path, index=False)
    print(f"[MC-RES] Recomputed Monte Carlo summary saved to '{summary_path}'")

    # Print aggregate metrics (mean ± std) across runs in a paper-ready style.
    print("\n[MC-RES] Aggregate metrics across runs (values in % where applicable):")
    metric_columns = ["test_auc", "test_f1", "test_acc", "tpr_at_1e4", "tpr_at_1e5", "tpr_at_1e6"]
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
    recompute_mc_summary()


