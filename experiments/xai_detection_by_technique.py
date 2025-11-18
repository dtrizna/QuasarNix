#!/usr/bin/env python3
"""
Per-template and per-technique detection analysis for reverse-shell templates.

This experiment trains an XGBoost model on synthesized commands with template
metadata, then summarizes detection performance:
- per individual template; and
- aggregated by primary binary (bash, python, perl, php, nc, socat, telnet, ...)
  and by higher-level technique family (scripting, netcat, file_descriptor, ...).

The goal is to support statements such as:
  "netcat-style shells (nc/socat) are detected at ~X% TPR,
   file-descriptor shells (/dev/tcp) at ~Y%,
   scripting shells (python/php/perl/ruby) at ~Z%."
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # type: ignore

plt.style.use(["science", "no-latex"])

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.augmentation import NixCommandAugmentationWithBaseline, read_template_file
from src.preprocessors import OneHotCustomVectorizer
from src.tabular_utils import training_tabular
from src.data_utils import load_data
from src.template_detection_analysis import (
    aggregate_detection_by_primary_binary,
    aggregate_detection_by_technique_family,
    detection_by_template,
)

# ============================================================================
# Configuration
# ============================================================================

SEED = 33
VOCAB_SIZE = 4096
LIMIT = 30000  # samples per class

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "nix_shell"
TEMPLATE_TRAIN_PATH = DATA_ROOT / "templates_train.txt"
TEMPLATE_TEST_PATH = DATA_ROOT / "templates_test.txt"

TOKENIZER = wordpunct_tokenize
TIMESTAMP = int(time.time())

OUT_DIR = ROOT / "experiments" / f"logs_xai_detection_by_technique_{TIMESTAMP}"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11


def get_threshold_at_fpr(y_true: np.ndarray, y_pred_proba: np.ndarray, target_fpr: float) -> float:
    """
    Find threshold that achieves target FPR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 1.0  # No threshold achieves this FPR
    return float(thresholds[valid_idx[-1]])


def get_threshold_with_highest_f1(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Find threshold that maximizes F1 score using ROC curve approximation.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # F1 approximation: 2 * TPR * (1 - FPR) / (TPR + 1 - FPR)
    # Simplified for ROC: 2 * tpr * (1 - fpr)
    f1_scores = 2 * tpr * (1 - fpr)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def build_attack_component_groups() -> Dict[str, List[str]]:
    """
    Attack-component (tactic) grouping aligned with explainability_commands_gpt.

    These components are originally defined at token level; here we approximate
    membership via substring checks on raw command strings.
    """
    return {
        "interpreter": [
            "python",
            "python2",
            "python3",
            "perl",
            "php",
            "ruby",
            "node",
            "nodejs",
        ],
        "shell_invoker": ["bash", "sh", "zsh", "ksh"],
        "net_utility": ["nc", "ncat", "socat", "telnet", "openssl", "curl", "wget", "ssh", "nmap"],
        "fd_redirection": [">", ">&", "2>&1", "0<&", "1>&", "/dev", "tcp", "udp"],
        "wrappers": ["sudo", "nohup", "setsid", "timeout", "time"],
        "obfuscation": ["base64", "eval", "exec", "xxd", "rev", "tr", "sed", "awk", "printf"],
        "ip_tokens": [".", "10", "127"],
    }


def load_data_with_metadata(
    root: Path,
    seed: int = 33,
    limit: int | None = None,
) -> Dict[str, object]:
    """
    Load training/test data with template metadata for malicious samples.

    Returns dict with:
        - X_train, y_train: training data
        - X_test, y_test: test data
        - test_template_ids: template index for each test sample (None for benign)
        - test_templates: list of unique test templates
    """
    print("[+] Loading baseline data...")
    paths = {
        "train_baseline": root / "data" / "nix_shell" / "train_baseline_real.parquet",
        "test_baseline": root / "data" / "nix_shell" / "test_baseline_real.parquet",
    }

    train_baseline_df = pd.read_parquet(paths["train_baseline"])
    test_baseline_df = pd.read_parquet(paths["test_baseline"])

    train_baseline = train_baseline_df["cmd"].tolist()
    test_baseline = test_baseline_df["cmd"].tolist()

    if limit:
        train_baseline = train_baseline[:limit]
        test_baseline = test_baseline[:limit]

    print("[+] Loading templates...")
    train_templates = read_template_file(TEMPLATE_TRAIN_PATH)
    test_templates = read_template_file(TEMPLATE_TEST_PATH)

    print(f"[+] Train templates: {len(train_templates)}")
    print(f"[+] Test templates: {len(test_templates)}")

    # Training malicious data (no metadata for efficiency)
    print("[+] Generating training malicious data...")
    train_gen = NixCommandAugmentationWithBaseline(
        templates=train_templates,
        legitimate_baseline=train_baseline,
        random_state=seed,
    )
    train_per_template = max(1, len(train_baseline) // len(train_templates))
    X_train_malicious = train_gen.generate_commands(train_per_template)

    # Test malicious data with per-template metadata
    print("[+] Generating test malicious data with metadata...")
    test_per_template = max(1, len(test_baseline) // len(test_templates))

    commands: List[str] = []
    template_ids: List[int] = []

    for template_idx, template in enumerate(test_templates):
        temp_gen = NixCommandAugmentationWithBaseline(
            templates=[template],
            legitimate_baseline=test_baseline,
            random_state=seed + 1 + template_idx,
        )
        template_commands = temp_gen.generate_commands(test_per_template)
        commands.extend(template_commands)
        template_ids.extend([template_idx] * len(template_commands))

    X_test_malicious = commands

    # Combine training data
    X_train = train_baseline + X_train_malicious
    y_train = np.array(
        [0] * len(train_baseline) + [1] * len(X_train_malicious),
        dtype=np.int8,
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    # Combine test data
    X_test = test_baseline + X_test_malicious
    y_test = np.array(
        [0] * len(test_baseline) + [1] * len(X_test_malicious),
        dtype=np.int8,
    )

    # Template ids for test set (None for benign)
    test_template_ids_full: List[int | None] = [None] * len(test_baseline) + template_ids

    # Shuffle test data while preserving metadata alignment
    indices = np.arange(len(X_test))
    indices = shuffle(indices, random_state=seed)

    X_test = [X_test[i] for i in indices]
    y_test = y_test[indices]
    test_template_ids_full = [test_template_ids_full[i] for i in indices]

    print(f"[+] Train size: {len(X_train)} ({int(y_train.sum())} malicious)")
    print(f"[+] Test size: {len(X_test)} ({int(y_test.sum())} malicious)")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_template_ids": test_template_ids_full,
        "test_templates": test_templates,
    }


def plot_bar_by_indexed_metric(
    series: pd.Series,
    num_samples: pd.Series,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "metric": series,
            "num_samples": num_samples,
        }
    ).sort_values("metric", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df))))
    colors = sns.color_palette("RdYlGn", len(df))

    ax.barh(df.index, df["metric"] * 100, color=colors)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    for i, (idx, row) in enumerate(df.iterrows()):
        # Percentage label at the end of each bar
        ax.text(
            row["metric"] * 100 + 1,
            i,
            f"{row['metric'] * 100:.1f}%",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def detection_by_attack_component(
    X_test: List[str],
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Compute detection metrics conditioned on presence of attack components.

    For each component (interpreter, shell_invoker, net_utility, ...), we look at
    malicious test commands that contain any of its indicator substrings and
    compute:
        - num_samples: number of such malicious commands
        - tpr: fraction correctly classified as malicious
        - mean_score: mean predicted malicious probability
    
    Args:
        threshold: Classification threshold (y_pred_proba > threshold → malicious)
    """
    groups = build_attack_component_groups()
    y_pred = (y_pred_proba > threshold).astype(int)

    malicious_indices = np.where(y_test == 1)[0]
    if malicious_indices.size == 0:
        return pd.DataFrame(columns=["component", "tpr", "mean_score", "num_samples"])

    cmds_lower = [X_test[i].lower() for i in malicious_indices]

    component_to_indices: Dict[str, List[int]] = {name: [] for name in groups.keys()}

    for global_idx, cmd_lower in zip(malicious_indices, cmds_lower):
        for name, indicators in groups.items():
            for indicator in indicators:
                if indicator.lower() in cmd_lower:
                    component_to_indices[name].append(global_idx)
                    break

    rows: List[Dict[str, object]] = []
    for name, idxs in component_to_indices.items():
        if not idxs:
            continue
        idx_array = np.asarray(idxs, dtype=int)
        num_samples = int(idx_array.size)

        correct = ((y_pred[idx_array] == 1) & (y_test[idx_array] == 1)).sum()
        tpr = float(correct) / num_samples if num_samples > 0 else 0.0
        mean_score = float(y_pred_proba[idx_array].mean())

        rows.append(
            {
                "component": name,
                "tpr": round(tpr, 4),
                "mean_score": round(mean_score, 4),
                "num_samples": num_samples,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("tpr", ascending=False).set_index("component")
    return df


def main() -> None:
    print("=" * 80)
    print("PER-TEMPLATE AND PER-TECHNIQUE DETECTION ANALYSIS")
    print("=" * 80)
    print(f"[!] ROOT: {ROOT}")
    print(f"[!] SEED: {SEED}, VOCAB_SIZE: {VOCAB_SIZE}, LIMIT: {LIMIT}")
    print(f"[!] OUT_DIR: {OUT_DIR}")

    data = load_data_with_metadata(ROOT, seed=SEED, limit=LIMIT)

    X_train: List[str] = data["X_train"]  # type: ignore[assignment]
    y_train: np.ndarray = data["y_train"]  # type: ignore[assignment]
    X_test: List[str] = data["X_test"]  # type: ignore[assignment]
    y_test: np.ndarray = data["y_test"]  # type: ignore[assignment]
    test_template_ids: List[int | None] = data["test_template_ids"]  # type: ignore[assignment]
    test_templates: List[str] = data["test_templates"]  # type: ignore[assignment]

    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)

    encoder = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
    print("[*] Fitting One-Hot encoder...")
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    print(f"[+] Vocabulary size: {len(encoder.vocab)}")
    print(f"[+] Train shape: {X_train_enc.shape}")
    print(f"[+] Test shape: {X_test_enc.shape}")

    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    model = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    trained = training_tabular(
        model=model,
        name="xgb_detection_by_technique",
        X_train_encoded=X_train_enc,
        X_test_encoded=X_test_enc,
        y_train=y_train,
        y_test=y_test,
        logs_folder=str(OUT_DIR / "xgboost_training"),
        model_file=None,
    )

    print("[+] Model training complete")

    y_pred = trained.predict(X_test_enc)
    y_pred_proba = trained.predict_proba(X_test_enc)[:, 1]

    overall_acc = float((y_pred == y_test).mean())
    overall_tpr = float(((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum())
    print(f"[+] Overall test accuracy: {overall_acc:.4f}")
    print(f"[+] Overall test TPR: {overall_tpr:.4f}")

    print("\n" + "=" * 80)
    print("PER-TEMPLATE DETECTION")
    print("=" * 80)

    det_by_template = detection_by_template(
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        template_ids=test_template_ids,
        test_templates=test_templates,
    )

    det_by_template.to_csv(OUT_DIR / "detection_by_template.csv", index=False)
    print("\n[+] Detection rates by individual template:")
    if not det_by_template.empty:
        print(det_by_template[["template_id", "primary_binary", "tpr", "num_samples"]].to_string())

    print("\n" + "=" * 80)
    print("AGGREGATION BY PRIMARY BINARY")
    print("=" * 80)

    by_binary = aggregate_detection_by_primary_binary(det_by_template)
    by_binary.to_csv(OUT_DIR / "detection_by_primary_binary.csv")
    print(by_binary)

    plot_bar_by_indexed_metric(
        series=by_binary["tpr"],
        num_samples=by_binary["num_samples"],
        xlabel="True Positive Rate (%)",
        title="Detection Rates by Primary Binary",
        out_path=PLOTS_DIR / "detection_by_primary_binary_barplot",
    )

    print("\n" + "=" * 80)
    print("AGGREGATION BY TECHNIQUE FAMILY")
    print("=" * 80)

    by_family = aggregate_detection_by_technique_family(det_by_template)
    by_family.to_csv(OUT_DIR / "detection_by_technique_family.csv")
    print(by_family)

    plot_bar_by_indexed_metric(
        series=by_family["tpr"],
        num_samples=by_family["num_samples"],
        xlabel="True Positive Rate (%)",
        title="Detection Rates by Technique Family",
        out_path=PLOTS_DIR / "detection_by_technique_family_barplot",
    )

    # ========================================================================
    # ATTACK COMPONENT ANALYSIS WITH MULTIPLE THRESHOLDS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DETECTION BY ATTACK COMPONENT - THRESHOLD WITH HIGHEST F1")
    print("=" * 80)
    
    threshold_f1 = get_threshold_with_highest_f1(y_test, y_pred_proba)
    print(f"[+] Threshold with highest F1: {threshold_f1:.6f}")
    
    by_component_f1 = detection_by_attack_component(
        X_test=X_test,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        threshold=threshold_f1,
    )
    by_component_f1.to_csv(OUT_DIR / "detection_by_attack_component_f1.csv")
    print(by_component_f1)

    if not by_component_f1.empty:
        plot_bar_by_indexed_metric(
            series=by_component_f1["tpr"],
            num_samples=by_component_f1["num_samples"],
            xlabel="True Positive Rate (%)",
            title=f"Detection by Attack Component (F1-optimal threshold={threshold_f1:.4f})",
            out_path=PLOTS_DIR / "detection_by_attack_component_f1",
        )
    
    print("\n" + "=" * 80)
    print("DETECTION BY ATTACK COMPONENT - FPR = 1e-4")
    print("=" * 80)
    
    threshold_1e4 = get_threshold_at_fpr(y_test, y_pred_proba, target_fpr=1e-4)
    print(f"[+] Threshold at FPR=1e-4: {threshold_1e4:.6f}")
    
    by_component_1e4 = detection_by_attack_component(
        X_test=X_test,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        threshold=threshold_1e4,
    )
    by_component_1e4.to_csv(OUT_DIR / "detection_by_attack_component_fpr_1e4.csv")
    print(by_component_1e4)

    if not by_component_1e4.empty:
        plot_bar_by_indexed_metric(
            series=by_component_1e4["tpr"],
            num_samples=by_component_1e4["num_samples"],
            xlabel="True Positive Rate (%)",
            title=f"Detection by Attack Component (FPR = $10^{{-4}}$, threshold={threshold_1e4:.4f})",
            out_path=PLOTS_DIR / "detection_by_attack_component_fpr_1e4",
        )
    
    # ========================================================================
    # ATTACK COMPONENT ANALYSIS AT FPR = 1e-6 (SAME DATASET)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DETECTION BY ATTACK COMPONENT - FPR = 1e-6")
    print("=" * 80)

    threshold_1e6 = get_threshold_at_fpr(y_test, y_pred_proba, target_fpr=1e-6)
    print(f"[+] Threshold at FPR=1e-6: {threshold_1e6:.6f}")

    by_component_1e6 = detection_by_attack_component(
        X_test=X_test,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        threshold=threshold_1e6,
    )
    by_component_1e6.to_csv(OUT_DIR / "detection_by_attack_component_fpr_1e6.csv")
    print(by_component_1e6)

    if not by_component_1e6.empty:
        plot_bar_by_indexed_metric(
            series=by_component_1e6["tpr"],
            num_samples=by_component_1e6["num_samples"],
            xlabel="True Positive Rate (%)",
            title=f"Detection by Attack Component (FPR = $10^{{-6}}$, threshold={threshold_1e6:.4f})",
            out_path=PLOTS_DIR / "detection_by_attack_component_fpr_1e6",
        )

    print("\n[+] Key technique-family contrasts:")
    for family in by_family.index:
        row = by_family.loc[family]
        print(
            f"  - {family}: TPR={row['tpr'] * 100:.1f}% "
            f"(N={int(row['num_samples'])})",
        )

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE - results saved to {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()


