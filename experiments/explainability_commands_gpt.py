#!/usr/bin/env python3
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.tokenize import wordpunct_tokenize

try:
    import shap  # type: ignore
except Exception:
    shap = None  # Lazy error later if SHAP is actually requested

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    from scipy import sparse  # type: ignore
except Exception:
    sparse = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Import QuasarNix modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessors import OneHotCustomVectorizer
from src.augmentation import NixCommandAugmentationWithBaseline, read_template_file
from src.tabular_utils import training_tabular

# ============================================================================
# CONFIGURATION - All values defined here (no CLI params)
# ============================================================================

SEED = 33
VOCAB_SIZE = 4096
LIMIT = 30000  # samples per class for faster experimentation

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "nix_shell"
TEMPLATE_TRAIN_PATH = DATA_ROOT / "templates_train.txt"
TEMPLATE_TEST_PATH = DATA_ROOT / "templates_test.txt"

# Output directory for results with timestamp
TIMESTAMP = int(time.time())
OUT_DIR = ROOT / "experiments" / f"results_commands_xai_{TIMESTAMP}"
OUT_DIR.mkdir(exist_ok=True, parents=True)

TOKENIZER = wordpunct_tokenize
TOP_K = 20  # Top-K groups for plots

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

print(f"[!] Configuration:")
print(f"    SEED: {SEED}")
print(f"    VOCAB_SIZE: {VOCAB_SIZE}")
print(f"    LIMIT: {LIMIT}")
print(f"    OUT_DIR: {OUT_DIR}")
print()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(limit: Optional[int] = None) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Load and prepare training and test data.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    print("[+] Loading baseline data...")
    train_baseline_df = pd.read_parquet(DATA_ROOT / "train_baseline_real.parquet")
    test_baseline_df = pd.read_parquet(DATA_ROOT / "test_baseline_real.parquet")
    
    train_baseline = train_baseline_df["cmd"].tolist()
    test_baseline = test_baseline_df["cmd"].tolist()
    
    if limit:
        train_baseline = train_baseline[:limit]
        test_baseline = test_baseline[:limit]
    
    print("[+] Loading templates...")
    train_templates = read_template_file(TEMPLATE_TRAIN_PATH)
    test_templates = read_template_file(TEMPLATE_TEST_PATH)
    
    print(f"    Train templates: {len(train_templates)}")
    print(f"    Test templates: {len(test_templates)}")
    
    # Generate malicious data
    print("[+] Generating training malicious data...")
    train_gen = NixCommandAugmentationWithBaseline(
        templates=train_templates,
        legitimate_baseline=train_baseline,
        random_state=SEED
    )
    train_per_template = max(1, len(train_baseline) // len(train_templates))
    X_train_malicious = train_gen.generate_commands(train_per_template)
    
    print("[+] Generating test malicious data...")
    test_gen = NixCommandAugmentationWithBaseline(
        templates=test_templates,
        legitimate_baseline=test_baseline,
        random_state=SEED + 1
    )
    test_per_template = max(1, len(test_baseline) // len(test_templates))
    X_test_malicious = test_gen.generate_commands(test_per_template)
    
    # Combine and shuffle
    X_train = train_baseline + X_train_malicious
    y_train = np.array([0] * len(train_baseline) + [1] * len(X_train_malicious), dtype=np.int8)
    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)
    
    X_test = test_baseline + X_test_malicious
    y_test = np.array([0] * len(test_baseline) + [1] * len(X_test_malicious), dtype=np.int8)
    X_test, y_test = shuffle(X_test, y_test, random_state=SEED)
    
    print(f"[+] Train size: {len(X_train)} ({sum(y_train)} malicious)")
    print(f"[+] Test size: {len(X_test)} ({sum(y_test)} malicious)")
    
    return X_train, y_train, X_test, y_test


def resolve_shap_values(shap_values: Any) -> np.ndarray:
    """
    Normalizes SHAP outputs into shape (n_samples, n_features).
    - For binary classifiers, SHAP may return list of 2 arrays (one per class).
      We take the positive class contributions (index 1) if available.
    """
    if isinstance(shap_values, list):
        # Prefer contributions to the positive class if present
        if len(shap_values) == 2 and shap_values[1] is not None:
            return shap_values[1]
        # Fall back to first element
        return shap_values[0]
    if isinstance(shap_values, np.ndarray):
        return shap_values
    raise ValueError("Unsupported SHAP value container.")


def build_utility_groups() -> Dict[str, List[str]]:
    """
    Mapping canonical utility -> list of exact token indicators.
    Tokens are assumed to come from wordpunct_tokenize (lower-cased).
    """
    return {
        "bash": ["bash", "sh", "zsh", "ksh"],  # grouped shells
        "python": ["python", "python2", "python3", "py"],
        "perl": ["perl"],
        "php": ["php"],
        "ruby": ["ruby"],
        "node": ["node", "nodejs"],
        "nc": ["nc", "ncat"],
        "socat": ["socat"],
        "telnet": ["telnet"],
        "openssl": ["openssl"],
        "curl": ["curl"],
        "wget": ["wget"],
        "ssh": ["ssh"],
        "busybox": ["busybox"],
        "nmap": ["nmap"],
    }


def build_tactic_groups() -> Dict[str, List[str]]:
    """
    Category -> list of indicative tokens (heuristic).
    """
    return {
        "interpreter": ["python", "python2", "python3", "perl", "php", "ruby", "node", "nodejs"],
        "shell_invoker": ["bash", "sh", "zsh", "ksh"],
        "net_utility": ["nc", "ncat", "socat", "telnet", "openssl", "curl", "wget", "ssh", "nmap"],
        "fd_redirection": [">", ">&", "2>&1", "0<&", "1>&", "/dev", "tcp", "udp"],
        "wrappers": ["sudo", "nohup", "setsid", "timeout", "time"],
        "obfuscation": ["base64", "eval", "exec", "xxd", "rev", "tr", "sed", "awk", "printf"],
        "ip_tokens": [".", "10", "127"],
    }


def tokens_for_groups(index_to_token: Dict[int, str],
                      group_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """
    Resolves feature indices per group by exact token equality against group indicators.
    """
    token_to_indices: Dict[str, List[int]] = {}
    for idx, tok in index_to_token.items():
        token_to_indices.setdefault(tok, []).append(idx)

    group_to_indices: Dict[str, List[int]] = {}
    for group_name, indicators in group_map.items():
        indices: List[int] = []
        for indicator in indicators:
            indices.extend(token_to_indices.get(indicator, []))
        # Deduplicate while preserving order
        seen = set()
        unique_indices = [i for i in indices if not (i in seen or seen.add(i))]
        group_to_indices[group_name] = unique_indices
    return group_to_indices


def sum_grouped_shap(shap_matrix: np.ndarray,
                     group_to_indices: Dict[str, List[int]]) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (grouped_shap_matrix, group_names)
      - grouped_shap_matrix: shape (n_samples, n_groups), each cell sums SHAP across group's indices.
    """
    n_samples = shap_matrix.shape[0]
    group_names = list(group_to_indices.keys())
    grouped = np.zeros((n_samples, len(group_names)), dtype=np.float32)
    for j, group in enumerate(group_names):
        cols = group_to_indices[group]
        if len(cols) == 0:
            continue
        grouped[:, j] = shap_matrix[:, cols].sum(axis=1)
    return grouped, group_names


def mean_abs_shap_per_group(grouped_shap: np.ndarray, group_names: List[str]) -> List[Tuple[str, float]]:
    mas = np.abs(grouped_shap).mean(axis=0)
    out = [(group_names[i], float(mas[i])) for i in range(len(group_names))]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def save_group_importance_csv(pairs: List[Tuple[str, float]], out_csv: str) -> None:
    with open(out_csv, "w") as f:
        f.write("group,mean_abs_shap\n")
        for group, score in pairs:
            f.write(f"{group},{score:.6f}\n")


def save_group_importance_json(pairs: List[Tuple[str, float]], out_json: str) -> None:
    with open(out_json, "w") as f:
        json.dump([{"group": g, "mean_abs_shap": s} for g, s in pairs], f, indent=2)


def plot_top_groups(pairs: List[Tuple[str, float]], out_png: str, top_k: int = 20, title: str = "") -> None:
    top = pairs[:top_k]
    labels = [g for g, _ in top][::-1]
    scores = [s for _, s in top][::-1]
    plt.figure(figsize=(8, max(2, 0.3 * len(labels))))
    plt.barh(labels, scores, color="#2b8cbe")
    plt.xlabel("Mean |SHAP| (grouped)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def mean_signed_shap_per_group(grouped_shap: np.ndarray, group_names: List[str]) -> List[Tuple[str, float]]:
    ms = grouped_shap.mean(axis=0)
    out = [(group_names[i], float(ms[i])) for i in range(len(group_names))]
    # Sort by absolute magnitude to surface strongest directional effects
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out


def plot_diverging_bars(pairs: List[Tuple[str, float]], out_png: str, top_k: int = 20, title: str = "") -> None:
    top = pairs[:top_k]
    labels = [g for g, _ in top][::-1]
    scores = [s for _, s in top][::-1]
    colors = ["#1a9850" if s > 0 else "#d73027" for s in scores]
    plt.figure(figsize=(8, max(2, 0.3 * len(labels))))
    plt.barh(labels, scores, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean SHAP (grouped)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_cumulative_explained(pairs_abs: List[Tuple[str, float]], out_png: str, title: str = "") -> None:
    # pairs_abs assumed sorted desc by abs magnitude
    vals = np.array([s for _, s in pairs_abs], dtype=np.float32)
    if vals.sum() <= 0:
        vals = np.ones_like(vals)
    frac = vals / vals.sum()
    cum = np.cumsum(frac)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    plt.xlabel("Top-N groups")
    plt.ylabel("Cumulative share of total |SHAP|")
    plt.ylim(0, 1.05)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_group_violins(grouped_shap: np.ndarray, group_names: List[str], out_png: str, 
                        top_pairs_abs: List[Tuple[str, float]], top_k: int = 12, title: str = "") -> None:
    """
    Create violin plots showing SHAP value distributions per tactic.
    Better than boxplots for showing distribution shape and density.
    """
    # Build data matrix for top groups by abs importance
    selected = [g for g, _ in top_pairs_abs[:top_k]]
    idxs = [group_names.index(g) for g in selected]
    
    # Create DataFrame for seaborn
    plot_data = []
    for group_name, idx in zip(selected, idxs):
        for shap_val in grouped_shap[:, idx]:
            plot_data.append({
                'Tactic': group_name,
                'SHAP Value': shap_val
            })
    df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * len(selected))))
    
    # Create violin plot
    # Use gradient colors based on mean SHAP value (more important = darker/redder)
    mean_shaps = [grouped_shap[:, idxs[selected.index(g)]].mean() for g in selected[::-1]]
    colors = [plt.cm.RdYlGn_r((val - min(mean_shaps)) / (max(mean_shaps) - min(mean_shaps) + 1e-10)) 
              for val in mean_shaps]
    
    sns.violinplot(
        data=df, 
        x='SHAP Value', 
        y='Tactic', 
        order=selected[::-1],  # Reverse to match importance order (top to bottom)
        inner='quartile',
        palette=colors,
        saturation=0.9,
        ax=ax
    )
    
    # Add mean markers
    for i, (group_name, idx) in enumerate(zip(selected[::-1], [idxs[selected.index(g)] for g in selected[::-1]])):
        mean_val = grouped_shap[:, idx].mean()
        ax.scatter(mean_val, i, color='black', s=100, zorder=10, marker='D', 
                  edgecolors='white', linewidths=1.5, label='Mean' if i == 0 else '')
    
    # Add reference line at 0
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='Neutral')
    
    # Labels and title
    ax.set_xlabel('SHAP Value per Sample\n(← Benign-indicating | Malicious-indicating →)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Tactic', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title + '\n(Positive SHAP pushes toward malicious; Negative toward benign)', 
                    fontsize=13, fontweight='bold', pad=15)
    else:
        ax.set_title('SHAP Distribution by Attack Tactic\n(Positive → Malicious; Negative → Benign)', 
                    fontsize=13, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_group_swarm_strip(grouped_shap: np.ndarray, group_names: List[str], out_png: str,
                           top_pairs_abs: List[Tuple[str, float]], top_k: int = 12, title: str = "") -> None:
    """
    Create strip plot with density overlay showing individual SHAP values.
    Alternative visualization that shows actual data points.
    """
    # Build data matrix for top groups by abs importance
    selected = [g for g, _ in top_pairs_abs[:top_k]]
    idxs = [group_names.index(g) for g in selected]
    
    # Sample data more aggressively for performance
    max_samples = 500
    if grouped_shap.shape[0] > max_samples:
        sample_idx = np.random.choice(grouped_shap.shape[0], max_samples, replace=False)
        grouped_shap_sampled = grouped_shap[sample_idx]
    else:
        grouped_shap_sampled = grouped_shap
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * len(selected))))
    
    # First plot violin for density (background)
    plot_data_full = []
    for group_name, idx in zip(selected, idxs):
        for shap_val in grouped_shap[:, idx]:
            plot_data_full.append({
                'Tactic': group_name,
                'SHAP Value': shap_val
            })
    df_full = pd.DataFrame(plot_data_full)
    
    sns.violinplot(
        data=df_full,
        x='SHAP Value',
        y='Tactic',
        order=selected[::-1],
        inner=None,
        color='lightgray',
        alpha=0.4,
        linewidth=1,
        ax=ax
    )
    
    # Then overlay strip plot with sampled data
    plot_data_sample = []
    for group_name, idx in zip(selected, idxs):
        for shap_val in grouped_shap_sampled[:, idx]:
            plot_data_sample.append({
                'Tactic': group_name,
                'SHAP Value': shap_val,
                'color': 'red' if shap_val > 0 else 'blue'
            })
    df_sample = pd.DataFrame(plot_data_sample)
    
    # Color points by sign
    for tactic_idx, tactic_name in enumerate(selected[::-1]):
        tactic_data = df_sample[df_sample['Tactic'] == tactic_name]
        pos_data = tactic_data[tactic_data['SHAP Value'] > 0]
        neg_data = tactic_data[tactic_data['SHAP Value'] <= 0]
        
        if len(pos_data) > 0:
            ax.scatter(pos_data['SHAP Value'], 
                      np.random.normal(tactic_idx, 0.15, len(pos_data)),
                      alpha=0.4, s=15, color='orangered', 
                      label='Malicious-indicating' if tactic_idx == 0 else '')
        if len(neg_data) > 0:
            ax.scatter(neg_data['SHAP Value'],
                      np.random.normal(tactic_idx, 0.15, len(neg_data)),
                      alpha=0.4, s=15, color='steelblue',
                      label='Benign-indicating' if tactic_idx == 0 else '')
    
    # Add mean and median markers
    for i, (group_name, idx) in enumerate(zip(selected[::-1], [idxs[selected.index(g)] for g in selected[::-1]])):
        mean_val = grouped_shap[:, idx].mean()
        median_val = np.median(grouped_shap[:, idx])
        ax.scatter(mean_val, i, color='red', s=120, zorder=10, marker='D',
                  edgecolors='white', linewidths=2, label='Mean' if i == 0 else '')
        ax.scatter(median_val, i, color='blue', s=120, zorder=10, marker='o',
                  edgecolors='white', linewidths=2, label='Median' if i == 0 else '')
    
    # Add reference line at 0
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='Neutral')
    
    # Labels and title
    ax.set_xlabel('SHAP Value per Sample\n(← Benign-indicating | Malicious-indicating →)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Tactic', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title + '\n(Positive SHAP pushes toward malicious; Negative toward benign)',
                    fontsize=13, fontweight='bold', pad=15)
    else:
        ax.set_title('SHAP Distribution by Attack Tactic (Individual Samples)\n(Positive → Malicious; Negative → Benign)',
                    fontsize=13, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def compute_token_level_shap(model: Any, X: np.ndarray) -> np.ndarray:
    assert shap is not None, "shap is required to compute SHAP values"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return resolve_shap_values(shap_values)


def main():
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(limit=LIMIT)
    
    # Encode features using One-Hot vectorizer
    print("\n[+] Encoding features with One-Hot vectorizer...")
    oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
    X_train_encoded = oh.fit_transform(X_train)
    X_test_encoded = oh.transform(X_test)
    
    vocab = oh.vocab  # Dict[str, int]
    index_to_token = {idx: token for token, idx in vocab.items()}
    
    print(f"    Vocabulary size: {len(vocab)}")
    print(f"    Train shape: {X_train_encoded.shape}")
    print(f"    Test shape: {X_test_encoded.shape}")
    
    # Save vocabulary for reference
    vocab_path = OUT_DIR / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"    Saved vocabulary to: {vocab_path}")
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    print("[*] Training XGBoost model...")
    
    trained_model = training_tabular(
        model=xgb_model,
        name="xgb_commands_xai",
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
        logs_folder=str(OUT_DIR / "model_logs"),
        model_file=None
    )
    
    print("[+] Model training complete")
    
    # Compute SHAP values
    print("\n" + "="*80)
    print("COMPUTING TOKEN-LEVEL SHAP VALUES")
    print("="*80)
    
    # Convert to dense for SHAP (if sparse)
    if sparse.issparse(X_test_encoded):
        X_test_dense = X_test_encoded.toarray().astype(np.float32)
    else:
        X_test_dense = X_test_encoded.astype(np.float32)
    
    print(f"[*] Computing SHAP values for {X_test_dense.shape[0]} test samples...")
    token_shap = compute_token_level_shap(trained_model, X_test_dense)
    print(f"[+] SHAP values computed: shape {token_shap.shape}")
    
    # Save token-level SHAP
    np.save(OUT_DIR / "token_shap_values.npy", token_shap)
    print(f"    Saved to: {OUT_DIR / 'token_shap_values.npy'}")
    
    # Create plots directory
    plots_dir = OUT_DIR / "plots"
    ensure_dir(str(plots_dir))
    
    # ========================================================================
    # UTILITY-LEVEL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("UTILITY-LEVEL GROUPED SHAP ANALYSIS")
    print("="*80)
    
    utility_groups = build_utility_groups()
    utility_indices = tokens_for_groups(index_to_token, utility_groups)
    utility_grouped, utility_names = sum_grouped_shap(token_shap, utility_indices)
    utility_pairs = mean_abs_shap_per_group(utility_grouped, utility_names)
    
    print(f"\n[+] Top utilities by mean |SHAP|:")
    for utility, score in utility_pairs[:10]:
        print(f"    {utility}: {score:.4f}")
    
    save_group_importance_csv(utility_pairs, str(OUT_DIR / "utilities_mean_abs_shap.csv"))
    save_group_importance_json(utility_pairs, str(OUT_DIR / "utilities_mean_abs_shap.json"))
    plot_top_groups(utility_pairs, str(plots_dir / "utilities_mean_abs_shap.png"),
                    top_k=TOP_K, title="Utility-level grouped SHAP")
    
    # Signed means
    utility_signed = mean_signed_shap_per_group(utility_grouped, utility_names)
    plot_diverging_bars(utility_signed, str(plots_dir / "utilities_mean_signed_shap.png"),
                        top_k=TOP_K, title="Utility-level mean SHAP (signed)")
    plot_cumulative_explained(utility_pairs, str(plots_dir / "utilities_cumulative_share.png"),
                              title="Utilities cumulative share of total |SHAP|")
    
    # Create multiple visualization types
    print("    Creating violin plot...")
    plot_group_violins(utility_grouped, utility_names,
                      str(plots_dir / "utilities_grouped_shap_violins.png"),
                      utility_pairs, top_k=min(TOP_K, 12),
                      title="Utilities grouped SHAP distribution")
    
    print("    Creating strip+density plot...")
    plot_group_swarm_strip(utility_grouped, utility_names,
                          str(plots_dir / "utilities_grouped_shap_strip.png"),
                          utility_pairs, top_k=min(TOP_K, 12),
                          title="Utilities grouped SHAP distribution")
    
    # ========================================================================
    # TACTIC-LEVEL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("TACTIC-LEVEL GROUPED SHAP ANALYSIS")
    print("="*80)
    
    tactic_groups = build_tactic_groups()
    tactic_indices = tokens_for_groups(index_to_token, tactic_groups)
    tactic_grouped, tactic_names = sum_grouped_shap(token_shap, tactic_indices)
    tactic_pairs = mean_abs_shap_per_group(tactic_grouped, tactic_names)
    
    print(f"\n[+] Top tactics by mean |SHAP|:")
    for tactic, score in tactic_pairs[:10]:
        print(f"    {tactic}: {score:.4f}")
    
    save_group_importance_csv(tactic_pairs, str(OUT_DIR / "tactics_mean_abs_shap.csv"))
    save_group_importance_json(tactic_pairs, str(OUT_DIR / "tactics_mean_abs_shap.json"))
    plot_top_groups(tactic_pairs, str(plots_dir / "tactics_mean_abs_shap.png"),
                    top_k=TOP_K, title="Tactic-level grouped SHAP")
    
    tactic_signed = mean_signed_shap_per_group(tactic_grouped, tactic_names)
    plot_diverging_bars(tactic_signed, str(plots_dir / "tactics_mean_signed_shap.png"),
                        top_k=TOP_K, title="Tactic-level mean SHAP (signed)")
    plot_cumulative_explained(tactic_pairs, str(plots_dir / "tactics_cumulative_share.png"),
                              title="Tactics cumulative share of total |SHAP|")
    
    # Create multiple visualization types
    print("    Creating violin plot...")
    plot_group_violins(tactic_grouped, tactic_names,
                      str(plots_dir / "tactics_grouped_shap_violins.png"),
                      tactic_pairs, top_k=min(TOP_K, 12),
                      title="Tactics grouped SHAP distribution")
    
    print("    Creating strip+density plot...")
    plot_group_swarm_strip(tactic_grouped, tactic_names,
                          str(plots_dir / "tactics_grouped_shap_strip.png"),
                          tactic_pairs, top_k=min(TOP_K, 12),
                          title="Tactics grouped SHAP distribution")
    
    # ========================================================================
    # PER-LABEL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PER-LABEL SHAP ANALYSIS")
    print("="*80)
    
    for name, grouped in [("utilities", utility_grouped), ("tactics", tactic_grouped)]:
        group_names = utility_names if name == "utilities" else tactic_names
        
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        
        if pos_mask.any():
            pos_pairs = mean_abs_shap_per_group(grouped[pos_mask], group_names)
            save_group_importance_csv(pos_pairs, str(OUT_DIR / f"{name}_mean_abs_shap_pos.csv"))
            pos_signed = mean_signed_shap_per_group(grouped[pos_mask], group_names)
            plot_diverging_bars(pos_signed, str(plots_dir / f"{name}_mean_signed_shap_pos.png"),
                                top_k=TOP_K, title=f"{name.capitalize()} mean SHAP (signed) - malicious")
        
        if neg_mask.any():
            neg_pairs = mean_abs_shap_per_group(grouped[neg_mask], group_names)
            save_group_importance_csv(neg_pairs, str(OUT_DIR / f"{name}_mean_abs_shap_neg.csv"))
            neg_signed = mean_signed_shap_per_group(grouped[neg_mask], group_names)
            plot_diverging_bars(neg_signed, str(plots_dir / f"{name}_mean_signed_shap_neg.png"),
                                top_k=TOP_K, title=f"{name.capitalize()} mean SHAP (signed) - benign")
    
    # Save grouped SHAP matrices for further analysis
    np.save(OUT_DIR / "utility_grouped_shap.npy", utility_grouped)
    np.save(OUT_DIR / "tactic_grouped_shap.npy", tactic_grouped)
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - Results saved to {OUT_DIR}")
    print("="*80)
    print(f"\nKey outputs:")
    print(f"  - Vocabulary: {OUT_DIR / 'vocabulary.json'}")
    print(f"  - Token SHAP: {OUT_DIR / 'token_shap_values.npy'}")
    print(f"  - Utilities CSV: {OUT_DIR / 'utilities_mean_abs_shap.csv'}")
    print(f"  - Tactics CSV: {OUT_DIR / 'tactics_mean_abs_shap.csv'}")
    print(f"  - Plots: {plots_dir}/")


if __name__ == "__main__":
    main()


