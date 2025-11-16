"""
Template-Level XAI Analysis for QuasarNix

This script extends the token-level SHAP analysis to:
1. Track which template generated each test sample
2. Compute detection rates per reverse shell technique (bash, python, perl, etc.)
3. Analyze SHAP values grouped by technique type
4. Extract important command patterns (n-grams) beyond single tokens
5. Generate paper-ready LaTeX tables showing technique-level insights

Addresses Reviewer 1's request: "expand on which specific command substrings or
feature patterns were most consistently identified as important, and how these
insights might inform practical defense design"
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import wordpunct_tokenize
from xgboost import XGBClassifier

# Import QuasarNix modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessors import OneHotCustomVectorizer
from src.augmentation import NixCommandAugmentationWithBaseline, read_template_file
from src.tabular_utils import training_tabular

import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

SEED = 33
VOCAB_SIZE = 4096
MAX_LEN = 128
LIMIT = 30000  # samples per class

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "nix_shell"
TEMPLATE_TRAIN_PATH = DATA_ROOT / "templates_train.txt"
TEMPLATE_TEST_PATH = DATA_ROOT / "templates_test.txt"

TOKENIZER = wordpunct_tokenize

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

print(f"[!] Script start time: {time.ctime()}")
print(f"[!] ROOT: {ROOT}")
print(f"[!] SEED: {SEED}, VOCAB_SIZE: {VOCAB_SIZE}, LIMIT: {LIMIT}")


# ============================================================================
# Section 1: Enhanced Data Generation with Template Metadata
# ============================================================================

def generate_malicious_data_with_metadata(
    templates: List[str],
    baseline: List[str],
    num_per_template: int,
    seed: int
) -> Tuple[List[str], List[int], List[str]]:
    """
    Generate malicious commands WITH template metadata tracking.

    Args:
        templates: List of template strings
        baseline: Legitimate commands for baseline sampling
        num_per_template: Number of examples to generate per template
        seed: Random seed

    Returns:
        commands: List of generated commands
        template_ids: Which template index generated each command
        template_strings: Original template string for each command
    """
    generator = NixCommandAugmentationWithBaseline(
        templates=templates,
        legitimate_baseline=baseline,
        random_state=seed
    )

    commands = []
    template_ids = []
    template_strings = []

    print(f"[+] Generating {num_per_template} examples per template for {len(templates)} templates")

    for template_idx, template in enumerate(tqdm(templates, desc="Generating per template")):
        # Generate commands for this specific template
        temp_gen = NixCommandAugmentationWithBaseline(
            templates=[template],  # Single template
            legitimate_baseline=baseline,
            random_state=seed + template_idx
        )

        template_commands = temp_gen.generate_commands(num_per_template)

        commands.extend(template_commands)
        template_ids.extend([template_idx] * len(template_commands))
        template_strings.extend([template] * len(template_commands))

    print(f"[!] Generated {len(commands)} total commands with metadata")
    return commands, template_ids, template_strings


def load_data_with_metadata(
    root: Path,
    seed: int = 33,
    limit: int = None
) -> Dict:
    """
    Load training/test data WITH template metadata for malicious samples.

    Returns dict with:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - X_test_malicious: Just malicious test commands
        - test_template_ids: Template ID for each malicious test sample
        - test_template_strings: Template string for each malicious test sample
        - test_templates: List of unique test templates
    """
    # Load baseline data
    print("[+] Loading baseline data...")
    paths = {
        "train_baseline": root / "data" / "nix_shell" / "train_baseline_real.parquet",
        "test_baseline": root / "data" / "nix_shell" / "test_baseline_real.parquet",
    }

    train_baseline_df = pd.read_parquet(paths["train_baseline"])
    test_baseline_df = pd.read_parquet(paths["test_baseline"])

    train_baseline = train_baseline_df["cmd"].tolist()
    test_baseline = test_baseline_df["cmd"].tolist()

    # Limit if specified
    if limit:
        train_baseline = train_baseline[:limit]
        test_baseline = test_baseline[:limit]

    # Load templates
    print("[+] Loading templates...")
    train_templates = read_template_file(TEMPLATE_TRAIN_PATH)
    test_templates = read_template_file(TEMPLATE_TEST_PATH)

    print(f"[+] Train templates: {len(train_templates)}")
    print(f"[+] Test templates: {len(test_templates)}")

    # Generate training data (without metadata tracking for efficiency)
    print("[+] Generating training malicious data...")
    train_gen = NixCommandAugmentationWithBaseline(
        templates=train_templates,
        legitimate_baseline=train_baseline,
        random_state=seed
    )
    train_per_template = max(1, len(train_baseline) // len(train_templates))
    X_train_malicious = train_gen.generate_commands(train_per_template)

    # Generate test data WITH metadata
    print("[+] Generating test malicious data WITH metadata...")
    test_per_template = max(1, len(test_baseline) // len(test_templates))
    X_test_malicious, test_template_ids, test_template_strings = \
        generate_malicious_data_with_metadata(
            test_templates, test_baseline, test_per_template, seed + 1
        )

    # Combine training data
    X_train = train_baseline + X_train_malicious
    y_train = np.array([0] * len(train_baseline) + [1] * len(X_train_malicious), dtype=np.int8)
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    # Combine test data
    X_test = test_baseline + X_test_malicious
    y_test = np.array([0] * len(test_baseline) + [1] * len(X_test_malicious), dtype=np.int8)

    # Create combined template metadata (None for benign, template_id for malicious)
    test_template_ids_full = [None] * len(test_baseline) + test_template_ids
    test_template_strings_full = [None] * len(test_baseline) + test_template_strings

    # Shuffle test data while preserving metadata alignment
    indices = np.arange(len(X_test))
    indices = shuffle(indices, random_state=seed)

    X_test = [X_test[i] for i in indices]
    y_test = y_test[indices]
    test_template_ids_full = [test_template_ids_full[i] for i in indices]
    test_template_strings_full = [test_template_strings_full[i] for i in indices]

    print(f"[+] Train size: {len(X_train)} ({sum(y_train)} malicious)")
    print(f"[+] Test size: {len(X_test)} ({sum(y_test)} malicious)")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_baseline': train_baseline,
        'X_train_malicious': X_train_malicious,
        'X_test_baseline': test_baseline,
        'X_test_malicious': X_test_malicious,
        'test_template_ids': test_template_ids_full,
        'test_template_strings': test_template_strings_full,
        'test_templates': test_templates
    }


# ============================================================================
# Section 2: Template Technique Classification
# ============================================================================

def classify_template_technique(template: str) -> Dict[str, any]:
    """
    Extract technique characteristics from template string.

    Returns dict with:
        - primary_binary: Main executable (bash, python, perl, ruby, php, nc, etc.)
        - technique_family: High-level category (scripting, netcat, file_descriptor, named_pipe)
        - network_primitives: List of network-related patterns in the template
    """
    techniques = {
        'primary_binary': None,
        'technique_family': None,
        'network_primitives': []
    }

    template_lower = template.lower()

    # Detect primary binary interpreter
    if 'python3' in template_lower:
        techniques['primary_binary'] = 'python3'
        techniques['technique_family'] = 'scripting'
        if 'socket.socket' in template:
            techniques['network_primitives'].append('socket.socket')
        if 'pty.spawn' in template:
            techniques['network_primitives'].append('pty.spawn')
        if 'os.dup2' in template:
            techniques['network_primitives'].append('os.dup2')

    elif 'python' in template_lower and 'python3' not in template_lower:
        techniques['primary_binary'] = 'python'
        techniques['technique_family'] = 'scripting'
        if 'socket.socket' in template:
            techniques['network_primitives'].append('socket.socket')
        if 'pty.spawn' in template:
            techniques['network_primitives'].append('pty.spawn')
        if 'os.dup2' in template:
            techniques['network_primitives'].append('os.dup2')

    elif 'perl' in template_lower:
        techniques['primary_binary'] = 'perl'
        techniques['technique_family'] = 'scripting'
        if 'Socket' in template or 'socket' in template_lower:
            techniques['network_primitives'].append('use Socket')
        if 'sockaddr_in' in template:
            techniques['network_primitives'].append('sockaddr_in')

    elif 'php' in template_lower:
        techniques['primary_binary'] = 'php'
        techniques['technique_family'] = 'scripting'
        if 'fsockopen' in template:
            techniques['network_primitives'].append('fsockopen')
        if 'proc_open' in template:
            techniques['network_primitives'].append('proc_open')
        if 'popen' in template:
            techniques['network_primitives'].append('popen')
        if 'shell_exec' in template:
            techniques['network_primitives'].append('shell_exec')

    elif 'ruby' in template_lower:
        techniques['primary_binary'] = 'ruby'
        techniques['technique_family'] = 'scripting'
        if 'TCPSocket' in template:
            techniques['network_primitives'].append('TCPSocket')
        if 'spawn' in template:
            techniques['network_primitives'].append('spawn')

    elif template_lower.startswith('nc ') or ' nc ' in template_lower:
        techniques['primary_binary'] = 'netcat'
        techniques['technique_family'] = 'netcat'
        if '-e' in template:
            techniques['network_primitives'].append('nc -e')
        if '-c' in template:
            techniques['network_primitives'].append('nc -c')

    elif 'awk' in template_lower:
        techniques['primary_binary'] = 'awk'
        techniques['technique_family'] = 'scripting'
        if '/inet/' in template:
            techniques['network_primitives'].append('/inet/')

    elif 'lua' in template_lower:
        techniques['primary_binary'] = 'lua'
        techniques['technique_family'] = 'scripting'
        if 'socket' in template_lower:
            techniques['network_primitives'].append('socket.tcp')

    elif 'zsh' in template_lower:
        techniques['primary_binary'] = 'zsh'
        techniques['technique_family'] = 'scripting'
        if 'ztcp' in template:
            techniques['network_primitives'].append('ztcp')

    elif 'mkfifo' in template:
        techniques['technique_family'] = 'named_pipe'
        techniques['primary_binary'] = 'bash'
        techniques['network_primitives'].append('mkfifo')

    elif 'telnet' in template_lower:
        techniques['primary_binary'] = 'telnet'
        techniques['technique_family'] = 'netcat'
        if 'mkfifo' in template:
            techniques['network_primitives'].append('mkfifo')

    elif 'echo' in template and '.v' in template:
        # V language technique
        techniques['primary_binary'] = 'v'
        techniques['technique_family'] = 'scripting'
        techniques['network_primitives'].append('v run')

    elif '/dev/tcp/' in template or '/dev/protocol_type/' in template_lower:
        techniques['technique_family'] = 'file_descriptor'
        techniques['primary_binary'] = 'bash'
        techniques['network_primitives'].append('/dev/tcp')

    elif 'rcat' in template_lower:
        techniques['primary_binary'] = 'rcat'
        techniques['technique_family'] = 'netcat'

    elif 'socat' in template_lower:
        techniques['primary_binary'] = 'socat'
        techniques['technique_family'] = 'netcat'

    # Default to bash if nothing else matched
    if techniques['primary_binary'] is None:
        techniques['primary_binary'] = 'bash'
        if techniques['technique_family'] is None:
            techniques['technique_family'] = 'shell'

    return techniques


# ============================================================================
# Section 3: Train Models
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

data = load_data_with_metadata(ROOT, seed=SEED, limit=LIMIT)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
test_template_ids = data['test_template_ids']
test_template_strings = data['test_template_strings']
test_templates = data['test_templates']

print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# One-Hot encoding
oh = OneHotCustomVectorizer(tokenizer=TOKENIZER, max_features=VOCAB_SIZE)
print("[*] Fitting One-Hot encoder...")
X_train_onehot = oh.fit_transform(X_train)
X_test_onehot = oh.transform(X_test)
features = list(oh.vocab.keys())

print(f"[+] Vocabulary size: {len(features)}")
print(f"[+] Train shape: {X_train_onehot.shape}")
print(f"[+] Test shape: {X_test_onehot.shape}")

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# Train GBDT model (regular training)
xgb_model = XGBClassifier(n_estimators=100, max_depth=10, random_state=SEED)

xgb_trained = training_tabular(
    model=xgb_model,
    name="xgb_template_analysis",
    X_train_encoded=X_train_onehot,
    X_test_encoded=X_test_onehot,
    y_train=y_train,
    y_test=y_test,
    logs_folder=str(ROOT / "experiments" / "results_template_xai" / "xgboost_training"),
    model_file=None
)

print("[+] Model training complete")

# Get predictions
y_pred = xgb_trained.predict(X_test_onehot)
y_pred_proba = xgb_trained.predict_proba(X_test_onehot)[:, 1]

print(f"[+] Overall test accuracy: {(y_pred == y_test).mean():.4f}")
print(f"[+] Overall test TPR: {((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum():.4f}")


# ============================================================================
# Section 4: Per-Template Detection Analysis
# ============================================================================

print("\n" + "="*80)
print("PER-TEMPLATE DETECTION ANALYSIS")
print("="*80)

def analyze_detection_by_template(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    template_ids: List,
    template_strings: List,
    test_templates: List[str]
) -> pd.DataFrame:
    """
    Compute detection rate per template.
    """
    results = []

    for template_idx, template in enumerate(test_templates):
        # Find all test samples from this template
        mask = np.array([tid == template_idx for tid in template_ids])

        if mask.sum() == 0:
            continue

        # Metrics
        n_samples = mask.sum()
        n_correct = ((y_pred[mask] == 1) & (y_test[mask] == 1)).sum()
        tpr = n_correct / n_samples if n_samples > 0 else 0
        mean_score = y_pred_proba[mask].mean()

        # Classify technique
        techniques = classify_template_technique(template)

        results.append({
            'template_id': template_idx,
            'template': template[:100] + '...' if len(template) > 100 else template,
            'primary_binary': techniques['primary_binary'],
            'technique_family': techniques['technique_family'],
            'network_primitives': ', '.join(techniques['network_primitives'][:3]),  # First 3
            'num_samples': n_samples,
            'tpr': round(tpr, 4),
            'mean_score': round(mean_score, 4)
        })

    return pd.DataFrame(results)


detection_by_template = analyze_detection_by_template(
    y_test, y_pred, y_pred_proba,
    test_template_ids, test_template_strings, test_templates
)

print("\n[+] Detection rates by individual template:")
print(detection_by_template[['template_id', 'primary_binary', 'tpr', 'num_samples']].to_string())

# Aggregate by technique
print("\n[+] Detection rates by technique (primary binary):")
by_technique = detection_by_template.groupby('primary_binary').agg({
    'tpr': 'mean',
    'mean_score': 'mean',
    'num_samples': 'sum'
}).round(4).sort_values('tpr', ascending=False)

print(by_technique)

# Save results
output_dir = ROOT / "experiments" / "results_template_xai"
output_dir.mkdir(exist_ok=True, parents=True)

detection_by_template.to_csv(output_dir / "detection_by_template.csv", index=False)
by_technique.to_csv(output_dir / "detection_by_technique.csv")

print(f"\n[+] Results saved to {output_dir}")

# ============================================================================
# Visualization 1: Bar plot of detection rates by technique
# ============================================================================

print("\n[+] Creating visualization 1: Detection rates by technique...")
fig, ax = plt.subplots(figsize=(10, 6))

by_technique_sorted = by_technique.sort_values('tpr', ascending=True)
colors = sns.color_palette("RdYlGn", len(by_technique_sorted))

ax.barh(by_technique_sorted.index, by_technique_sorted['tpr'] * 100, color=colors)
ax.set_xlabel('True Positive Rate (%)', fontsize=13)
ax.set_ylabel('Reverse Shell Technique', fontsize=13)
ax.set_title('Detection Rates by Reverse Shell Technique Type', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)

# Add value labels on bars
for i, (idx, row) in enumerate(by_technique_sorted.iterrows()):
    ax.text(row['tpr'] * 100 + 1, i, f"{row['tpr']*100:.1f}%",
            va='center', fontsize=10)

# Add sample size annotations
for i, (idx, row) in enumerate(by_technique_sorted.iterrows()):
    ax.text(5, i, f"N={int(row['num_samples']):,}",
            va='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plot_path = output_dir / "plots" / "detection_by_technique_barplot.pdf"
plot_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"    Saved: {plot_path}")

# ============================================================================
# Visualization 2: Heatmap of per-template detection rates
# ============================================================================

print("[+] Creating visualization 2: Per-template detection heatmap...")

# Group by technique and show individual templates
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by technique and TPR
detection_by_template_plot = detection_by_template.sort_values(
    ['primary_binary', 'tpr'], ascending=[True, False]
)

# Create color mapping
cmap = sns.diverging_palette(10, 130, as_cmap=True)
tpr_colors = [plt.cm.RdYlGn(tpr) for tpr in detection_by_template_plot['tpr'].values]

y_pos = np.arange(len(detection_by_template_plot))
ax.barh(y_pos, detection_by_template_plot['tpr'] * 100, color=tpr_colors)

# Y-axis labels: template_id + primary_binary
labels = [f"{row['primary_binary']}-{row['template_id']}"
          for _, row in detection_by_template_plot.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)

ax.set_xlabel('True Positive Rate (%)', fontsize=13)
ax.set_ylabel('Template (Technique-ID)', fontsize=13)
ax.set_title('Detection Rates per Individual Template', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)

# Add technique group separators
current_technique = None
for i, (_, row) in enumerate(detection_by_template_plot.iterrows()):
    if row['primary_binary'] != current_technique:
        ax.axhline(i - 0.5, color='black', linewidth=1.5, alpha=0.3)
        current_technique = row['primary_binary']

plt.tight_layout()
plot_path = output_dir / "plots" / "detection_per_template_heatmap.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"    Saved: {plot_path}")

# ============================================================================
# Section 5: SHAP Analysis Grouped by Technique
# ============================================================================

print("\n" + "="*80)
print("SHAP ANALYSIS BY TECHNIQUE")
print("="*80)

# Create SHAP explainer
print("[*] Creating SHAP explainer...")
explainer = shap.Explainer(xgb_trained, feature_names=features)

# Compute explanations (on malicious samples only for technique analysis)
malicious_mask = y_test == 1
X_test_malicious_onehot = X_test_onehot[malicious_mask]
malicious_template_ids = [tid for tid, label in zip(test_template_ids, y_test) if label == 1]

print(f"[*] Computing SHAP values for {X_test_malicious_onehot.shape[0]} malicious test samples...")
explanations = explainer(X_test_malicious_onehot)

print("[+] SHAP values computed")


def shap_analysis_by_technique(
    explanations,
    template_ids: List,
    test_templates: List[str],
    features: List[str],
    top_n: int = 10
) -> Dict:
    """
    Compute SHAP values grouped by technique type.

    Returns dict mapping technique -> {malicious: {}, benign: {}, num_samples: int}
    """
    # Group SHAP values by primary binary
    technique_shap_values = defaultdict(list)

    for idx, template_id in enumerate(template_ids):
        if template_id is None:
            continue

        template = test_templates[template_id]
        techniques = classify_template_technique(template)
        binary = techniques['primary_binary']

        technique_shap_values[binary].append(explanations.values[idx])

    # Aggregate and extract top features
    results = {}

    for binary, shap_arrays in technique_shap_values.items():
        # Stack all SHAP values for this technique
        combined_shap = np.vstack(shap_arrays)
        mean_shap = np.mean(combined_shap, axis=0)

        # Top malicious tokens (highest positive SHAP)
        top_malicious_idx = np.argsort(mean_shap)[-top_n:][::-1]
        top_malicious = {features[i]: round(mean_shap[i], 4) for i in top_malicious_idx}

        # Top benign tokens (most negative SHAP)
        top_benign_idx = np.argsort(mean_shap)[:top_n]
        top_benign = {features[i]: round(mean_shap[i], 4) for i in top_benign_idx}

        results[binary] = {
            'malicious': {k: float(v) for k, v in top_malicious.items()},
            'benign': {k: float(v) for k, v in top_benign.items()},
            'num_samples': int(combined_shap.shape[0])
        }

    return results


technique_shap = shap_analysis_by_technique(
    explanations, malicious_template_ids, test_templates, features, top_n=15
)

print("\n[+] SHAP analysis by technique:")
for binary, data in sorted(technique_shap.items()):
    print(f"\n{binary.upper()} (N={data['num_samples']}):")
    print(f"  Top malicious tokens: {list(data['malicious'].items())[:5]}")
    print(f"  Top benign tokens: {list(data['benign'].items())[:5]}")

# Save SHAP results
with open(output_dir / "shap_by_technique.json", "w") as f:
    json.dump(technique_shap, f, indent=2)

print(f"\n[+] SHAP results saved to {output_dir / 'shap_by_technique.json'}")

# ============================================================================
# Visualization 3: SHAP importance by technique (heatmap)
# ============================================================================

print("[+] Creating visualization 3: SHAP token importance heatmap...")

# Create a matrix of top tokens per technique
techniques_list = sorted(technique_shap.keys())
all_tokens = set()

# Collect all important tokens across techniques
for binary in techniques_list:
    all_tokens.update(list(technique_shap[binary]['malicious'].keys())[:10])
    all_tokens.update(list(technique_shap[binary]['benign'].keys())[:10])

all_tokens = sorted(all_tokens)[:30]  # Limit to top 30 overall

# Build matrix
shap_matrix = np.zeros((len(techniques_list), len(all_tokens)))

for i, binary in enumerate(techniques_list):
    for j, token in enumerate(all_tokens):
        if token in technique_shap[binary]['malicious']:
            shap_matrix[i, j] = technique_shap[binary]['malicious'][token]
        elif token in technique_shap[binary]['benign']:
            shap_matrix[i, j] = technique_shap[binary]['benign'][token]

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    shap_matrix,
    xticklabels=all_tokens,
    yticklabels=techniques_list,
    cmap='RdBu_r',
    center=0,
    annot=False,
    fmt='.2f',
    cbar_kws={'label': 'Mean SHAP Value'},
    ax=ax
)

ax.set_xlabel('Token', fontsize=13)
ax.set_ylabel('Technique', fontsize=13)
ax.set_title('Token Importance (SHAP) by Reverse Shell Technique', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plot_path = output_dir / "plots" / "shap_by_technique_heatmap.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"    Saved: {plot_path}")

# ============================================================================
# Visualization 4: Top malicious tokens per technique (grouped bar)
# ============================================================================

print("[+] Creating visualization 4: Top malicious tokens per technique...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, binary in enumerate(sorted(technique_shap.keys())[:6]):  # Top 6 techniques
    ax = axes[idx]
    data = technique_shap[binary]

    # Top 10 malicious tokens
    mal_tokens = list(data['malicious'].items())[:10]
    tokens = [t[0] for t in mal_tokens]
    values = [t[1] for t in mal_tokens]

    colors = ['green' if v > 0 else 'red' for v in values]
    ax.barh(range(len(tokens)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel('SHAP Value', fontsize=11)
    ax.set_title(f"{binary.upper()} (N={data['num_samples']})", fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

# Hide unused subplots if less than 6 techniques
for idx in range(len(technique_shap), 6):
    axes[idx].axis('off')

plt.suptitle('Top Malicious Tokens by Technique (SHAP Values)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()

plot_path = output_dir / "plots" / "shap_top_tokens_by_technique.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"    Saved: {plot_path}")


# ============================================================================
# Section 6: Command Pattern (N-gram) Analysis
# ============================================================================

print("\n" + "="*80)
print("COMMAND PATTERN (N-GRAM) ANALYSIS")
print("="*80)


def extract_important_ngrams(
    X_test: List[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    template_ids: List,
    test_templates: List[str],
    min_n: int = 3,
    max_n: int = 10,
    max_features: int = 1000
) -> Dict:
    """
    Extract important command substrings (character n-grams) that correlate
    with correct detections, grouped by technique.
    """
    # Only analyze correctly detected malicious samples
    correct_malicious_mask = (y_test == 1) & (y_pred == 1)
    correct_malicious_cmds = [cmd for cmd, correct in zip(X_test, correct_malicious_mask) if correct]
    correct_malicious_tids = [tid for tid, correct in zip(template_ids, correct_malicious_mask) if correct]

    if len(correct_malicious_cmds) == 0:
        return {}

    # Fit character n-gram vectorizer on correctly detected samples
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(min_n, max_n),
        max_features=max_features,
        min_df=5  # Must appear in at least 5 commands
    )

    print(f"[*] Extracting {min_n}-{max_n} character n-grams from {len(correct_malicious_cmds)} correctly detected commands...")
    X_ngrams = vectorizer.fit_transform(correct_malicious_cmds)
    ngram_features = vectorizer.get_feature_names_out()

    # Overall most frequent n-grams
    ngram_counts = np.array(X_ngrams.sum(axis=0)).flatten()
    top_overall_idx = np.argsort(ngram_counts)[-30:][::-1]
    top_overall = {str(ngram_features[i]): int(ngram_counts[i]) for i in top_overall_idx}

    # Per-technique n-grams
    technique_ngrams = {}

    for template_idx, template in enumerate(test_templates):
        techniques = classify_template_technique(template)
        binary = techniques['primary_binary']

        if binary not in technique_ngrams:
            technique_ngrams[binary] = []

        # Find commands from this template that were correctly detected
        template_cmd_indices = [
            i for i, tid in enumerate(correct_malicious_tids) if tid == template_idx
        ]

        if len(template_cmd_indices) > 0:
            # Get n-grams for these commands
            template_ngrams = X_ngrams[template_cmd_indices].sum(axis=0)
            technique_ngrams[binary].append(np.array(template_ngrams).flatten())

    # Aggregate per-technique top n-grams
    technique_top_ngrams = {}
    for binary, ngram_arrays in technique_ngrams.items():
        if len(ngram_arrays) == 0:
            continue

        combined = np.sum(ngram_arrays, axis=0)
        top_idx = np.argsort(combined)[-15:][::-1]
        technique_top_ngrams[binary] = {
            str(ngram_features[i]): int(combined[i]) for i in top_idx
        }

    return {
        'overall': top_overall,
        'by_technique': technique_top_ngrams
    }


ngram_results = extract_important_ngrams(
    X_test, y_test, y_pred, test_template_ids, test_templates,
    min_n=4, max_n=12, max_features=2000
)

print("\n[+] Top overall command substrings:")
for ngram, count in list(ngram_results['overall'].items())[:20]:
    print(f"  '{ngram}': {count}")

print("\n[+] Top command substrings by technique:")
for binary, ngrams in sorted(ngram_results['by_technique'].items()):
    print(f"\n{binary.upper()}:")
    for ngram, count in list(ngrams.items())[:10]:
        print(f"  '{ngram}': {count}")

# Save n-gram results
with open(output_dir / "important_ngrams.json", "w") as f:
    json.dump(ngram_results, f, indent=2)

print(f"\n[+] N-gram results saved to {output_dir / 'important_ngrams.json'}")


# ============================================================================
# Section 7: Generate LaTeX Tables and Outputs
# ============================================================================

print("\n" + "="*80)
print("GENERATING LATEX TABLES")
print("="*80)


def generate_latex_detection_table(df_by_technique: pd.DataFrame) -> str:
    """Generate LaTeX table for detection rates by technique."""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Detection rates by reverse shell technique type}")
    latex.append("\\label{tab:detection_by_technique}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Technique} & \\textbf{TPR} & \\textbf{Avg Score} & \\textbf{N Samples} \\\\")
    latex.append("\\midrule")

    for binary, row in df_by_technique.iterrows():
        tpr_pct = f"{row['tpr']*100:.1f}\\%"
        score = f"{row['mean_score']:.3f}"
        n = int(row['num_samples'])
        latex.append(f"{binary} & {tpr_pct} & {score} & {n:,} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_latex_shap_table(technique_shap: Dict, top_n: int = 10) -> str:
    """Generate LaTeX table for SHAP values by technique."""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Top tokens by SHAP importance for each reverse shell technique}")
    latex.append("\\label{tab:shap_by_technique}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lp{6cm}p{6cm}}")
    latex.append("\\toprule")
    latex.append("\\textbf{Technique} & \\textbf{Malicious Tokens (SHAP)} & \\textbf{Benign Tokens (SHAP)} \\\\")
    latex.append("\\midrule")

    for binary in sorted(technique_shap.keys()):
        data = technique_shap[binary]

        # Format malicious tokens
        mal_tokens = list(data['malicious'].items())[:top_n]
        mal_str = ", ".join([f"\\texttt{{{tok}}} ({val:.3f})" for tok, val in mal_tokens])

        # Format benign tokens
        ben_tokens = list(data['benign'].items())[:top_n]
        ben_str = ", ".join([f"\\texttt{{{tok}}} ({val:.3f})" for tok, val in ben_tokens])

        latex.append(f"{binary} & {mal_str} & {ben_str} \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_latex_ngrams_table(ngram_results: Dict, top_n: int = 15) -> str:
    """Generate LaTeX table for important n-grams by technique."""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Most frequent command substrings (n-grams) by technique}")
    latex.append("\\label{tab:ngrams_by_technique}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lp{10cm}}")
    latex.append("\\toprule")
    latex.append("\\textbf{Technique} & \\textbf{Command Substrings (frequency)} \\\\")
    latex.append("\\midrule")

    by_technique = ngram_results.get('by_technique', {})

    for binary in sorted(by_technique.keys()):
        ngrams = list(by_technique[binary].items())[:top_n]
        ngram_str = ", ".join([f"\\texttt{{{ng}}} ({cnt})" for ng, cnt in ngrams])
        latex.append(f"{binary} & {ngram_str} \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


# Generate all LaTeX tables
latex_detection = generate_latex_detection_table(by_technique)
latex_shap = generate_latex_shap_table(technique_shap, top_n=8)
latex_ngrams = generate_latex_ngrams_table(ngram_results, top_n=10)

# Save LaTeX outputs
with open(output_dir / "table_detection_by_technique.tex", "w") as f:
    f.write(latex_detection)

with open(output_dir / "table_shap_by_technique.tex", "w") as f:
    f.write(latex_shap)

with open(output_dir / "table_ngrams_by_technique.tex", "w") as f:
    f.write(latex_ngrams)

print("\n[+] LaTeX tables generated:")
print(f"  - {output_dir / 'table_detection_by_technique.tex'}")
print(f"  - {output_dir / 'table_shap_by_technique.tex'}")
print(f"  - {output_dir / 'table_ngrams_by_technique.tex'}")

print("\n" + "="*80)
print("SUMMARY FOR PAPER")
print("="*80)

print("\n## Key Findings for Explainability Section:\n")

# 1. Technique-level detection summary
print("### 1. Detection Rates by Technique:")
for binary, row in by_technique.head(5).iterrows():
    print(f"- {binary}: {row['tpr']*100:.1f}% TPR (N={int(row['num_samples'])})")

# 2. Command pattern insights
print("\n### 2. Important Command Patterns:")
print("Most discriminative command substrings across all techniques:")
for ngram, count in list(ngram_results['overall'].items())[:10]:
    print(f"- '{ngram}' (appears {count} times)")

# 3. Technique-specific insights
print("\n### 3. Technique-Specific Token Importance:")
for binary in list(technique_shap.keys())[:5]:
    data = technique_shap[binary]
    top_mal = list(data['malicious'].keys())[:3]
    top_ben = list(data['benign'].keys())[:3]
    print(f"\n{binary.upper()} (N={data['num_samples']}):")
    print(f"  Malicious indicators: {', '.join(top_mal)}")
    print(f"  Benign indicators: {', '.join(top_ben)}")

print("\n" + "="*80)
print(f"ANALYSIS COMPLETE - Results saved to {output_dir}")
print("="*80)
print(f"\n[!] Script end time: {time.ctime()}")
