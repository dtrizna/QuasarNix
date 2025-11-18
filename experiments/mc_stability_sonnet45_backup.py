"""
Monte Carlo Repeated Training for Stability Analysis
Claude Sonnet 4.5 - 2025

This script performs Monte Carlo repeated training experiments to assess:
1. Synthesis variance (placeholder sampling)
2. Training variance (model initialization, batch order, dropout)

FIXED components (by design):
- Template split: train templates vs test templates (prevents data leakage)
- Temporal baseline split: enterprise data collected at T (train) and T+1month (test)

VARIABLE components (across runs):
- Synthesis randomness: placeholder values (IPs, ports, variable names, etc.)
- Training randomness: model weights, batch shuffling, dropout masks
"""

import os
import sys
import time
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from nltk.tokenize import wordpunct_tokenize

import torch

# Add root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.augmentation import NixCommandAugmentationConfig, read_template_file, NixCommandAugmentationWithBaseline
from src.preprocessors import OneHotCustomVectorizer, CommandTokenizer
from src.data_utils import create_dataloader, commands_to_loader
from src.models import *
from src.lit_utils import train_lit_model
from src.tabular_utils import training_tabular
from watermark import watermark
from lightning.fabric.utilities.seed import seed_everything

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NUM_RUNS = 3

@dataclass
class MCExperimentConfig:
    """Configuration for a single MC run."""
    run_id: int
    seed: int
    synthesis_config: NixCommandAugmentationConfig
    description: str = ""

# ==============================================================================
# FIXED COMPONENTS (Preserved Across All Runs)
# ==============================================================================

# These paths are FIXED - templates and temporal baseline splits never change
FIXED_PATHS = {
    'train_templates': ROOT / 'data' / 'nix_shell' / 'templates_train.txt',
    'test_templates': ROOT / 'data' / 'nix_shell' / 'templates_test.txt',
    'enterprise_baseline_train': ROOT / 'data' / 'nix_shell' / 'enterprise_baseline_train.parquet',
    'enterprise_baseline_test': ROOT / 'data' / 'nix_shell' / 'enterprise_baseline_test.parquet',
}

# Model hyperparameters (fixed for reproducibility)
MODEL_CONFIG = {
    'vocab_size': 4096,
    'embedded_dim': 64,
    'max_len': 128,
    'batch_size': 1024,
    'dropout': 0.5,
    'learning_rate': 1e-3,
    'scheduler': 'onecycle',
    'epochs': 20,
    'dataloader_workers': 4,
    # Tabular model params
    'xgb_params': {
        'n_estimators': 100,
        'max_depth': 10,
    },
    'rf_params': {
        'n_estimators': 100,
        'max_depth': 10,
    }
}

# Models to train (set to None to train all, or specify list of model names)
MODELS_TO_TRAIN = [
    '_tabular_xgb_onehot',  # GBDT (XGBoost)
    '_tabular_rf_onehot',   # Random Forest
    '_tabular_mlp_onehot',  # MLP (One-Hot)
    'cnn',                   # 1D-CNN
]

# Uncomment to train all models
# MODELS_TO_TRAIN = None

# ==============================================================================
# SYNTHESIS CONFIGURATIONS (Variable Across Runs)
# ==============================================================================

def create_synthesis_configs() -> List[MCExperimentConfig]:
    """Create 3 different synthesis configurations with different random seeds."""

    # Config 1: Default (baseline from paper)
    config1 = NixCommandAugmentationConfig(
        nix_shells=["sh", "bash", "dash"],
        nix_shell_folders=["/bin/", "/usr/bin/"],
        default_filepaths=["/tmp/f", "/tmp/t"],
        path_roots=["/tmp/", "/home/user/", "/var/www/"],
        folder_lengths=[1, 8],
        nr_of_random_filepaths=5,
        default_variable_names=["port", "host", "cmd", "p", "s", "c"],
        nr_of_random_variables=5,
        default_ips=["127.0.0.1"],
        nr_of_random_ips=5,
        default_ports=[8080, 9001, 80, 443, 53, 22, 8000, 8888],
        nr_of_random_ports=5,
        default_fd_number=3,
        nr_of_random_fd_numbers=5,
    )

    # Config 2: More diverse paths and variables
    config2 = NixCommandAugmentationConfig(
        nix_shells=["sh", "bash", "dash"],
        nix_shell_folders=["/bin/", "/usr/bin/"],
        default_filepaths=["/tmp/f", "/tmp/t", "/dev/shm/x"],
        path_roots=["/tmp/", "/home/user/", "/var/www/", "/dev/shm/"],
        folder_lengths=[1, 8, 12],
        nr_of_random_filepaths=8,
        default_variable_names=["port", "host", "cmd", "p", "s", "c", "sock", "conn"],
        nr_of_random_variables=8,
        default_ips=["127.0.0.1", "192.168.1.100"],
        nr_of_random_ips=8,
        default_ports=[8080, 9001, 80, 443, 53, 22, 8000, 8888, 4444, 1337],
        nr_of_random_ports=8,
        default_fd_number=3,
        nr_of_random_fd_numbers=5,
    )

    # Config 3: Conservative (less randomness, more threat intel focus)
    config3 = NixCommandAugmentationConfig(
        nix_shells=["sh", "bash", "dash"],
        nix_shell_folders=["/bin/", "/usr/bin/"],
        default_filepaths=["/tmp/f", "/tmp/t"],
        path_roots=["/tmp/", "/home/user/"],
        folder_lengths=[1, 6],
        nr_of_random_filepaths=3,
        default_variable_names=["port", "host", "cmd", "p"],
        nr_of_random_variables=3,
        default_ips=["127.0.0.1"],
        nr_of_random_ips=3,
        default_ports=[8080, 9001, 80, 443, 4444],
        nr_of_random_ports=3,
        default_fd_number=3,
        nr_of_random_fd_numbers=3,
    )

    # Create NUM_RUNS runs with varying seeds and rotating configs
    configs = []
    seeds = [33 + i for i in range(NUM_RUNS)]
    synthesis_configs = [config1, config2, config3]

    for i, seed in enumerate(seeds):
        config = MCExperimentConfig(
            run_id=i,
            seed=seed,
            synthesis_config=synthesis_configs[i % 3],
            description=f"Synthesis Config {(i % 3) + 1}, Seed {seed}"
        )
        configs.append(config)

    return configs

# ==============================================================================
# DATA GENERATION (Variable Synthesis)
# ==============================================================================

def generate_data_for_run(run_config: MCExperimentConfig, output_dir: Path) -> Dict[str, Path]:
    """
    Generate synthetic data for a single MC run.

    FIXED:
    - Template split (train vs test)
    - Enterprise baseline split (temporal)

    VARIES:
    - Placeholder sampling (controlled by seed and synthesis_config)
    """
    print(f"\n{'='*80}")
    print(f"[Run {run_config.run_id}] Generating data with seed={run_config.seed}")
    print(f"[Run {run_config.run_id}] {run_config.description}")
    print(f"{'='*80}\n")

    # Create run-specific paths
    run_dir = output_dir / f"run_{run_config.run_id:02d}_seed_{run_config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {
        'train_malicious': run_dir / 'train_rvrs_real.parquet',
        'test_malicious': run_dir / 'test_rvrs_real.parquet',
        'train_baseline': run_dir / 'train_baseline_real.parquet',
        'test_baseline': run_dir / 'test_baseline_real.parquet',
    }

    # Save config
    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'run_id': run_config.run_id,
            'seed': run_config.seed,
            'description': run_config.description,
            'synthesis_config': asdict(run_config.synthesis_config),
        }, f, indent=2)

    # ======== FIXED: Load enterprise baselines (temporal split) ========
    train_baseline = pd.read_parquet(FIXED_PATHS['enterprise_baseline_train'])
    test_baseline = pd.read_parquet(FIXED_PATHS['enterprise_baseline_test'])

    # ======== FIXED: Read template files ========
    train_templates = read_template_file(FIXED_PATHS['train_templates'])
    test_templates = read_template_file(FIXED_PATHS['test_templates'])

    # ======== VARIES: Create generators with seed and config ========
    train_gen = NixCommandAugmentationWithBaseline(
        templates=train_templates,
        legitimate_baseline=train_baseline['cmd'].tolist(),
        random_state=run_config.seed,
        config=run_config.synthesis_config,
    )

    test_gen = NixCommandAugmentationWithBaseline(
        templates=test_templates,
        legitimate_baseline=test_baseline['cmd'].tolist(),
        random_state=run_config.seed + 1,  # different seed for test set
        config=run_config.synthesis_config,
    )

    # Determine number of examples per template for class balance
    train_per_template = max(1, len(train_baseline) // len(train_templates))
    test_per_template = max(1, len(test_baseline) // len(test_templates))

    print(f"[+] Generating {train_per_template} train examples per template ({len(train_templates)} templates)")
    train_malicious = train_gen.generate_commands(train_per_template)

    print(f"[+] Generating {test_per_template} test examples per template ({len(test_templates)} templates)")
    test_malicious = test_gen.generate_commands(test_per_template)

    # ======== Save to parquet ========
    train_malicious_df = pd.DataFrame({"cmd": train_malicious})
    test_malicious_df = pd.DataFrame({"cmd": test_malicious})

    train_malicious_df.to_parquet(data_paths['train_malicious'])
    test_malicious_df.to_parquet(data_paths['test_malicious'])
    train_baseline.to_parquet(data_paths['train_baseline'])
    test_baseline.to_parquet(data_paths['test_baseline'])

    print(f"[+] Synthetic data saved to {run_dir}\n")

    return data_paths

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def get_all_models(seed: int) -> Dict[str, Any]:
    """Initialize all models with specified seed."""
    cfg = MODEL_CONFIG

    # Sequential models (embedding-based)
    mlp_seq_model = SimpleMLPWithEmbedding(
        vocab_size=cfg['vocab_size'], embedding_dim=cfg['embedded_dim'],
        output_dim=1, hidden_dim=[256, 64, 32], use_positional_encoding=False,
        max_len=cfg['max_len'], dropout=cfg['dropout']
    )
    cnn_model = CNN1DGroupedModel(
        vocab_size=cfg['vocab_size'], embed_dim=cfg['embedded_dim'],
        num_channels=32, kernel_sizes=[2, 3, 4, 5], mlp_hidden_dims=[64, 32],
        output_dim=1, dropout=cfg['dropout']
    )
    lstm_model = BiLSTMModel(
        vocab_size=cfg['vocab_size'], embed_dim=cfg['embedded_dim'],
        hidden_dim=32, mlp_hidden_dims=[64, 32], output_dim=1, dropout=cfg['dropout']
    )
    cnn_lstm_model = CNN1D_BiLSTM_Model(
        vocab_size=cfg['vocab_size'], embed_dim=cfg['embedded_dim'],
        num_channels=32, kernel_size=3, lstm_hidden_dim=32,
        mlp_hidden_dims=[64, 32], output_dim=1, dropout=cfg['dropout']
    )
    mean_transformer_model = MeanTransformerEncoder(
        vocab_size=cfg['vocab_size'], d_model=cfg['embedded_dim'],
        nhead=4, num_layers=2, dim_feedforward=128, max_len=cfg['max_len'],
        dropout=cfg['dropout'], mlp_hidden_dims=[64,32], output_dim=1
    )
    cls_transformer_model = CLSTransformerEncoder(
        vocab_size=cfg['vocab_size'], d_model=cfg['embedded_dim'],
        nhead=4, num_layers=2, dim_feedforward=128, max_len=cfg['max_len'],
        dropout=cfg['dropout'], mlp_hidden_dims=[64,32], output_dim=1
    )
    attpool_transformer_model = AttentionPoolingTransformerEncoder(
        vocab_size=cfg['vocab_size'], d_model=cfg['embedded_dim'],
        nhead=4, num_layers=2, dim_feedforward=128, max_len=cfg['max_len'],
        dropout=cfg['dropout'], mlp_hidden_dims=[64,32], output_dim=1
    )
    neurlux = NeurLuxModel(
        vocab_size=cfg['vocab_size'], embed_dim=cfg['embedded_dim'],
        max_len=cfg['max_len'], hidden_dim=32, output_dim=1, dropout=cfg['dropout']
    )

    # Tabular models
    rf_model_onehot = RandomForestClassifier(**cfg['rf_params'], random_state=seed)
    xgb_model_onehot = XGBClassifier(**cfg['xgb_params'], random_state=seed)
    log_reg_onehot = LogisticRegression(random_state=seed)
    mlp_tab_model_onehot = SimpleMLP(
        input_dim=cfg['vocab_size'], output_dim=1,
        hidden_dim=[64, 32], dropout=cfg['dropout']
    )

    models = {
        "_tabular_mlp_onehot": mlp_tab_model_onehot,
        "_tabular_rf_onehot": rf_model_onehot,
        "_tabular_xgb_onehot": xgb_model_onehot,
        "_tabular_log_reg_onehot": log_reg_onehot,
        "mlp_seq": mlp_seq_model,
        "attpool_transformer": attpool_transformer_model,
        "cls_transformer": cls_transformer_model,
        "mean_transformer": mean_transformer_model,
        "neurlux": neurlux,
        "cnn": cnn_model,
        "lstm": lstm_model,
        "cnn_lstm": cnn_lstm_model,
    }

    return models

def train_model(
    model_name: str,
    model: Any,
    X_train_cmds: List[str],
    y_train: np.ndarray,
    X_train_onehot: np.ndarray,
    X_train_loader: Any,
    seed: int,
    run_dir: Path
) -> Any:
    """Train a single model (tabular or sequential)."""
    cfg = MODEL_CONFIG
    start = time.time()

    if model_name.startswith("_tabular"):
        # Tabular models use One-Hot encoding
        if "_mlp_" in model_name:
            # MLP needs DataLoader
            train_loader = create_dataloader(
                X_train_onehot, y_train,
                batch_size=cfg['batch_size'],
                workers=cfg['dataloader_workers'],
                shuffle=True
            )
            seed_everything(seed)
            trained_model = train_lit_model(
                train_loader, None, model, model_name,
                log_folder=run_dir,
                epochs=cfg['epochs'],
                learning_rate=cfg['learning_rate'],
                scheduler=cfg['scheduler'],
                scheduler_budget=cfg['epochs'] * len(train_loader)
            )
        else:
            # RF, XGB, LogReg use sklearn API
            seed_everything(seed)
            trained_model = training_tabular(
                model, model_name,
                X_train_onehot, None,  # No test set during training
                y_train, None,
                logs_folder=run_dir
            )
    else:
        # Sequential models use token embeddings
        seed_everything(seed)
        trained_model = train_lit_model(
            X_train_loader, None, model, model_name,
            log_folder=run_dir,
            epochs=cfg['epochs'],
            learning_rate=cfg['learning_rate'],
            scheduler=cfg['scheduler'],
            scheduler_budget=cfg['epochs'] * len(X_train_loader)
        )

    elapsed = time.time() - start
    print(f"[!] Training {model_name} completed in {elapsed:.2f}s")
    return trained_model

def prepare_all_features(
    X_train_cmds: List[str],
    X_test_cmds: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    run_dir: Path,
    seed: int
) -> Dict[str, Any]:
    """Prepare all feature encodings (One-Hot and Embeddings)."""
    cfg = MODEL_CONFIG
    features = {}

    # === ONE-HOT ENCODING (for tabular models) ===
    onehot_tokenizer_path = run_dir / f'tokenizer_onehot_{cfg["vocab_size"]}.pkl'
    if onehot_tokenizer_path.exists():
        print(f"[!] Loading One-Hot tokenizer from {onehot_tokenizer_path}")
        with open(onehot_tokenizer_path, 'rb') as f:
            oh_tokenizer = pickle.load(f)
    else:
        print(f"[*] Creating One-Hot tokenizer...")
        oh_tokenizer = OneHotCustomVectorizer(
            tokenizer=wordpunct_tokenize,
            max_features=cfg['vocab_size']
        )
        oh_tokenizer.fit(X_train_cmds)
        with open(onehot_tokenizer_path, 'wb') as f:
            pickle.dump(oh_tokenizer, f)

    print(f"[*] Transforming One-Hot features...")
    features['X_train_onehot'] = oh_tokenizer.transform(X_train_cmds)
    features['X_test_onehot'] = oh_tokenizer.transform(X_test_cmds)

    # === EMBEDDING ENCODING (for sequential models) ===
    embed_tokenizer_path = run_dir / f'tokenizer_embed_{cfg["vocab_size"]}.pkl'
    if embed_tokenizer_path.exists():
        print(f"[!] Loading Embedding tokenizer from {embed_tokenizer_path}")
        with open(embed_tokenizer_path, 'rb') as f:
            embed_tokenizer = pickle.load(f)
    else:
        print(f"[*] Creating Embedding tokenizer...")
        embed_tokenizer = CommandTokenizer(
            tokenizer_fn=wordpunct_tokenize,
            vocab_size=cfg['vocab_size'],
            max_len=cfg['max_len']
        )
        X_train_tokens = embed_tokenizer.tokenize(X_train_cmds)
        embed_tokenizer.build_vocab(X_train_tokens)
        with open(embed_tokenizer_path, 'wb') as f:
            pickle.dump(embed_tokenizer, f)

    print(f"[*] Creating DataLoaders for sequential models...")
    features['X_train_loader'] = commands_to_loader(
        X_train_cmds, embed_tokenizer,
        y=y_train, workers=cfg['dataloader_workers'],
        batch_size=cfg['batch_size'], shuffle=True
    )
    features['X_test_loader'] = commands_to_loader(
        X_test_cmds, embed_tokenizer,
        y=y_test, workers=cfg['dataloader_workers'],
        batch_size=cfg['batch_size'], shuffle=False
    )

    return features

# ==============================================================================
# EVALUATION
# ==============================================================================

def get_tpr_at_fpr(y_true, y_pred_proba, fpr_target=1e-6):
    """Calculate TPR at specific FPR threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Handle edge case where FPR never reaches target
    valid_idx = np.where(fpr <= fpr_target)[0]
    if len(valid_idx) == 0:
        return 0.0

    return tpr[valid_idx[-1]]

def evaluate_tabular_model(model, model_name: str, X_test, y_test, fpr_targets=[1e-7, 1e-6, 1e-5, 1e-4]):
    """Evaluate tabular model (sklearn/xgboost)."""
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test)

    results = {}
    for fpr in fpr_targets:
        tpr = get_tpr_at_fpr(y_test, y_pred_proba, fpr)
        results[f'tpr_at_fpr_{fpr:.0e}'] = tpr * 100

    return results, y_pred_proba

def evaluate_sequential_model(model, model_name: str, test_loader, y_test, fpr_targets=[1e-7, 1e-6, 1e-5, 1e-4]):
    """Evaluate sequential model (PyTorch)."""
    import lightning as L
    from src.lit_utils import LitProgressBar

    # Configure trainer for inference
    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[LitProgressBar()],
        logger=False,
        enable_checkpointing=False,
    )

    # Get predictions
    predictions = trainer.predict(model, test_loader)
    y_pred_logits = torch.cat([p for p in predictions]).cpu().numpy().squeeze()
    y_pred_proba = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()

    results = {}
    for fpr in fpr_targets:
        tpr = get_tpr_at_fpr(y_test, y_pred_proba, fpr)
        results[f'tpr_at_fpr_{fpr:.0e}'] = tpr * 100

    return results, y_pred_proba

def load_and_evaluate_model(
    model_name: str,
    model,
    features: Dict,
    y_test: np.ndarray,
    run_dir: Path,
    fpr_targets=[1e-7, 1e-6, 1e-5, 1e-4]
) -> tuple:
    """Load trained model and evaluate."""

    if model_name.startswith("_tabular"):
        # Load tabular model
        if "_mlp_" in model_name:
            # Load PyTorch MLP from checkpoint
            checkpoint_dir = run_dir / f"{model_name}_csv" / "version_0" / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
                if checkpoint_files:
                    from src.lit_utils import PyTorchLightningModel
                    loaded_model = PyTorchLightningModel.load_from_checkpoint(
                        checkpoint_path=str(checkpoint_files[0]),
                        model=model,
                        learning_rate=MODEL_CONFIG['learning_rate']
                    )
                    loaded_model.eval()

                    # Create test loader for MLP
                    test_loader = create_dataloader(
                        features['X_test_onehot'], y_test,
                        batch_size=MODEL_CONFIG['batch_size'],
                        workers=MODEL_CONFIG['dataloader_workers'],
                        shuffle=False
                    )
                    return evaluate_sequential_model(loaded_model, model_name, test_loader, y_test, fpr_targets)
        else:
            # Load sklearn model
            model_file = run_dir / model_name / "model.pkl"
            if "_xgb_" in model_name:
                model_file = run_dir / model_name / "model.xgboost"

            if model_file.exists():
                if "_xgb_" in model_name:
                    loaded_model = XGBClassifier()
                    loaded_model.load_model(str(model_file))
                else:
                    with open(model_file, 'rb') as f:
                        loaded_model = pickle.load(f)

                return evaluate_tabular_model(loaded_model, model_name, features['X_test_onehot'], y_test, fpr_targets)
    else:
        # Load sequential model (CNN, LSTM, etc)
        checkpoint_dir = run_dir / f"{model_name}_csv" / "version_0" / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoint_files:
                from src.lit_utils import PyTorchLightningModel
                loaded_model = PyTorchLightningModel.load_from_checkpoint(
                    checkpoint_path=str(checkpoint_files[0]),
                    model=model,
                    learning_rate=MODEL_CONFIG['learning_rate']
                )
                loaded_model.eval()
                return evaluate_sequential_model(loaded_model, model_name, features['X_test_loader'], y_test, fpr_targets)

    # If model not found, return zeros
    print(f"[WARNING] Model {model_name} not found, returning zero metrics")
    results = {f'tpr_at_fpr_{fpr:.0e}': 0.0 for fpr in fpr_targets}
    return results, np.zeros(len(y_test))

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_mc_experiment(output_base_dir: Path):
    """
    Run Monte Carlo repeated training experiment.

    For each run:
    1. FIXED: Load same train/test templates
    2. FIXED: Load same enterprise baseline split
    3. VARIES: Re-synthesize with new seed and config
    4. VARIES: Train model with new seed
    5. Evaluate and store results
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Verify fixed paths exist
    for name, path in FIXED_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Required fixed path does not exist: {path}")

    print(f"\n{'='*80}")
    print(f"MONTE CARLO STABILITY EXPERIMENT")
    print(f"{'='*80}")
    print(f"Output directory: {output_base_dir}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"\nFIXED components:")
    for name, path in FIXED_PATHS.items():
        print(f"  - {name}: {path}")
    print(f"\nVARIABLE components:")
    print(f"  - Synthesis randomness (placeholder sampling)")
    print(f"  - Training randomness (model initialization)")
    print(f"{'='*80}\n")

    # Create experiment configs
    configs = create_synthesis_configs()[:NUM_RUNS]

    # Storage for results
    all_results = []
    all_predictions = {}

    # Run experiments
    for config in configs:
        run_start = time.time()

        print(f"\n[DEBUG] Starting run {config.run_id} with seed {config.seed}")

        # 1. Generate data (VARIES: synthesis with different seed/config)
        run_dir = output_base_dir / f"run_{config.run_id:02d}_seed_{config.seed}"
        print(f"[DEBUG] Run directory: {run_dir}")
        data_paths = generate_data_for_run(config, output_base_dir)
        print(f"[DEBUG] Data generation complete")

        # 2. Load data from run-specific directory
        train_baseline = pd.read_parquet(data_paths['train_baseline'])
        test_baseline = pd.read_parquet(data_paths['test_baseline'])
        train_malicious = pd.read_parquet(data_paths['train_malicious'])
        test_malicious = pd.read_parquet(data_paths['test_malicious'])

        X_train_baseline = train_baseline['cmd'].tolist()
        X_train_malicious = train_malicious['cmd'].tolist()
        X_test_baseline = test_baseline['cmd'].tolist()
        X_test_malicious = test_malicious['cmd'].tolist()

        # Combine and create labels
        from sklearn.utils import shuffle
        X_train_cmds = X_train_baseline + X_train_malicious
        y_train = np.array([0] * len(X_train_baseline) + [1] * len(X_train_malicious), dtype=np.int8)
        X_train_cmds, y_train = shuffle(X_train_cmds, y_train, random_state=config.seed)

        X_test_cmds = X_test_baseline + X_test_malicious
        y_test = np.array([0] * len(X_test_baseline) + [1] * len(X_test_malicious), dtype=np.int8)
        X_test_cmds, y_test = shuffle(X_test_cmds, y_test, random_state=config.seed)

        print(f"[Run {config.run_id}/{NUM_RUNS}] Loaded {len(X_train_cmds)} train, {len(X_test_cmds)} test samples")

        # 3. Prepare features
        tokenizer_path = run_dir / f'tokenizer_onehot_{MODEL_CONFIG["vocab_size"]}.pkl'
        X_train, X_test, tokenizer = prepare_features(
            X_train_cmds,
            X_test_cmds,
            tokenizer_path,
            config.seed
        )

        # 4. Train model (VARIES: initialization with different seed)
        model = train_xgb_model(X_train, y_train, config.seed)

        # 5. Evaluate
        metrics, y_pred_proba = evaluate_model(model, X_test, y_test)

        # 6. Store results
        result_entry = {
            'run_id': config.run_id,
            'seed': config.seed,
            'description': config.description,
            'train_samples': len(X_train_cmds),
            'test_samples': len(X_test_cmds),
            'runtime_seconds': time.time() - run_start,
            **metrics
        }
        all_results.append(result_entry)
        all_predictions[f'run_{config.run_id:02d}'] = y_pred_proba

        # Save run-specific results
        pd.DataFrame([result_entry]).to_csv(run_dir / 'metrics.csv', index=False)

        print(f"\n[Run {config.run_id}] Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}%")
        print(f"  Runtime: {result_entry['runtime_seconds']:.2f}s\n")

    # ==============================================================================
    # AGGREGATE RESULTS
    # ==============================================================================

    results_df = pd.DataFrame(all_results)

    # Calculate statistics
    metrics_cols = [col for col in results_df.columns if col.startswith('tpr_at_fpr_')]
    stats = []

    for metric in metrics_cols:
        stats.append({
            'metric': metric,
            'mean': results_df[metric].mean(),
            'std': results_df[metric].std(),
            'min': results_df[metric].min(),
            'max': results_df[metric].max(),
            'ci_lower': results_df[metric].quantile(0.025),
            'ci_upper': results_df[metric].quantile(0.975),
        })

    stats_df = pd.DataFrame(stats)

    # Save results
    results_df.to_csv(output_base_dir / f'all_runs_results_{NUM_RUNS}.csv', index=False)
    stats_df.to_csv(output_base_dir / 'aggregated_statistics.csv', index=False)

    with open(output_base_dir / 'all_predictions.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY - DETAILED STATISTICS")
    print(f"{'='*80}\n")
    print(stats_df.to_string(index=False))
    print(f"\n{'='*80}\n")

    # ==============================================================================
    # PAPER-READY FORMAT: Mean ± Std for Main Results Table
    # ==============================================================================

    print(f"{'='*80}")
    print(f"PAPER-READY FORMAT (for main results table)")
    print(f"{'='*80}\n")
    print(f"Model: GBDT (XGBoost)")
    print(f"Training: {len(configs)} independent runs with different synthesis configs and random seeds\n")

    # Format each metric as "mean ± std"
    for metric in metrics_cols:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        ci_lower = results_df[metric].quantile(0.025)
        ci_upper = results_df[metric].quantile(0.975)

        # Extract FPR value from metric name (e.g., 'tpr_at_fpr_1e-06' -> '10^-6')
        fpr_str = metric.replace('tpr_at_fpr_', '')
        # Convert scientific notation to latex format
        if 'e-0' in fpr_str:
            exp = fpr_str.split('e-0')[-1]
            fpr_latex = f"10^-{exp}"
        else:
            fpr_latex = fpr_str

        print(f"TPR @ FPR={fpr_latex:>6}: {mean_val:5.2f}% ± {std_val:4.2f}%  (95% CI: [{ci_lower:5.2f}%, {ci_upper:5.2f}%])")

    print(f"\n{'='*80}")
    print(f"TABLE FORMAT (copy-paste ready):")
    print(f"{'='*80}\n")

    # Create table-ready format
    print("Metric                    | Mean ± Std      | 95% CI")
    print("-" * 60)
    for metric in sorted(metrics_cols, reverse=True):  # Sort by FPR (descending)
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        ci_lower = results_df[metric].quantile(0.025)
        ci_upper = results_df[metric].quantile(0.975)

        fpr_str = metric.replace('tpr_at_fpr_', '')
        if 'e-0' in fpr_str:
            exp = fpr_str.split('e-0')[-1]
            metric_name = f"TPR @ FPR=10^-{exp}"
        else:
            metric_name = f"TPR @ FPR={fpr_str}"

        print(f"{metric_name:25} | {mean_val:5.2f} ± {std_val:4.2f}% | [{ci_lower:5.2f}, {ci_upper:5.2f}]")

    print(f"\n{'='*80}\n")

    return results_df, stats_df, all_predictions

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    timestamp = int(time.time())
    output_dir = ROOT / "experiments" / f"logs_mc_stability_sonnet45_{timestamp}"

    print(watermark(packages="numpy,pandas,sklearn,xgboost", python=True))
    print(f"\n[!] Experiment start time: {time.ctime()}\n")

    results_df, stats_df, predictions = run_mc_experiment(output_dir)

    print(f"\n[!] Experiment end time: {time.ctime()}")
    print(f"[!] Results saved to: {output_dir}")
