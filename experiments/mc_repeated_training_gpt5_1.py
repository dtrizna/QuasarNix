import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from nltk.tokenize import wordpunct_tokenize

# Ensure project root is on sys.path so that internal modules using `src.*` imports resolve correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import load_data, load_tokenizer
from src.preprocessors import OneHotCustomVectorizer
from src.tabular_utils import training_tabular
from src.augmentation import (
    NixCommandAugmentationConfig,
    NixCommandAugmentationWithBaseline,
    read_template_file,
)
from src.scoring import get_tpr_at_fpr

timestamp = int(time.time())


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo repeated training."""

    num_runs: int = 3
    seeds: List[int] = None
    vocab_size: int = 4096
    max_len: int = 128
    logs_folder: Path = Path(f"experiments/logs_mc_stability_gpt5_{timestamp}")
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


def monte_carlo_xgb(mc_cfg: MonteCarloConfig) -> None:
    """
    Run Monte Carlo repeated training for XGB with different augmentation presets and seeds.
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
    tokenizer: OneHotCustomVectorizer = load_tokenizer(
        tokenizer_type="onehot",
        train_cmds=X_train_cmds,
        vocab_size=mc_cfg.vocab_size,
        max_len=mc_cfg.max_len,
        tokenizer_fn=wordpunct_tokenize,
        suffix="_mc_xgb",
        logs_folder=str(mc_cfg.logs_folder),
    )

    presets = build_augmentation_presets()

    results: List[Dict[str, Any]] = []

    for run_idx in range(mc_cfg.num_runs):
        seed = mc_cfg.seeds[run_idx]
        preset = presets[run_idx % len(presets)]

        print(f"\n[MC-XGB] Run {run_idx + 1}/{mc_cfg.num_runs} | seed={seed}")
        print(f"[MC-XGB] Using augmentation preset index {run_idx % len(presets)}")

        # ===== Variable components per run: malicious training synthesis + model seed =====
        # Determine how many synthetic examples per template to generate to roughly
        # match the original malignant class size.
        original_malicious_count = int(y_train.sum())
        templates_count = len(train_templates)
        per_template = max(1, original_malicious_count // templates_count)

        print(
            f"[MC-XGB] Generating malicious training data: "
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

        # Encode train/test with the fixed tokenizer
        X_train_encoded = tokenizer.transform(X_train_run)
        X_test_encoded = tokenizer.transform(X_test_fixed)

        # XGB model with run-specific random state
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )

        run_name = f"_tabular_xgb_onehot_mc_run_{run_idx}_seed_{seed}"
        run_logs_folder = mc_cfg.logs_folder

        start = time.time()
        trained_model = training_tabular(
            model=xgb_model,
            name=run_name,
            X_train_encoded=X_train_encoded,
            X_test_encoded=X_test_encoded,
            y_train=y_train_run,
            y_test=y_test_fixed,
            logs_folder=str(run_logs_folder),
        )
        elapsed = time.time() - start
        print(f"[MC-XGB] Run {run_idx + 1} training finished in {elapsed:.2f}s")

        # Collect simple summary metrics for this run
        y_test_scores = trained_model.predict_proba(X_test_encoded)[:, 1]
        y_test_pred = (y_test_scores >= 0.5).astype(int)

        # TPR at ultra-low FPR=1e-6 (consistent with main paper evaluation)
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
    summary_path = mc_cfg.logs_folder / "mc_xgb_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"[MC-XGB] Monte Carlo summary saved to '{summary_path}'")

    # Print aggregate metrics (mean ± std) across runs in a paper-ready style.
    print("\n[MC-XGB] Aggregate metrics across runs (values in % where applicable):")
    metric_columns = ["test_auc", "test_f1", "test_acc", "tpr_at_1e6"]
    for col in metric_columns:
        mean_val = 100.0 * results_df[col].mean()
        std_val = 100.0 * results_df[col].std()
        print(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")

    # Elapsed time summary (seconds)
    if "elapsed_sec" in results_df.columns:
        mean_time = results_df["elapsed_sec"].mean()
        std_time = results_df["elapsed_sec"].std()
        print(f"  elapsed_sec: {mean_time:.2f} ± {std_time:.2f} s")


if __name__ == "__main__":
    cfg = MonteCarloConfig()
    monte_carlo_xgb(cfg)
