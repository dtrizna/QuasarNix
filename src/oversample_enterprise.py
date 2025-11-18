#!/usr/bin/env python3
"""
Utility script to oversample enterprise benign baseline and regenerate
synthetic datasets at the scale reported in the paper.

It:
  1. Reads the raw enterprise command corpus from:
       data/nix_shell/bash_commands_private/auditd_msft_ArgsNormalizedUnique.cm
  2. Splits it into train/test prototype sets.
  3. Oversamples with replacement to target sizes roughly matching
     |X_train^{legit}| ~= 266k, |X_test^{legit}| ~= 235k.
  4. Writes oversampled enterprise baselines:
       enterprise_baseline_train_ovsampling.parquet
       enterprise_baseline_test_ovsampling.parquet
  5. Invokes the existing generate_synthetic_data(...) routine with
     paths pointing to *_ovsampling.parquet files to produce:
       train_baseline_real_ovsampling.parquet
       test_baseline_real_ovsampling.parquet
       train_rvrs_real_ovsampling.parquet
       test_rvrs_real_ovsampling.parquet

Core synthesis logic in src/augmentation.py and src/data_utils.py remains
unchanged; this script only prepares alternative datasets for dtype='oversampled'.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make sure repository root is on the path when running as a script
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import generate_synthetic_data  # type: ignore


SEED: int = 33

# Target baseline sizes, chosen to align with the magnitudes reported in the paper
TARGET_TRAIN_BASELINE: int = 266_000
TARGET_TEST_BASELINE: int = 235_000

RAW_ENTERPRISE_CM = ROOT / "data" / "nix_shell" / "bash_commands_private" / "auditd_msft_ArgsNormalizedUnique.cm"


def oversample_enterprise_baseline() -> tuple[Path, Path]:
    """
    Load the raw enterprise command corpus and oversample it into train/test
    baselines of approximately 266k / 235k commands, respectively.
    """
    if not RAW_ENTERPRISE_CM.exists():
        raise FileNotFoundError(
            f"Raw enterprise baseline file not found at '{RAW_ENTERPRISE_CM}'. "
            "Expected a file with one command per line."
        )

    with RAW_ENTERPRISE_CM.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Raw enterprise baseline file '{RAW_ENTERPRISE_CM}' is empty.")

    # Use a fixed split ratio roughly matching the original train/test sizes in the paper.
    train_proto, test_proto = train_test_split(lines, test_size=0.47, random_state=SEED)

    rng = np.random.default_rng(SEED)

    train_idx = rng.integers(0, len(train_proto), size=TARGET_TRAIN_BASELINE)
    test_idx = rng.integers(0, len(test_proto), size=TARGET_TEST_BASELINE)

    train_cmds = [train_proto[i] for i in train_idx]
    test_cmds = [test_proto[i] for i in test_idx]

    data_root = ROOT / "data" / "nix_shell"
    data_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"cmd": train_cmds})
    test_df = pd.DataFrame({"cmd": test_cmds})

    train_path = data_root / "enterprise_baseline_train_ovsampling.parquet"
    test_path = data_root / "enterprise_baseline_test_ovsampling.parquet"

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"[+] Oversampled enterprise baseline written to:\n  {train_path}\n  {test_path}")
    return train_path, test_path


def main() -> None:
    """
    Entry point: oversample enterprise baseline and regenerate synthetic
    train/test datasets using the existing generate_synthetic_data logic.
    """
    train_enterprise_path, test_enterprise_path = oversample_enterprise_baseline()

    data_root = ROOT / "data" / "nix_shell"

    paths = {
        "train_baseline": data_root / "train_baseline_real_ovsampling.parquet",
        "test_baseline": data_root / "test_baseline_real_ovsampling.parquet",
        "train_malicious": data_root / "train_rvrs_real_ovsampling.parquet",
        "test_malicious": data_root / "test_rvrs_real_ovsampling.parquet",
        "enterprise_baseline_train": train_enterprise_path,
        "enterprise_baseline_test": test_enterprise_path,
    }

    print("[*] Generating synthetic datasets for oversampled enterprise baseline...")
    generate_synthetic_data(paths=paths, root=ROOT, seed=SEED, baseline="real")
    print("[+] Synthetic datasets for oversampled baseline created.")


if __name__ == "__main__":
    main()


