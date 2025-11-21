from __future__ import annotations

"""
Ad-hoc script to compute diversity and coverage statistics
for synthesized reverse-shell commands used in QuasarNix.

It inspects the parquet files produced by the data synthesis
pipeline and reports:
  - total vs. unique malicious commands in train and test;
  - per-template diversity statistics for the training split.

These metrics are used to support the discussion in Sec. 4.2.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import re
import sys

import pandas as pd

# Ensure project root is on sys.path so that `src` can be imported when this
# script is executed as a standalone module.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.augmentation import read_template_file, NixCommandAugmentationConfig


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file, supporting both single-file and directory layouts."""
    if path.is_dir():
        files = list(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under directory {path}")
        return pd.read_parquet(files[0])
    if not path.exists():
        raise FileNotFoundError(f"Parquet file '{path}' does not exist")
    return pd.read_parquet(path)


def compute_global_stats(
    cmds: List[str],
) -> Tuple[int, int, float]:
    """Return total, unique counts and uniqueness ratio."""
    total = len(cmds)
    unique = len(set(cmds))
    ratio = unique / total if total else 0.0
    return total, unique, ratio


def compute_per_template_stats(
    cmds: List[str],
    templates: List[str],
) -> Dict[str, float]:
    """
    Approximate per-template diversity assuming commands are generated
    in contiguous blocks per template, as in generate_synthetic_data().
    """
    n_templates = len(templates)
    if n_templates == 0:
        raise ValueError("Template list is empty")

    total = len(cmds)
    base = total // n_templates
    remainder = total % n_templates

    per_template_unique: List[int] = []
    start = 0
    for i in range(n_templates):
        extra = 1 if i < remainder else 0
        end = start + base + extra
        subset = cmds[start:end]
        per_template_unique.append(len(set(subset)))
        start = end

    if not per_template_unique:
        raise ValueError("No per-template slices computed")

    per_template_unique_sorted = sorted(per_template_unique)
    median = per_template_unique_sorted[len(per_template_unique_sorted) // 2]

    return {
        "templates": n_templates,
        "per_template_min_unique": float(min(per_template_unique)),
        "per_template_median_unique": float(median),
        "per_template_max_unique": float(max(per_template_unique)),
    }


def compute_placeholder_value_counts(cmds: List[str]) -> Dict[str, int]:
    """
    Approximate diversity of placeholder value instantiations by counting
    distinct shells, IP addresses, ports, file paths and variable names
    observed across synthesized malicious commands.
    """
    # Reconstruct the default shell list used by the augmentation config
    cfg = NixCommandAugmentationConfig()
    shell_candidates: List[str] = []
    for sh in cfg.nix_shells:
        shell_candidates.append(sh)
        for folder in cfg.nix_shell_folders:
            shell_candidates.append(folder + sh)
    shell_candidates = list(set(shell_candidates))

    shells = set()
    ips = set()
    ports = set()
    paths = set()
    vars_ = set()

    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    port_pattern = re.compile(r":(\d{2,5})\b")
    path_pattern = re.compile(r"(?:^|\s)(/[\w/.\-]+)")
    var_pattern = re.compile(r"\$(\w+)")

    for cmd in cmds:
        # Shell binaries: simple whitespace tokenisation is sufficient here
        for tok in cmd.split():
            if tok in shell_candidates:
                shells.add(tok)

        ips.update(ip_pattern.findall(cmd))
        ports.update(port_pattern.findall(cmd))
        paths.update(path_pattern.findall(cmd))
        vars_.update(var_pattern.findall(cmd))

    return {
        "shells": len(shells),
        "ips": len(ips),
        "ports": len(ports),
        "paths": len(paths),
        "vars": len(vars_),
    }


def main(baseline: str = "real") -> None:
    data_root = ROOT / "data" / "nix_shell"

    train_templates = read_template_file(data_root / "templates_train.txt")
    test_templates = read_template_file(data_root / "templates_test.txt")

    train_mal_path = data_root / f"train_rvrs_{baseline}.parquet"
    test_mal_path = data_root / f"test_rvrs_{baseline}.parquet"

    print(f"[*] Using data root: {data_root}")
    print(f"[*] Loading malicious train from: {train_mal_path}")
    print(f"[*] Loading malicious test  from: {test_mal_path}")

    train_mal_df = _load_parquet(train_mal_path)
    test_mal_df = _load_parquet(test_mal_path)

    train_cmds = train_mal_df["cmd"].astype(str).tolist()
    test_cmds = test_mal_df["cmd"].astype(str).tolist()

    # Global uniqueness
    tr_total, tr_unique, tr_ratio = compute_global_stats(train_cmds)
    te_total, te_unique, te_ratio = compute_global_stats(test_cmds)

    print("\n=== Global malicious diversity ===")
    print(f"Train malicious: total={tr_total}, unique={tr_unique}, ratio={tr_ratio:.4f}")
    print(f"Test  malicious: total={te_total}, unique={te_unique}, ratio={te_ratio:.4f}")

    # Per-template (train) diversity
    per_template_train = compute_per_template_stats(train_cmds, train_templates)

    # Per-template (test) diversity
    per_template_test = compute_per_template_stats(test_cmds, test_templates)

    # Placeholder value diversity on training malicious commands
    placeholder_counts = compute_placeholder_value_counts(train_cmds)

    print("\n=== Per-template diversity (train) ===")
    print(f"# templates (train): {per_template_train['templates']}")
    print(
        "Unique commands per template (min / median / max): "
        f"{per_template_train['per_template_min_unique']:.0f} / "
        f"{per_template_train['per_template_median_unique']:.0f} / "
        f"{per_template_train['per_template_max_unique']:.0f}"
    )

    print("\n=== Per-template diversity (test) ===")
    print(f"# templates (test): {per_template_test['templates']}")
    print(
        "Unique commands per template (min / median / max): "
        f"{per_template_test['per_template_min_unique']:.0f} / "
        f"{per_template_test['per_template_median_unique']:.0f} / "
        f"{per_template_test['per_template_max_unique']:.0f}"
    )

    print("\n=== Placeholder value diversity (train malicious) ===")
    print(
        "Shells: {shells}, IPs: {ips}, ports: {ports}, paths: {paths}, variables: {vars}".format(
            **placeholder_counts
        )
    )


if __name__ == "__main__":
    main()


