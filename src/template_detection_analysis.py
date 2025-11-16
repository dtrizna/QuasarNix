from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemplateTechnique:
    primary_binary: str
    technique_family: str
    network_primitives: List[str]


def classify_template_technique(template: str) -> Dict[str, Any]:
    """
    Classify a reverse-shell template into a primary binary / technique family.

    The logic mirrors the heuristics used in the template-level explainability
    experiments, but is implemented here as a side-effect-free utility that can
    be reused and unit tested.
    """
    techniques: Dict[str, Any] = {
        "primary_binary": None,
        "technique_family": None,
        "network_primitives": [],
    }

    template_lower = template.lower()

    # Scripting interpreters
    if "python3" in template_lower:
        techniques["primary_binary"] = "python3"
        techniques["technique_family"] = "scripting"
        if "socket.socket" in template:
            techniques["network_primitives"].append("socket.socket")
        if "pty.spawn" in template:
            techniques["network_primitives"].append("pty.spawn")
        if "os.dup2" in template:
            techniques["network_primitives"].append("os.dup2")

    elif "python" in template_lower and "python3" not in template_lower:
        techniques["primary_binary"] = "python"
        techniques["technique_family"] = "scripting"
        if "socket.socket" in template:
            techniques["network_primitives"].append("socket.socket")
        if "pty.spawn" in template:
            techniques["network_primitives"].append("pty.spawn")
        if "os.dup2" in template:
            techniques["network_primitives"].append("os.dup2")

    elif "perl" in template_lower:
        techniques["primary_binary"] = "perl"
        techniques["technique_family"] = "scripting"
        if "socket" in template_lower:
            techniques["network_primitives"].append("use Socket")
        if "sockaddr_in" in template:
            techniques["network_primitives"].append("sockaddr_in")

    elif "php" in template_lower:
        techniques["primary_binary"] = "php"
        techniques["technique_family"] = "scripting"
        if "fsockopen" in template:
            techniques["network_primitives"].append("fsockopen")
        if "proc_open" in template:
            techniques["network_primitives"].append("proc_open")
        if "popen" in template:
            techniques["network_primitives"].append("popen")
        if "shell_exec" in template:
            techniques["network_primitives"].append("shell_exec")

    elif "ruby" in template_lower:
        techniques["primary_binary"] = "ruby"
        techniques["technique_family"] = "scripting"
        if "tcpsocket" in template_lower:
            techniques["network_primitives"].append("TCPSocket")
        if "spawn" in template:
            techniques["network_primitives"].append("spawn")

    # Netcat-style utilities
    elif template_lower.startswith("nc ") or " nc " in template_lower:
        techniques["primary_binary"] = "netcat"
        techniques["technique_family"] = "netcat"
        if "-e" in template:
            techniques["network_primitives"].append("nc -e")
        if "-c" in template:
            techniques["network_primitives"].append("nc -c")

    elif "awk" in template_lower:
        techniques["primary_binary"] = "awk"
        techniques["technique_family"] = "scripting"
        if "/inet/" in template:
            techniques["network_primitives"].append("/inet/")

    elif "lua" in template_lower:
        techniques["primary_binary"] = "lua"
        techniques["technique_family"] = "scripting"
        if "socket" in template_lower:
            techniques["network_primitives"].append("socket.tcp")

    elif "zsh" in template_lower:
        techniques["primary_binary"] = "zsh"
        techniques["technique_family"] = "scripting"
        if "ztcp" in template:
            techniques["network_primitives"].append("ztcp")

    # Named pipes
    elif "mkfifo" in template:
        techniques["technique_family"] = "named_pipe"
        techniques["primary_binary"] = "bash"
        techniques["network_primitives"].append("mkfifo")

    # Telnet / socat / rcat and similar helpers
    elif "telnet" in template_lower:
        techniques["primary_binary"] = "telnet"
        techniques["technique_family"] = "netcat"
        if "mkfifo" in template:
            techniques["network_primitives"].append("mkfifo")

    elif "rcat" in template_lower:
        techniques["primary_binary"] = "rcat"
        techniques["technique_family"] = "netcat"

    elif "socat" in template_lower:
        techniques["primary_binary"] = "socat"
        techniques["technique_family"] = "netcat"

    # File-descriptor based reverse shells
    elif "/dev/tcp/" in template_lower or "/dev/protocol_type/" in template_lower:
        techniques["technique_family"] = "file_descriptor"
        techniques["primary_binary"] = "bash"
        techniques["network_primitives"].append("/dev/tcp")

    # Fallback to generic bash/shell
    if techniques["primary_binary"] is None:
        techniques["primary_binary"] = "bash"
        if techniques["technique_family"] is None:
            techniques["technique_family"] = "shell"

    return techniques


def _as_numpy(array: Iterable[int] | np.ndarray) -> np.ndarray:
    return np.asarray(list(array) if not isinstance(array, np.ndarray) else array)


def detection_by_template(
    y_test: Iterable[int],
    y_pred: Iterable[int],
    y_pred_proba: Iterable[float],
    template_ids: Iterable[int | None],
    test_templates: List[str],
) -> pd.DataFrame:
    """
    Compute per-template detection metrics.

    Each row corresponds to a template index and includes:
    - template_id
    - template (possibly truncated)
    - primary_binary / technique_family
    - network_primitives (comma-separated)
    - num_samples, tpr, mean_score
    """
    y_test_arr = _as_numpy(y_test).astype(np.int8)
    y_pred_arr = _as_numpy(y_pred).astype(np.int8)
    y_pred_proba_arr = _as_numpy(y_pred_proba).astype(np.float32)
    template_ids_list = list(template_ids)

    if not (len(y_test_arr) == len(y_pred_arr) == len(y_pred_proba_arr) == len(template_ids_list)):
        raise ValueError("All input arrays must have the same length.")

    results: List[Dict[str, Any]] = []

    for template_idx, template in enumerate(test_templates):
        mask = np.array([tid == template_idx for tid in template_ids_list], dtype=bool)
        n_samples = int(mask.sum())
        if n_samples == 0:
            continue

        # True positive rate for malicious label (assumes y_test == 1 for malicious)
        n_correct = int(((y_pred_arr[mask] == 1) & (y_test_arr[mask] == 1)).sum())
        tpr = n_correct / n_samples if n_samples > 0 else 0.0
        mean_score = float(y_pred_proba_arr[mask].mean())

        tech = classify_template_technique(template)

        results.append(
            {
                "template_id": template_idx,
                "template": template[:100] + "..." if len(template) > 100 else template,
                "primary_binary": tech["primary_binary"],
                "technique_family": tech["technique_family"],
                "network_primitives": ", ".join(tech["network_primitives"][:3]),
                "num_samples": n_samples,
                "tpr": round(tpr, 4),
                "mean_score": round(mean_score, 4),
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["primary_binary", "tpr"], ascending=[True, False]).reset_index(drop=True)
    return df


def aggregate_detection_by_primary_binary(detection_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-template metrics into primary-binary level statistics.

    Returns a DataFrame indexed by primary_binary with:
    - mean TPR across templates
    - mean prediction score across templates
    - total number of samples
    """
    if detection_df.empty:
        return pd.DataFrame(columns=["tpr", "mean_score", "num_samples"])

    grouped = (
        detection_df.groupby("primary_binary")
        .agg(
            tpr=("tpr", "mean"),
            mean_score=("mean_score", "mean"),
            num_samples=("num_samples", "sum"),
        )
        .round(4)
        .sort_values("tpr", ascending=False)
    )
    return grouped


def aggregate_detection_by_technique_family(detection_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-template metrics into higher-level technique families.

    Useful for statements such as:
    - netcat-style shells (nc/socat/telnet)
    - file-descriptor shells (/dev/tcp)
    - scripting shells (python/php/perl/ruby/awk/lua/zsh)
    """
    if detection_df.empty:
        return pd.DataFrame(columns=["tpr", "mean_score", "num_samples"])

    grouped = (
        detection_df.groupby("technique_family")
        .agg(
            tpr=("tpr", "mean"),
            mean_score=("mean_score", "mean"),
            num_samples=("num_samples", "sum"),
        )
        .round(4)
        .sort_values("tpr", ascending=False)
    )
    return grouped



