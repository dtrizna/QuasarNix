import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from sigma.collection import SigmaCollection
from sigma.conditions import (
    ConditionAND,
    ConditionOR,
    ConditionNOT,
    ConditionFieldEqualsValueExpression,
    ConditionValueExpression,
)
from sigma.types import SigmaString, SigmaType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT)
from src.data_utils import load_data


def _wildcard_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a Sigma-style wildcard pattern (using * and ?) into a Python regex.
    This is deliberately conservative and only used when the SigmaString contains wildcards.
    """
    parts = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts))


def _sigma_value_matches(value: SigmaType, candidate: str) -> bool:
    """
    Very lightweight matcher for Sigma value types against a single string.

    - If the value is a SigmaString without wildcards, we do a substring check.
    - If it contains wildcards, we translate them to regex and search.
    - For other SigmaType subclasses, we fall back to a simple substring match on their plain value.
    """
    if isinstance(value, SigmaString):
        plain = value.to_plain()
        if value.contains_special():
            regex = _wildcard_to_regex(plain)
            return bool(regex.search(candidate))
        # No wildcards → treat as keyword / substring
        return plain in candidate

    # Fallback for other SigmaType implementations
    try:
        plain_val: Any = value.to_plain()
    except Exception:
        plain_val = str(value)
    return str(plain_val) in candidate


def _eval_condition(node: Any, candidate: str) -> bool:
    """
    Evaluate a parsed Sigma condition tree against a single command-line string.

    We intentionally ignore field names (Image, CommandLine, ParentImage, ...) and
    just match all leaf patterns against the full command line. This is sufficient
    for our synthetic command-line corpus.
    """
    if isinstance(node, ConditionAND):
        return all(_eval_condition(arg, candidate) for arg in node.args)
    if isinstance(node, ConditionOR):
        return any(_eval_condition(arg, candidate) for arg in node.args)
    if isinstance(node, ConditionNOT):
        # NOT should always have exactly one child
        return not _eval_condition(node.args[0], candidate)
    if isinstance(node, ConditionFieldEqualsValueExpression):
        return _sigma_value_matches(node.value, candidate)
    if isinstance(node, ConditionValueExpression):
        return _sigma_value_matches(node.value, candidate)

    # Unexpected node type – be conservative and return False
    return False


def rule_matches_command(rule, command_line: str) -> bool:
    """
    Check whether a Sigma rule matches a given command-line string.

    pySigma doesn't provide a .matches() API; instead we:
      1) Parse the rule's condition into a condition tree, and
      2) Evaluate that tree against the command line using simple wildcard / substring logic.
    """
    detections = getattr(rule, "detection", None)
    if detections is None or not getattr(detections, "parsed_condition", None):
        return False

    # Sigma allows multiple condition strings; treat the rule as matched if any is satisfied.
    for cond in detections.parsed_condition:
        tree = cond.parsed
        if _eval_condition(tree, command_line):
            return True
    return False


def get_tpr_at_fpr(predicted_probs: np.ndarray, true_labels: np.ndarray, fpr_needed: float) -> float:
    """
    Compute TPR at a given FPR using the ROC curve, mirroring the logic from
    experiments/results_roc_ablation_models.ipynb.
    """
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    if np.all(np.isnan(fpr)):
        return float("nan")

    mask = fpr <= fpr_needed
    if not np.any(mask):
        return float("nan")
    return float(tpr[mask][-1])


def evaluate_sigma_on_commands(
    commands: List[str],
    labels: np.ndarray,
    root: str,
    limit: Optional[int] = None,
    random_state: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the Sigma rule set against a list of command-line strings.

    This is written to be importable from experiments such as mc_repeated_training.py.

    Args:
        commands: List of command-line strings (typically the test split).
        labels: Binary ground-truth labels (0 = benign, 1 = malicious).
        root: Project root folder containing ``data/signatures``.
        limit: If set, randomly subsample at most ``limit`` commands.
        random_state: Seed for the subsampling RNG.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Dictionary with AUC, F1, accuracy, and TPR at several FPR levels.
    """
    start = time.time()

    y = np.asarray(labels, dtype=np.int8)
    n_total = len(y)

    if n_total != len(commands):
        raise ValueError(
            f"Length mismatch between commands ({len(commands)}) and labels ({n_total})."
        )

    # Optional random subsampling for efficiency / uncertainty estimates
    if limit is not None and n_total > limit:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=limit, replace=False)
        y_eval = y[idx]
        commands_eval = [commands[i] for i in idx]
    else:
        y_eval = y
        commands_eval = list(commands)

    # Folder containing Sigma rule YAML files
    sigma_rule_folder = os.path.join(root, "data", "signatures")
    sigma_rule_yamls = [
        os.path.join(sigma_rule_folder, f)
        for f in os.listdir(sigma_rule_folder)
        if f.endswith((".yaml", ".yml"))
    ]
    sigma_rule_collection = SigmaCollection.load_ruleset(sigma_rule_yamls)

    y_pred_binary = np.zeros_like(y_eval, dtype=np.int8)

    iterator = enumerate(commands_eval)
    if show_progress:
        iterator = enumerate(
            tqdm(
                commands_eval,
                desc="Evaluating commands on Sigma rules",
                total=len(commands_eval),
            )
        )

    for i, command_line in iterator:
        matched = False
        for rule in sigma_rule_collection:
            if rule_matches_command(rule, command_line):
                matched = True
                break
        y_pred_binary[i] = 1 if matched else 0

    # Treat the binary outputs as probabilities in {0, 1}
    y_pred_probs = y_pred_binary.astype(float)

    # Compute metrics (similar to results_roc_ablation_models)
    # Guard against degenerate single-class subsets for safety.
    if len(np.unique(y_eval)) > 1:
        auc = roc_auc_score(y_eval, y_pred_probs)
    else:
        auc = float("nan")

    f1 = f1_score(y_eval, y_pred_binary)
    precision = precision_score(y_eval, y_pred_binary, zero_division=0)
    recall = recall_score(y_eval, y_pred_binary, zero_division=0)
    acc = accuracy_score(y_eval, y_pred_binary)

    tpr_at_1e4 = get_tpr_at_fpr(y_pred_probs, y_eval, fpr_needed=1e-4)
    tpr_at_1e5 = get_tpr_at_fpr(y_pred_probs, y_eval, fpr_needed=1e-5)
    tpr_at_1e6 = get_tpr_at_fpr(y_pred_probs, y_eval, fpr_needed=1e-6)

    elapsed = time.time() - start

    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "acc": float(acc),
        "tpr_at_1e4": float(tpr_at_1e4),
        "tpr_at_1e5": float(tpr_at_1e5),
        "tpr_at_1e6": float(tpr_at_1e6),
        "elapsed_sec": float(elapsed),
        "n_eval": int(len(y_eval)),
    }


if __name__ == "__main__":
    # ===========================================
    # Load data (use full test split for metrics)
    # ===========================================
    X_train_cmds, y_train, X_test_cmds, y_test, *_ = load_data(ROOT, seed=33, limit=None)
    y_test = np.asarray(y_test, dtype=np.int8)
    print(f"[+] Test set size: {len(X_test_cmds)} commands")

    metrics = evaluate_sigma_on_commands(
        commands=X_test_cmds,
        labels=y_test,
        root=ROOT,
        limit=50000,
        random_state=33,
        show_progress=True,
    )

    print("\n=== Sigma rule-set metrics on test set ===")
    print(f"AUC:        {metrics['auc']*100:.6f}%")
    print(f"Accuracy:   {metrics['acc']*100:.4f}%")
    print(f"F1-score:   {metrics['f1']*100:.4f}%")
    print(f"Precision:  {metrics['precision']*100:.4f}%")
    print(f"Recall:     {metrics['recall']*100:.4f}%")
    print("\nTPR at fixed FPR thresholds:")
    print(f"  TPR @ FPR=1e-4: {metrics['tpr_at_1e4']*100:.4f}%")
    print(f"  TPR @ FPR=1e-5: {metrics['tpr_at_1e5']*100:.4f}%")
    print(f"  TPR @ FPR=1e-6: {metrics['tpr_at_1e6']*100:.4f}%")
