import os
import sys
import json
import yaml
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.signatures import SignaturesEvolved
from src.evasion import attack_evasive_tricks
from src.data_utils import load_data

ROOT = os.path.dirname(os.path.abspath('__file__'))
sigma_yml_path = os.path.join(ROOT, "data", "rvrs_sigma.yml")
LIMIT = None
SEED = 42

signature_params = {
    "true_positive_threshold": 6,
    "sliding_window_stride": 1,
    "sliding_window_padding": 2,
}

LOG_DIR = os.path.join(ROOT, "logs_signatures_evolved")
os.makedirs(LOG_DIR, exist_ok=True)


def update_scores_with_metrics(scores, run_type):
    tp = scores[run_type]["malicious"]
    fp = scores[run_type]["baseline"]
    tn = len(X_train_baseline_cmd) - fp
    fn = len(X_train_malicious_cmd) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # add to results
    scores[run_type]["f1"] = f1
    scores[run_type]["accuracy"] = accuracy
    scores[run_type]["precision"] = precision
    scores[run_type]["recall"] = recall
    return scores


if __name__ == '__main__':
    with open(sigma_yml_path, "r") as f:
        sigma = yaml.load(f, Loader=yaml.FullLoader)
    patterns = sigma['detection']['keywords']
    s = SignaturesEvolved(signatures=patterns, **signature_params)

    X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd = load_data(SEED, LIMIT)
    
    X_train_malicious_cmd_with_attack_cmd = []
    for cmd in tqdm(X_train_malicious_cmd):
        cmd_a = attack_evasive_tricks(cmd, [], attack_parameter=0.7)
        X_train_malicious_cmd_with_attack_cmd.append(cmd_a)

    results = {
        "sliding_similarity": {
            "baseline": None,
            "malicious": None,
            "malicious_with_evasions": None,
        },
        "default": {
            "baseline": None,
            "malicious": None,
            "malicious_with_evasions": None,
        },
    }

    datasets = {
        "baseline": X_train_baseline_cmd,
        "malicious": X_train_malicious_cmd,
        "malicious_with_evasions": X_train_malicious_cmd_with_attack_cmd,
    }

    analysis_times_sliding = []
    analysis_times_default = []
    for data_type, commands in datasets.items():
        print(f"\n\n============= {data_type.replace('_', ' ').capitalize()} =============\n")
        
        # sliding windows similairy stats
        print("[*] Sliding window similarity analysis...")
        total_matches = 0
        for cmd in tqdm(commands):
            now = time.time()
            match, distance_value, signature, fragment = s.analyze_command(cmd)
            if match:
                total_matches += 1
            analysis_times_sliding.append(time.time() - now)
        print("[!] Total commands:", len(commands))
        print("[!] Total matches:", total_matches)
        results["sliding_similarity"][data_type] = total_matches
        
        # default stats
        print("\n[*] Default analysis...")
        total_matches = 0
        for cmd in tqdm(commands):
            now = time.time()
            for pattern in patterns:
                match = pattern in cmd
                if match:
                    total_matches += 1
                    break
            analysis_times_default.append(time.time() - now)
        print("[!] Total commands:", len(commands))
        print("[!] Total matches:", total_matches)
        results["default"][data_type] = total_matches

    print(f"\n\n[!] Average analysis time of sliding similarity: {np.mean(analysis_times_sliding)*1000:.4f}ms")
    print(f"[!] Average analysis time of basic signature match: {np.mean(analysis_times_default)*1000:.4f}ms")

    results = update_scores_with_metrics(results, "sliding_similarity")
    results = update_scores_with_metrics(results, "default")

    with open(os.path.join(LOG_DIR, f"signatures_init_results_lim_{LIMIT}_seed_{SEED}.json"), "w") as f:
        json.dump(results, f, indent=4)
