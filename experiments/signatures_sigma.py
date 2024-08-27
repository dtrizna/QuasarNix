import os
from sigma.collection import SigmaCollection


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(ROOT)
from src.data_utils import load_data

if __name__ == "__main__":
    *_, X_train_malicious_cmd, _, X_test_malicious_cmd, _ = load_data(ROOT, seed=33, limit=100)
    command_lines = X_train_malicious_cmd
    print(len(command_lines))

    # Folder containing Sigma rule YAML files
    sigma_rule_folder = os.path.join(ROOT, "data", "signatures")
    sigma_rule_yamls = [os.path.join(sigma_rule_folder, f) for f in os.listdir(sigma_rule_folder) if f.endswith(".yaml")]
    sigma_rule_collection = SigmaCollection.load_ruleset(sigma_rule_yamls)
    print(len(sigma_rule_collection))

    # Loop through each command line
    for command_line in command_lines:
        # Check if command line matches any Sigma rule
        for rule in sigma_rule_collection:
            if rule.matches(command_line):
                print(f"Matched: {rule.title}")
                break
        else:
            print(f"No match: {command_line}")
