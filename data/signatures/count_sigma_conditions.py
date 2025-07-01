#!/usr/bin/env python3
"""
Count, for all Sigma rules in this folder, both

1. The number of selector blocks – the keys directly under `detection`
   except the special key ``condition``;
2. The number of *leaf* conditions inside those selectors
   (every individual value such as a literal string, number, etc.).

Example
-------
For

detection:
    selection_a:
        Image|endswith: '/bash'
        ParentImage|endswith: '/tmp/'
    selection_b:
        - CommandLine|contains:
              - '| sh '
              - '| bash '
    condition: selection_a and selection_b

Selector blocks = 2  (selection_a, selection_b)  
Leaf conditions  = 5  ('/bash', '/tmp/', '| sh ', '| bash ', list item wrapper)

The script prints an aggregated total for the whole folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except ModuleNotFoundError as exc:  # pragma: no cover
    sys.exit(
        "[!] PyYAML is required:  pip install pyyaml\n"
        f"    ({exc})"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _count_leaf_values(obj) -> int:
    """Recursively count primitive leaf values inside *obj*."""
    if isinstance(obj, dict):
        return sum(_count_leaf_values(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_leaf_values(i) for i in obj)
    # Primitive (str, int, bool, etc.)
    return 1


def selectors_and_leaves_in_rule(rule: dict) -> tuple[int, int]:
    """
    Return  (selector_count, leaf_condition_count)  for a single Sigma rule.
    """
    detection = rule.get("detection", {})
    selector_blocks = 0
    leaf_conditions = 0

    for key, value in detection.items():
        if key == "condition":
            continue  # not a selector
        selector_blocks += 1
        leaf_conditions += _count_leaf_values(value)

    return selector_blocks, leaf_conditions


def counts_in_file(path: Path) -> tuple[int, int]:
    """Aggregate selector / leaf counts for all documents in *path*."""
    selectors = leaves = 0
    with path.open(encoding="utf-8") as fh:
        for doc in yaml.safe_load_all(fh):
            if isinstance(doc, dict):
                s, l = selectors_and_leaves_in_rule(doc)
                selectors += s
                leaves += l
    return selectors, leaves


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    folder = Path(__file__).parent
    total_selectors = total_leaves = 0

    for file in folder.iterdir():
        if file.suffix.lower() in {".yml", ".yaml"}:
            s, l = counts_in_file(file)
            total_selectors += s
            total_leaves += l

    print(
        f"Folder '{folder.name}':\n"
        f"  selector blocks : {total_selectors}\n"
        f"  leaf conditions : {total_leaves}"
    )


if __name__ == "__main__":
    main()
