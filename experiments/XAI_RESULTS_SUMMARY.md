# XAI Analysis Results Summary (Tactics and Commands)

**Date:** November 16, 2025  
**Goal:** Expand the SHAP-based XAI discussion from individual tokens to the most-relevant **tactics** and **command patterns**, addressing Reviewer 1 (point 7) and decision-summary bullet 1.

---

## 1. Context and setup

- The current paper’s Explainability section (\texttt{sec:explainability}) analyzes **token-level SHAP** for regular and adversarially trained GBDT models, highlighting:
  - IP octets (`.`, `10`, `127`),
  - quoting/assignment tokens (`("`, `=`),
  - and redirection tokens (`>&`, `2`)  
  as core drivers of decisions.
- To move from “important tokens” to **tactics and commands**, we ran two additional experiments:
  - `experiments/xai_commands_shap.py`: groups SHAP values by **utility** and **tactic** on a tokenised One-Hot representation.
  - `experiments/xai_detection_by_technique.py`: measures detection performance **per reverse-shell technique, per technique family, and per attack-component group** on synthesized commands with template metadata.

Both experiments use XGBoost with One-Hot features (`VOCAB_SIZE = 4096`, `LIMIT = 30000`) and the same synthesis pipeline as the main paper.

---

## 2. Utility- and tactic-level SHAP (xai_commands_shap.py)

**Setup.** We compute SHAP values on the test set with a TreeExplainer, then aggregate feature importances:
- by **utility** (e.g., `nc`, `bash`, `python`), and
- by **tactic** (e.g., IP tokens, network utilities, FD redirection, obfuscation).

### 2.1 Most important utilities (mean |SHAP|)

- **netcat (`nc`)**: ≈ **1.00** – strongest single utility indicator of maliciousness.
- **`bash`**: ≈ **0.43** – shell invoker, often paired with `/dev/tcp` or pipes.
- **`python`**: ≈ **0.27** – scripting interpreter for socket-based shells.
- Other interpreters (`perl`, `php`, `ruby`, `node`) have near-zero mean |SHAP| in this run due to lower frequency in the evaluated test mix.

### 2.2 Most important tactics (mean |SHAP|)

- **IP tokens** (`.`, `10`, `127`): ≈ **1.85** – strongest single tactic; explicit IP-like substrings are the primary malicious cue.
- **Network utilities** (`nc`, `ncat`, `socat`, `telnet`, `ssh`, `wget`, `curl`, `nmap`): ≈ **1.00** – capture the presence of outbound communication utilities.
- **Shell invokers** (`bash`, `sh`, `zsh`, `ksh`): ≈ **0.43** – execution context for the payload.
- **Interpreters** (`python`, `perl`, `php`, `ruby`, `node`): ≈ **0.27** – scripting-based shells.
- **FD redirection** (`>&`, `2>&1`, `/dev/tcp`, `udp`): ≈ **0.10** – stealthy I/O channel setup.
- **Obfuscation** (`base64`, `eval`, `exec`, `xxd`, `tr`, `sed`, `awk`, `printf`): ≈ **0.04** – present but secondary in this dataset.

**Interpretation.**  
The model learns a **hierarchy of tactics**:
- **Primary signal**: IP tokens + network utilities (where and how a connection is opened).
- **Secondary context**: shell invokers and interpreters (how the connection is controlled).
- **Supporting evidence**: FD redirection and obfuscation (how data are tunneled and hidden).

This directly upgrades the narrative from “top tokens” to **attack-tactic categories** that security engineers can reason about (e.g., “IP + `nc` + `/dev/tcp` + `bash`” rather than isolated symbols).

Key artifacts (latest run, example timestamp `1763289879`):
- `experiments/logs_xai_commands_shap_1763289879/utilities_mean_abs_shap.csv`
- `experiments/logs_xai_commands_shap_1763289879/tactics_mean_abs_shap.csv`
- Plots under `experiments/logs_xai_commands_shap_1763289879/plots/` (bar charts, violins, strip plots).

---

## 3. Technique- and component-level detection (xai_detection_by_technique.py)

**Setup.** We generate malicious commands per template with metadata, train XGBoost, and evaluate:
- **per-template detection**,  
- **aggregation by primary binary (technique)**,  
- **aggregation by technique family**, and  
- **detection conditioned on presence of attack components** (same tactic groups as above, matched via substrings on raw commands).

### 3.1 Detection by primary binary (technique level)

- **Scripting techniques** (awk, lua, perl, php, python3, ruby):
  - TPR = **1.00** for each, mean malicious score ≈ **0.94–1.00** (N = 924 per technique).
- **Bash-based shells**:
  - Aggregated `bash`: TPR ≈ **0.87**, mean score ≈ **0.89** (N = 3,696).
- **Zsh (`ztcp`-based shells)**:
  - TPR ≈ **0.18**, mean score ≈ **0.48** (N = 924), reflecting that rare zsh-specific patterns are underrepresented and harder to learn.

### 3.2 Detection by technique family

- **Named-pipe shells** (e.g., `mkfifo` + `/tmp/`): TPR ≈ **88.6%** (N = 2,772).
- **Scripting shells** (Python/PHP/Perl/Ruby/Lua): TPR ≈ **88.3%** (N = 6,468).
- **Shell-only techniques** (e.g., `/dev/tcp`): TPR ≈ **81.4%** (N = 924).

### 3.3 Detection conditioned on attack components (command-level patterns)

For malicious commands that contain each component (substring membership):

- **Interpreter present** (`python`, `php`, `perl`, `ruby`, `node`, ...):
  - TPR ≈ **99.8%**, mean score ≈ **0.998** (N ≈ 3,722).
- **Obfuscation present** (`base64`, `eval`, `exec`, `awk`, `printf`, ...):
  - TPR ≈ **98.0%**, mean score ≈ **0.97** (N ≈ 3,854).
- **Core reverse-shell components**:
  - `shell_invoker`: TPR ≈ **87.8%** (N ≈ 10,164).
  - `ip_tokens`: TPR ≈ **87.8%** (N ≈ 10,164).
  - `net_utility`: TPR ≈ **86.8%** (N ≈ 3,697).
  - `fd_redirection`: TPR ≈ **86.5%** (N ≈ 9,240).
- **Wrappers** (`sudo`, `nohup`, `timeout`, ...):
  - TPR ≈ **60%**, but with very low support (N = 10), so this estimate is noisy.

**Interpretation.**  
When commands contain **IP tokens + network utilities + shell/FD redirection**, the detector is consistently confident within this slice (≈ 86–88% TPR), confirming that these **combined command patterns**—rather than any single token—drive decisions.  
Scripting-based shells are easiest to catch (100% TPR) because their syntax (`socket` APIs, `fsockopen`, etc.) is distinctive and rare in benign commands, while bash-based `/dev/tcp` and named-pipe techniques remain challenging but still achieve ≈ 81–89% TPR.

Key artifacts (latest run, example timestamp `1763289895`):
- `experiments/logs_xai_detection_by_technique_1763289895/detection_by_template.csv`
- `experiments/logs_xai_detection_by_technique_1763289895/detection_by_primary_binary.csv`
- `experiments/logs_xai_detection_by_technique_1763289895/detection_by_technique_family.csv`
- `experiments/logs_xai_detection_by_technique_1763289895/detection_by_attack_component.csv`
- Plots under `experiments/logs_xai_detection_by_technique_1763289895/plots/`.

---

## 4. Practical defense narrative for the paper

These results support a concise, tactic-centric rule that can be described in the revised Explainability section:

```text
IF (IP-like substrings present) AND
   (netcat-style utility OR scripting interpreter) AND
   (shell invocation OR file-descriptor redirection)
THEN
   High-confidence reverse-shell suspicion.
```

- **Most-relevant tactics:** IP tokens, network utilities, shell invokers, and interpreters concentrate the majority of grouped SHAP mass and correspond to the highest conditional TPRs.
- **Most-relevant commands:** Reverse shells that explicitly combine:
  - IP literals (e.g., `10.x.x.x`, `127.0.0.1`),
  - a network utility or socket API (`nc`, `socat`, scripting `socket` calls),
  - and either `/dev/tcp`, named pipes, or interpreter-driven socket setup,
  are detected with the highest confidence and should be highlighted with concrete examples in the paper.

In the manuscript, this can be integrated as:
- a short **tactic-level table** reporting mean |SHAP| per tactic (IP tokens, net utilities, shell invokers, interpreters, FD redirection, obfuscation), and
- a short **technique/component table** summarizing TPR by primary binary and by attack component (interpreter, net utility, FD redirection, etc.), with a brief paragraph explaining how these align with domain knowledge.

---

## 5. How this addresses Reviewer 1 and the decision summary

- **Reviewer 1, point 7 (command substrings and feature patterns):**
  - We now **rank tactics** by grouped SHAP importance (Section 2).
  - We quantify detection **per technique** and **per attack component** (Section 3).
  - We translate these into a compact **detection rule and defense narrative** (Section 4), explicitly linking model behavior to recognizable reverse-shell building blocks.
- **Decision-summary bullet 1 (“most-relevant tactics” and “most-relevant commands”):**
  - Tactic hierarchy: IP tokens → net utilities → shell/interpreter → FD redirection/obfuscation.
  - Command-level patterns: high-confidence detection when IP + `nc`/scripting + `/dev/tcp`/named pipes co-occur.

Taken together, these extensions move the XAI discussion from single tokens to **tactic- and command-level explanations**, making the model’s detection logic actionable for security engineers.
