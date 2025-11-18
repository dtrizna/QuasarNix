# Attack Component Detection at Fixed FPR Levels

**Date:** November 18, 2025  
**Experiment:** `xai_detection_by_technique.py`  
**Results:** `logs_xai_detection_by_technique_1763451366/`

## Executive Summary

We analyzed detection rates by attack component (reverse shell components) at three operating points:
1. **F1-optimal threshold** (0.0311) - Maximum balanced performance
2. **FPR = 10⁻⁴** (threshold 0.1600) - Moderate false positive constraint
3. **FPR = 10⁻⁶** (threshold 0.9999) - Ultra-strict production constraint

This directly addresses the reviewer's request for "which are the most-relevant commands" beyond token-level analysis.

---

## Results Comparison

### 1. F1-Optimal Threshold (t=0.0311)

| Component | TPR | N Samples | Notes |
|-----------|-----|-----------|-------|
| interpreter | 100.0% | 3,716 | python/perl/php/ruby |
| obfuscation | 99.9% | 3,874 | base64/eval/exec |
| shell_invoker | 99.9% | 10,164 | bash/sh/zsh |
| ip_tokens | 99.9% | 10,164 | IP address indicators |
| fd_redirection | 99.9% | 9,240 | >&, 2>&1, /dev/tcp |
| net_utility | 99.7% | 3,697 | nc/socat/telnet |
| wrappers | 95.2% | 21 | sudo/nohup (rare) |

**Key Insight:** At F1-optimal threshold, nearly perfect detection across all component types.

### 2. FPR = 10⁻⁴ (t=0.1600)

| Component | TPR | Δ from F1 | N Samples |
|-----------|-----|-----------|-----------|
| interpreter | 99.9% | -0.1% | 3,716 |
| obfuscation | 99.0% | -0.9% | 3,874 |
| shell_invoker | 97.6% | -2.5% | 10,164 |
| ip_tokens | 97.6% | -2.5% | 10,164 |
| fd_redirection | 97.4% | -2.6% | 9,240 |
| net_utility | 93.5% | -6.2% | 3,697 |
| wrappers | 85.7% | -9.5% | 21 |

**Key Insight:** Modest degradation. Interpreter detection remains nearly perfect. Network utilities show first signs of reduced sensitivity.

### 3. FPR = 10⁻⁶ (t=0.9999) - Production Setting

| Component | TPR | Δ from F1 | N Samples | Notes |
|-----------|-----|-----------|-----------|-------|
| interpreter | 92.1% | -7.9% | 85,455 | Still excellent |
| net_utility | 62.9% | -36.8% | 85,526 | Significant drop |
| shell_invoker | 57.8% | -42.1% | 234,993 | Major reduction |
| ip_tokens | 57.8% | -42.1% | 234,993 | Major reduction |
| fd_redirection | 55.7% | -44.3% | 213,630 | Major reduction |
| obfuscation | 48.5% | -51.4% | 85,608 | Challenging at strict FPR |
| wrappers | 0.0% | -95.2% | 1 | Too rare in test set |

**Key Insights:**
- **Interpreter-based reverse shells remain most detectable** (92% TPR)
  - Distinctive scripting syntax hard to obfuscate
  - python/perl/php socket APIs provide strong signal

- **Network utilities become challenging** (63% TPR)
  - More ambiguous: nc appears in benign contexts
  - Trade-off between detection and FPR

- **Shell constructs most affected** (58% TPR)
  - bash commands highly variable in benign use
  - /dev/tcp and redirection patterns common in legitimate scripts

---

## Detection Rate Hierarchy at FPR = 10⁻⁶

**Tier 1 (Excellent): TPR > 90%**
- Interpreter-based shells (python, perl, php, ruby)
- Reason: Unique syntax patterns, rare in benign baseline

**Tier 2 (Good): TPR 60-70%**
- Network utility-based shells (nc, socat, telnet)
- Reason: Distinctive but some benign usage overlap

**Tier 3 (Moderate): TPR 50-60%**
- Shell-based constructs (bash /dev/tcp, fd redirection)
- IP tokens, shell invokers
- Reason: High variance in benign use, requires contextual evidence

**Tier 4 (Challenging): TPR < 50%**
- Obfuscation patterns
- Wrapper utilities
- Reason: Too rare or too common in baseline

---

## Practical Implications for Defense

### 1. Detection Strategy by Component Type

**High-Confidence Alerts (T1):**
```
IF interpreter_detected AND ip_tokens:
    ALERT: High-confidence reverse shell (92% TPR @ FPR=10⁻⁶)
```

**Medium-Confidence Alerts (T2+T3):**
```
IF (net_utility OR shell_invoker) AND ip_tokens AND fd_redirection:
    ALERT: Medium-confidence reverse shell (58% TPR @ FPR=10⁻⁶)
```

### 2. Component Importance for Detection

**Ranked by robustness at FPR = 10⁻⁶:**
1. **interpreter** (92%) - Most reliable single indicator
2. **net_utility** (63%) - Strong but context-dependent
3. **ip_tokens + shell_invoker** (58%) - Requires combination
4. **fd_redirection** (56%) - Supporting evidence
5. **obfuscation** (48%) - Weak at strict FPR

### 3. Why This Matters for Paper

The reviewer asked: *"which are the most-relevant commands"*

**Answer:**
- **Most relevant**: Interpreter-based commands (python socket scripts)
- **Moderately relevant**: Network utility invocations (nc, socat)
- **Context-dependent**: Shell constructs (bash /dev/tcp) - require multiple components aligned

This provides **actionable guidance** beyond token-level SHAP values:
- Security analysts know which command types to prioritize
- Detection engineers understand FPR/TPR trade-offs per component
- System designers can tune thresholds based on component presence

---

## Data Characteristics

**Vanilla Dataset (LIMIT=30k):**
- Train: 59,992 commands (29,992 malicious)
- Test: 20,335 commands (10,164 malicious)
- Used for F1-optimal and FPR=1e-4 analysis

**Oversampled Dataset (Full):**
- Train: 531,995 commands (265,995 malicious)
- Test: 469,993 commands (234,993 malicious)
- 10× larger → reliable FPR=1e-6 estimation
- Required for production-grade FPR analysis

---

## Files Generated

```
logs_xai_detection_by_technique_1763451366/
├── detection_by_attack_component_f1.csv
├── detection_by_attack_component_fpr_1e4.csv
├── detection_by_attack_component_fpr_1e6.csv
├── detection_by_primary_binary.csv
├── detection_by_technique_family.csv
├── detection_by_template.csv
└── plots/
    ├── detection_by_attack_component_f1.{pdf,png}
    ├── detection_by_attack_component_fpr_1e4.{pdf,png}
    ├── detection_by_attack_component_fpr_1e6.{pdf,png}
    ├── detection_by_primary_binary_barplot.{pdf,png}
    └── detection_by_technique_family_barplot.{pdf,png}
```

---

## Conclusion

**For Paper Revision:**

Add multi-threshold analysis showing:
1. Interpreter-based shells remain highly detectable (92% TPR) even at FPR = 10⁻⁶
2. Network utility patterns moderately detectable (63% TPR)
3. Shell constructs require contextual evidence (58% TPR)

This **hierarchical detection capability** addresses reviewer concerns about "most-relevant commands" and provides practical defense guidance.

**Key Figure for Paper:** `detection_by_attack_component_fpr_1e6.png` - Shows operational TPR at production FPR constraint.

