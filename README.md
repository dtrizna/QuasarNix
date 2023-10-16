# Quasar: Reverse Shell Detection with Machine Learning

<img src="img/quasaroutflow.png" height="400">

## ToDo

1. ~~One-class runs: `results_one_class_models.ipynb`~~

2. Adversarial robustness tests:
   1. Poisoning:
      1. with malicious tokens
      2. with tokens from sparse regions
      3. hybrid
   2. Evasion (black box):
      > NOTE: Black box with NL2Bash coded: `adversarial_evasion.py`
      - run `adversarial_blackbox.py` with PROD config: (a) ~~one-hot~~ (b) sequential: CNN | CLS_Transformer
      - consider alternative manipulations, e.g. simple evasions (e.g. `bash -i` -> `bash -li` or `bash -c` -> `bash -ci`)
   3. Rerun experiments for different architectures.
   4. Signature baseline -- stay robust? Worth showing.
   5. Defend:
      1. hybrid approach: ML + signatures in process?
      2. adversarial training

3. Command splitting, e.g.: `rm /tmp/f; mkfifo /tmp/f; cat /tmp/f | /bin/sh -i 2>&1 | nc` to passing through ML model as separate commands?
   > Battista: "Used in Certification schemes." <-- ??

4. Explainability tests (???):
   1. Integrated gradients TODO
   2. Attention weights from transformer TODO

> NOTE: Starting tensorboard session for a logdir foler: `tensorboard --logdir FOLDER --port 8080`
