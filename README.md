# Quasar: Reverse Shell Detection with Machine Learning

<img src="img/quasaroutflow.png" height="400">

## ToDo

1. Ablation test for scaling laws (model sizes 300k, 1M, 3M) and learning rates.
   > NOTE: Coded in `ablation_scaling_laws_and_lr.py` --> need to:
      (1) select appropriate MLP sizes (and vocab?)
      (2) run with PROD config;

2. Comparse baseline performances of different model architectures: `ablation_models.py`
   > NOTE: After chosing appropriate LR for these model sizes, run with PROD config.

3. Explainability tests:
   1. Integrated gradients TODO
   2. Attention weights from transformer TODO

4. Adversarial robustness tests:
   1. Evasion: (i) black box (ii) white box
      > NOTE: Black box with NL2Bash coded: `adversarial_blackbox.py` --> need to run with PROD config.
   2. Poisoning TODO
