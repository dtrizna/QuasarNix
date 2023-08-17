# Quasar: Reverse Shell Detection with Machine Learning

<img src="img/quasaroutflow.png" height="400">

NOTE: `tensorboard --logdir FOLDER --port 8080`

## ToDo

1. Comparse baseline performances of different model architectures: `ablation_models.py`

2. Explainability tests:
   1. Integrated gradients TODO
   2. Attention weights from transformer TODO

3. Adversarial robustness tests:
   1. Evasion: (i) black box (ii) white box
      > NOTE: Black box with NL2Bash coded: `adversarial_blackbox.py`
         --> need to: run with PROD config
   2. Poisoning TODO
