# FairShift

This repository contains the code folders used for the paper submission. FairShift is the method proposed in the paper, and these folders contain experiments that apply FairShift to EDGE and FairWire variants.

The folder names correspond to the variants described in the paper:

| Folder | Paper variant |
| --- | --- |
| `EDGE_fairness` | EDGE-cond FairShift-F |
| `EDGE_fairness_loss` | EDGE-cond FairShift-T |
| `FairWire_feature` | FW-fc |
| `FairWire_feature_shift` | FW-fc FairShift-F |
| `FairWire_feature_fairness_loss` | FW-fc FairShift-T |
| `FairWire_fairness_loss` | FW-bb FairShift-T and FW backbone |
| `FairWire_shift` | FW-bb FairShift-F |

The EDGE folders, FW-bb folders, and FW-fc folders form implementation groups. Checkpoints such as `.pt` files produced by wandb can be moved within the corresponding group when running or reproducing experiments.

Generated artifacts such as plots, CSV files, logs, checkpoints, cached files, and experiment-output directories are intentionally excluded from this repository.
