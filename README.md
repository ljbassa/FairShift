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

## SP and equal opportunity (EO)

The EDGE variants and `FairWire_feature_fairness_loss` support choosing the fairness target with
`--fair_score_metric sp|eo`; SP remains the default.
The existing folder READMEs include commands, required input assets, and output paths:

| Folder | EO support | Instructions |
| --- | --- | --- |
| `EDGE_fairness` | Fixed sampling-time EO guidance and LP evaluation | [EO grid and baseline](EDGE_fairness/README.md#equal-opportunity-eo-guidance) |
| `EDGE_fairness_loss` | Learned Stage-2 EO controller and LP evaluation | [Controller and grid guide](EDGE_fairness_loss/README.md) |
| `FairWire_feature` | LP evaluation of EO | [Evaluation guide](FairWire_feature/README.md) |
| `FairWire_feature_shift` | LP evaluation of EO; sampling guidance targets SP | [Evaluation and Pareto guide](FairWire_feature_shift/README.md#eo-pareto-selection-and-guidance) |
| `FairWire_feature_fairness_loss` | Learned Stage-2 EO controller, fixed EO guidance and LP evaluation | [Controller, grid and sampling guide](FairWire_feature_fairness_loss/README.md#sp--eo-controller-selection) |
| `FairWire_fairness_loss` | LP evaluation of EO; controller targets SP | [Evaluation guide](FairWire_fairness_loss/README.md) |
| `FairWire_shift` | LP evaluation of EO; sampling guidance targets SP | [Evaluation guide](FairWire_shift/README.md) |

Here EO is a soft-score gap on positive edges: the absolute difference in mean link-prediction scores between
same-group and different-group pairs. It is not a thresholded true-positive-rate gap or an equalized-odds metric.
The EDGE and feature-controller training/sampling surrogates use detached soft-positive weights from the unguided denoiser; downstream
evaluation uses actual held-out positive edges. These quantities have different roles and need not move together.
Use `lp/eo_abs_gap_mean` for the per-graph-averaged downstream EO gap, together with `lp/auc_mean` for utility.
Missing positive comparison groups produce an undefined EO value, not evidence of a zero gap.

The EDGE tools and `FairWire_feature_fairness_loss` separate SP and EO outputs into `sp/` and `eo/`
subdirectories. Other FairWire folders retain their own CLI and output conventions; check their individual
guides before using `--fair_score_metric eo`. The feature shift sampler continues to apply SP guidance,
while its evaluator reports both SP and EO.

## Code synchronization and local execution

This checkout retains the seven paper variants above. Source updates include reusable scripts, configuration
examples and regression tests, while preserving the aggregate repository's variant-specific options, such as
FairWire diffusion-step selection and normalized stateful SP guidance. The existing folder guides are extended
in place. Standalone source updates do not replace a more capable aggregate implementation.

Dataset files, trained checkpoints and experiment results must be prepared separately. Code synchronization excludes
generated artifacts even when a standalone source repository previously tracked them. JSON/YAML configuration
examples and the small ORCA input fixtures remain source inputs. No reported experiment outputs are bundled.

Activate the environment described by the relevant folder's requirements before running commands. Limit CPU
affinity with `taskset -c <allowed_cpu_list>` and numerical-library threads with `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`. EO launchers default to one numerical-library thread and accept
environment overrides for the Python executable and prepared Stage-1 assets. Use `CUDA_VISIBLE_DEVICES` to
choose the physical GPU when the evaluator uses logical `cuda:0`. A code synchronization does not run training.
