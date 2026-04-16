# stage1_split_cifar100

Split-CIFAR100 continual-learning benchmark: CIFAR100 partitioned into 20 sequential tasks of 5 classes each.

## Goal

Reproduce Mamba-CL and Inf-SSM baselines, then demonstrate **+8–10% average accuracy** over Mamba-CL and **+4%** over Inf-SSM when the SWR generator + null-space projector are added on top of the Mamba backbone.

## Metrics

- Average Accuracy (AA) after all 20 tasks
- Average Forgetting (AF)
- Backward Transfer (BWT)
- Memory footprint (should stay constant across tasks — the headline claim)

## Outputs

Runs write to `../../results/stage1_split_cifar100/<run_id>/`.
