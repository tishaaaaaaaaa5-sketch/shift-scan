# Matched mixed-precision experiment runbook

## Conditions

All trained student conditions use the same pretrained Mamba-2 130M initialization, FineWeb-Edu streaming split, compute dtype, token budget, sequence length, micro-batch, accumulation, learning rate, state-distillation objective, and evaluation harness.

| ID | Projections | Purpose |
|---|---|---|
| A | Full-precision pretrained reference, original exponential decay | Reference |
| B | All blocks ternary | Uniform ternary quality gap |
| C | Two blocks selected by validation sensitivity remain full precision | Sensitivity-guided mixed precision |
| D | Two random blocks remain full precision | Parameter-matched control |

To isolate projection quantization, B/C/D all use original exponential decay (not shift decay).

## Run on a CUDA machine

Install the repository requirements and lm-evaluation-harness, then run:

    pip install -r requirements.txt
    pip install lm-eval
    bash scripts/run_matched_experiment.sh

Defaults: 2M training tokens per run, seeds 0/1/2, two protected blocks, 512-token sequences, micro-batch 1, accumulation 4. Override with environment variables, e.g. TOKENS=10000000 SEEDS="0 1 2 3 4" bash scripts/run_matched_experiment.sh.

The runner first measures each block's protection gain on the held-out validation shard. It freezes the top two blocks and a seeded random two-block control, then trains B/C/D for each seed and evaluates all seven lm-evaluation-harness tasks. A is evaluated once because the full-precision reference is identical across seeds.

## Interpretation

- B worse than A quantifies the ternary quality gap.
- C better than B indicates a mixed-precision benefit.
- C better than D tests whether sensitivity-guided allocation beats random allocation at the same protected parameter count.
- Compare paired seed results and report means and standard deviations. A is one fixed pretrained reference, not a three-seed training distribution.
- The sensitivity-selection shard is calibration data, not an independent test set.
- The script does not automatically claim statistical significance. Three seeds are screening evidence.

## Limitations

The calibration procedure tests one-block restoration from an all-ternary pretrained model. It is a first-order sensitivity proxy, not causal attribution of final trained-model quality. Both input and output projections in a selected block remain full precision. The runner creates no results until executed on a real CUDA system. Preserve logs, manifests, checkpoints, and exact git SHA.
