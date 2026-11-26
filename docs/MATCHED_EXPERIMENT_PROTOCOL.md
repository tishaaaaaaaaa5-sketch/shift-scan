# Matched experiment protocol: reducing the ternary quality gap

This document defines the next controlled experiment. It is a protocol, not a report of completed results.

## Objective

Measure whether sensitivity-guided mixed precision and improved ternary calibration reduce the language-model quality gap while preserving a clearly specified compression or arithmetic benefit.

## Experimental arms

Run every arm from the same pretrained Mamba-2 130M checkpoint, tokenizer, training-token stream, sequence length, optimizer schedule, batch/accumulation configuration, evaluation code and hardware class.

| Arm | Weight representation | Purpose |
|---|---|---|
| A. FP reference | FP16/BF16 | Matched quality reference |
| B. Uniform ternary | Ternary on all eligible target projections | Baseline quantization penalty |
| C. Activation-aware ternary | Ternary with activation-aware scales | Calibration contribution |
| D. Sensitivity-guided mixed precision | Ternary except protected high-sensitivity layers | Mixed-precision contribution |
| E. QAT + mixed precision | Same policy as D, trained with QAT/distillation | Training adaptation contribution |

Do not compare runs with different token budgets, sequence lengths, data subsets or evaluation dtypes as if they were matched.

## Controls and preregistration

Before running:
1. Freeze the source commit, checkpoint SHA-256, tokenizer revision and dataset shard identifiers.
2. Freeze all training hyperparameters and evaluation commands.
3. Select at least three seeds; use the same seed list for every arm.
4. Record the layer-selection rule before inspecting test results.
5. Keep validation data for selection and reserve a separate test stream for final reporting.
6. Do not tune thresholds or choose checkpoints on the final test set.

## Required outcomes

Report each seed and aggregate mean ± standard deviation for:
- validation and held-out language-model perplexity;
- cross-entropy difference against the matched FP reference;
- long-context loss by position at fixed context lengths;
- parameter count and effective bits per parameter;
- checkpoint size and peak memory;
- prefill throughput and decode tokens/s at fixed batch sizes;
- median and p95 latency after warm-up;
- energy or power only when measured with a documented instrument and protocol.

Report kernel microbenchmarks separately from end-to-end model throughput.

## Layer-sensitivity analysis

For each eligible projection, record:
- module name and tensor shape;
- baseline activation/weight statistics;
- quantization error and output/state perturbation;
- validation loss change when quantized individually;
- cumulative loss change under the selected mixed-precision policy.

Publish the complete layer table, not only selected examples. Define the protection threshold using validation data only.

## Statistical reporting

Use paired seed-level comparisons where possible. Publish raw per-seed values, confidence intervals and the exact statistical procedure. Three seeds are a minimum exploratory estimate, not a guarantee of high statistical power.

## Acceptance criteria

Do not call the quality gap “reduced” unless the matched held-out comparison shows a reproducible improvement over uniform ternary across seeds. Do not claim a speedup unless end-to-end measurements beat the matched FP baseline on the same hardware and workload.

## Run manifest template

Each run should save a machine-readable manifest with:
- run ID, git commit and source archive hash;
- model/checkpoint and tokenizer revisions;
- dataset name, split, shard and checksum;
- seed, token budget, sequence length and optimizer settings;
- quantization policy and protected-layer list;
- Python, PyTorch, CUDA, driver and GPU versions;
- exact command line and evaluation command;
- output artifact paths and SHA-256 checksums.

The current repository contains exploratory results with different protocols. This protocol is intended to make the next comparison genuinely matched; it does not retroactively make prior runs comparable.
