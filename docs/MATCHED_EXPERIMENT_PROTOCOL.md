# Matched experiment protocol

This is a proposed protocol, not a report of completed experiments.

## Question and conditions

At a fixed token budget and evaluation setup, does sensitivity-guided mixed precision reduce the quality gap relative to uniformly ternary projections?

| ID | Condition | Purpose |
|---|---|---|
| A | FP16/full-precision baseline | Reference quality and runtime |
| B | Uniform ternary projections | Measure ternary quality gap |
| C | Sensitivity-guided mixed precision | Test protection of sensitive layers |
| D | Randomly protected layers, matched protected-parameter budget | Control for simply using more full-precision parameters |

Use the same Mamba-2 130M initialization, tokenizer, data stream, optimizer schedule, sequence length, token budget, batch/accumulation settings, checkpoint-selection rule, and evaluation code for all conditions.

## Controls

- Use at least 3 seeds for screening; target 5+ for stronger estimates.
- Predefine seeds and publish failed/incomplete runs.
- Keep train/validation/test token streams disjoint; record source revisions and hashes.
- Fix calibration data and sensitivity scoring before test evaluation.
- Match C and D by protected parameter count, not layer count.
- Use identical benchmark versions, prompts, sample limits, and scoring settings.
- Report per-seed results, mean ± standard deviation, and paired uncertainty intervals.
- Do not pool metrics from different token budgets, checkpoints, datasets, or evaluation scripts.

## Outcomes

Primary: held-out validation negative log-likelihood / perplexity.

Secondary: LAMBADA accuracy and perplexity; HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and WinoGrande accuracy; parameter count by precision; checkpoint size; peak memory; throughput and latency; energy only if directly measured; per-layer sensitivity and protected-parameter fraction.

## Analysis

1. B vs A quantifies the uniform ternary gap.
2. C vs B quantifies the mixed-precision benefit.
3. C vs D tests whether sensitivity-guided allocation beats a parameter-matched random control.
4. Report paired per-seed differences and all attempted runs.

## Run manifest

Record git SHA/dirty status; Python, OS, CUDA, driver, GPU, PyTorch, Triton, Transformers, Datasets, and lm-eval versions; immutable model and dataset revisions; tokenizer revision; file hashes; seed; token count; sequence length; batch and accumulation; optimizer/scheduler; precision and quantization settings; exact command; checkpoint/evaluation hashes; timestamps; exit status; hardware measurements.

## Status

No new matched experiment is claimed here. Existing HARPQ Pilot B is a single-seed 2M-token pilot and must remain separate from earlier 10M-token studies. See [environment capture](ENVIRONMENT_CAPTURE.md) and [reproducibility notes](REPRODUCIBILITY.md).
