# `real_harpq_fast2M_s2` — Pilot B

This directory contains one **single-seed HARPQ pilot**, not a matched comparison. The full run manifest is `real_run_manifest.json`; it records all stages as successful and stores the exact training command and environment.

## Run configuration

- Mamba-2 130M (`state-spaces/mamba2-130m`)
- FineWeb-Edu `sample-10BT`
- 2,000,000 training tokens; sequence length 512; 976 optimizer steps
- Seed 2; micro-batch 1; gradient accumulation 4; learning rate 1e-4
- FP16; Tesla T4; Python 3.13.16; PyTorch 2.14.1+cu130; CUDA 13.0
- HARPQ shift decay, horizon 128, uniform lag weighting; ternary projections with group size 128 and activation-aware scales

## Held-out evaluation

| Metric | Teacher | HARPQ student |
|---|---:|---:|
| WikiText word perplexity | 24.8712 | 186.7261 |
| LAMBADA perplexity | 16.7889 | 2717.4212 |
| LAMBADA accuracy | 43.72% | 5.61% |
| HellaSwag acc_norm | 35.26% | 27.55% |
| PIQA acc_norm | 63.98% | 56.20% |
| ARC-Easy acc_norm | 42.09% | 33.88% |
| ARC-Challenge acc_norm | 24.23% | 21.93% |
| WinoGrande accuracy | 52.33% | 51.38% |

The student remains substantially behind the teacher, especially on language modeling and LAMBADA. Do not present this as a successful student replication.

## Kernel and calibration findings

- HARPQ calibration processed 200,192 tokens and sampled 20,000 coefficients. Retention RMSE changed from 0.034525 (nearest) to 0.034355 (HARPQ); this is a calibration diagnostic only.
- Kernel correctness checks passed. The custom ternary Triton GEMV was 2.35×–17.05× slower than FP16 in the recorded T4 microbenchmarks. No speedup claim is supported.

## Files

- `real_run_manifest.json`: exact command, environment, settings, source archive hash, stage status and artifact hashes.
- `teacher_fineweb_ppl.json`, `student_fineweb_ppl.json`: zero-shot task metrics and long-context loss-by-position values.
- `teacher_vs_student.csv`: direct teacher/student comparison.
- `fineweb_harpq_calibration.json`: coefficient calibration diagnostics.
- `kernel_test_status.json`, `kernels.log`: correctness and latency evidence.
- `tests.log`: repository correctness test output.
- PNGs: result visualizations generated from the JSON/CSV values.

## Do not pool

Pilot B differs from the Pilot A 10M-token, 2,048-sequence, 305-step, 2e-4 protocol. Keep all values in this folder separate from Pilot A and from the recovered 1M-token LR-probe records.