# Reproducibility and experiment protocol

This repository contains multiple experiment families. **Do not pool their metrics**: Pilot A and Pilot B differ in training-token budget, sequence length, optimizer steps, learning rate, validation protocol and final evaluation dtype.

## 1. Verified runtime record: Pilot B (`real_harpq_fast2M_s2`)

Source of truth: [`results/real_harpq_fast2M_s2/real_run_manifest.json`](../results/real_harpq_fast2M_s2/real_run_manifest.json).

| Field | Recorded value |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA Tesla T4 |
| Platform | Linux 6.6.159+, x86_64, glibc 2.39 |
| Model | `state-spaces/mamba2-130m` |
| Data | `HuggingFaceFW/fineweb-edu`, subset `sample-10BT` |
| Training tokens | 2,000,000 |
| Sequence length | 512 |
| Micro-batch | 1 |
| Gradient accumulation | 4 |
| Tokens / optimizer step | 2,048 |
| Optimizer steps | 976 |
| Learning rate | 1e-4 |
| Dtype | FP16 |
| Seed | 2 |
| Validation | 64 sequences; evaluation every 100 steps |
| Decay | shift; horizon-aware quantizer; horizon 128; uniform lag weighting |
| Ternary projections | enabled; group size 128; activation-aware scales |
| State loss | hybrid; weight 0.05 |

### Exact training command

Run from the `shift_scan/` source directory after installing dependencies and confirming the GPU:

```bash
python train_qat.py \
  --name real_harpq_fast2M_s2 \
  --pretrained state-spaces/mamba2-130m \
  --decay shift \
  --decay-group-size 1 \
  --decay-quantizer horizon \
  --decay-horizon 128 \
  --decay-lag-weight uniform \
  --ternary 1 \
  --group-size 128 \
  --activation-aware 1 \
  --state-weight 0.05 \
  --state-loss-type hybrid \
  --tokens 2000000 \
  --seq-len 512 \
  --micro-bs 1 \
  --accum 4 \
  --lr 0.0001 \
  --dtype fp16 \
  --val-seqs 64 \
  --eval-every 100 \
  --save-every 250 \
  --log-every 10 \
  --seed 2 \
  --out runs
```

### Exact evaluation and diagnostic commands

```bash
# CPU/GPU-independent repository correctness suite (CPU is sufficient for core tests)
python tests.py

# HARPQ calibration on the recorded held-out FineWeb-Edu shard
python calibrate_decay.py \
  --pretrained state-spaces/mamba2-130m \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --tokens 200000 \
  --seq-len 512 \
  --horizon 128 \
  --weight uniform \
  --dtype fp16 \
  --max-coefficients 20000 \
  --out results/real_harpq_fast2M_s2/fineweb_harpq_calibration.json

# Teacher and student zero-shot evaluation (7 tasks, zero-shot, no sample cap)
python eval_lm.py \
  --pretrained state-spaces/mamba2-130m \
  --dtype fp16 --ppl-length \
  --out results/real_harpq_fast2M_s2/teacher_fineweb_ppl.json

python eval_lm.py \
  --ckpt runs/real_harpq_fast2M_s2/last.pt \
  --dtype fp16 --ppl-length \
  --out results/real_harpq_fast2M_s2/student_fineweb_ppl.json

# Correctness + synthetic-shape kernel microbenchmarks
python kernels.py
```

The benchmark microbenchmarks use synthetic tensor shapes and are **not** end-to-end decode throughput. They do not establish a speed or energy claim.

## 2. Pilot A: 10M-token matched variants (separate protocol)

The preserved result summary records three variants (`nearest`, `harpq`, `dyadic2`) and seeds 0–2, but the complete per-run CLI command and full environment freeze were not preserved for every run. Available configuration facts:

| Setting | Pilot A |
|---|---|
| Training tokens | 10M |
| Sequence length | 2,048 |
| Tokens / step | 32,768 |
| Optimizer steps | 305 |
| Peak learning rate | 2e-4 |
| Final lm-eval dtype | BF16 |
| Hardware | Colab T4 for nearest seed 2 and seeds 0/1; Kaggle T4 for HARPQ and Dyadic2 seed 2 |
| Training-time validation | default trainer validation |

Pilot A's exact command is not reconstructed here because the per-run command line was not retained. Do not use the Pilot B command as if it were the Pilot A command.

## 3. Pilot B: 2M-token HARPQ run (single seed)

Pilot B is a single HARPQ run (seed 2), not a matched multi-variant comparison. Its complete manifest and selected outputs are preserved under `results/real_harpq_fast2M_s2/`.

Recorded final evaluation included WikiText word PPL 186.73, LAMBADA accuracy 5.61%, HellaSwag acc_norm 27.55%, PIQA acc_norm 56.20%, ARC-Easy acc_norm 33.88%, ARC-Challenge acc_norm 21.93%, and WinoGrande accuracy 51.38%. Teacher values were only recorded for tasks shown in `teacher_vs_student.csv`; do not infer missing teacher baselines.

## 4. Calibration and kernel result interpretation

- HARPQ calibration processed 200,192 tokens and sampled 20,000 coefficients. Retention RMSE changed from 0.034525 (nearest) to 0.034355 (HARPQ), about a 0.5% reduction. This is a coefficient diagnostic, not a model-quality or speed result.
- The custom ternary Triton GEMV correctness checks passed. In the recorded T4 microbenchmarks, the ternary path was 2.35×–17.05× slower than FP16 depending on shape and batch. No GPU speedup or energy reduction is supported.
- The integer shift-scan reference matched the int32 state exactly; reported output max error was 3.73e-08 for the kernel test.

## 5. Dependency reproducibility

- `.python-version` records the exact Python runtime for the Pilot B manifest.
- `requirements-t4-pilot-b.txt` pins the exact PyTorch build recorded for Pilot B.
- The uploaded run bundle did **not** include a complete `pip freeze`; exact versions for Transformers, Datasets, Triton, lm-eval and transitive packages are not available. The base `requirements.txt` therefore remains a set of lower bounds, not a complete lockfile. Capture a full `python -m pip freeze` in the next controlled rerun before claiming a fully pinned environment.
- A separate earlier manifest recorded PyTorch `2.11.0+cu130`; it is not the environment record for the Pilot B result in this directory. Keep the manifests attached to their own run IDs.

## 6. Reproduction checklist

1. Use the exact source archive SHA recorded in `real_run_manifest.json`.
2. Install the runtime matching `.python-version` and the PyTorch build in `requirements-t4-pilot-b.txt`.
3. Record a complete `pip freeze`, NVIDIA driver, GPU model and VRAM.
4. Run `python tests.py` before training.
5. Execute the exact training command above.
6. Run teacher and student evaluation with the same tasks, dtype and no sample cap.
7. Preserve checkpoints, JSON outputs, logs and SHA-256 hashes in a run-specific folder.
8. Do not combine Pilot A and Pilot B metrics in one aggregate table.