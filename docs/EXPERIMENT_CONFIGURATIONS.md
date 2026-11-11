# Shift-Scan experiment configuration index

| Configuration | Purpose | Dataset/model | Training protocol | Status / reporting rule |
|---|---|---|---|---|
| Pilot A / nearest | Matched decay baseline | Mamba-2 130M; FineWeb-Edu | 10M tokens, seq 2048, 305 steps, peak LR 2e-4, BF16 eval | 3-seed family; full exact command not retained |
| Pilot A / HARPQ | Horizon-aware decay | Same family | Same 10M-token protocol; HARPQ horizon 128 | Keep seed-level values visible; small n |
| Pilot A / Dyadic2 | Two-term dyadic decay | Same family | Same 10M-token protocol; 105-level grid | Keep seed-level values visible; small n |
| Pilot B / `real_harpq_fast2M_s2` | Faster pilot and end-to-end evaluation | Mamba-2 130M; FineWeb-Edu | 2M tokens, seq 512, 976 steps, LR 1e-4, FP16 eval, seed 2 | Single run; not a matched comparison; see run-specific manifest |
| HARPQ calibration | Retention-curve coefficient diagnostic | Mamba-2 130M; FineWeb-Edu sample-10BT | 200,192 tokens; 20,000 coefficients; horizon 128 | Diagnostic only; not perplexity or speed |
| Triton kernel microbench | Projection-kernel latency | Synthetic tensor shapes; Tesla T4 | Batches 1 and 8 | Correctness passed; ternary Triton slower than FP16 |
| ResNet-20 seed-variance | Separate quantizer study | CIFAR-10 | 7 quantizers × 5 seeds | Separate project/experiment; do not mix with Mamba results |

The Pilot A and Pilot B protocols differ in sequence length, training token count, optimizer steps, learning rate, training validation and final evaluation dtype. They must not be pooled.