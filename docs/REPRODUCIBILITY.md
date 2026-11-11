# Reproducibility notes

This repository contains research notebooks, a report, output bundles and recovered run records. It is an experiment archive; it is not yet a packaged Python library or a one-command reproduction environment.

## Reported result families

| Result in README | Evidence category | Important qualification |
|---|---|---|
| TWN ternary PPL | Trained student runs | 4M-token result reports 3 seeds; 12M-token result is one run |
| Decay quantizer PPL / paired interval | Post-training decay evaluation | Keep this evaluation distinct from the seed-variance summary |
| Layer sensitivity | Post-training perturbation | README displays selected layers, not a complete layer-by-layer dump |
| Kernel latency | Custom Triton kernel measurement | One reported configuration; no broad speedup claim |

## Seed-variance report

The root `report.md` records only two seeds per method for its `student_ppl` analysis and explicitly warns that p-values and spread estimates are unstable at this sample size. It should be presented as exploratory evidence, not as a statistically conclusive ranking. Its perplexity values are from a different evaluation artifact/configuration than the post-training decay comparison in the README.

## Before reproducing a result

For each run, record and preserve:

- exact model checkpoint and checkpoint provenance;
- tokenizer and dataset revision / token-stream construction;
- evaluation split, token count and sequence length;
- quantizer configuration, calibration data and learned scales;
- random seed and all relevant training hyperparameters;
- Python, PyTorch, CUDA, Triton and driver versions;
- GPU model, precision mode, batch size and benchmark methodology;
- exact notebook / script cell sequence and output artifact.


- Hardware measurements should be repeated across batch sizes and sequence lengths with warm-up, synchronization and multiple trials before making general performance claims.
