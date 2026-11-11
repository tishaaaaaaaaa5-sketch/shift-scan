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

The [seed-variance report](seed-variance-report.md) records only two seeds per method for its `student_ppl` analysis and explicitly warns that p-values and spread estimates are unstable at this sample size. Present it as exploratory evidence, not as a statistically conclusive ranking. Its perplexity values are from a different evaluation artifact/configuration than the post-training decay comparison in the README.

## Notebook organization

The canonical controlled validation notebook is `notebooks/validation/ShiftScan_Controlled_Validation.ipynb`. Earlier variants are retained under `notebooks/archive/validation-versions/` so their history is not lost. The roadmap notebook is `notebooks/roadmap/ShiftScan_Research_Roadmap.ipynb`; its earlier predecessor is archived.

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

## Current limitations

- The notebook archive contains historical variants; compare their contents and outputs before treating them as interchangeable.
- The repository does not yet provide a verified pinned dependency lockfile or a tested one-command reproduction script.
- Large binary output bundles are preserved as research artifacts; document their provenance before treating them as canonical checkpoints.
- Hardware measurements should be repeated across batch sizes and sequence lengths with warm-up, synchronization and multiple trials before making general performance claims.
