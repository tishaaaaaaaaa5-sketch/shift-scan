# Seed-variance report: `student_ppl` (lower is better)

Source: `/mnt/user-data/outputs/notebook_data/seed0_seed1_training.csv`. Seed-level values (reps averaged). **n per method: {'dyadic2': 2, 'harpq': 2, 'nearest': 2}**

> WARNING: fewer than 5 seeds for some methods. sd, p-values and power numbers are unstable at this n; report every individual value and avoid the word 'significant'.

## Per-method seed spread

| method | n_seeds | mean | sd | min | max | range | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| dyadic2 | 2 | 101.4 | 2.383 | 99.74 | 103.1 | 3.37 | 99.74 | 103.1 |
| harpq | 2 | 98.65 | 3.318 | 96.3 | 101 | 4.693 | 96.3 | 101 |
| nearest | 2 | 102.8 | 9.48 | 96.08 | 109.5 | 13.41 | 96.08 | 109.5 |


## Pairwise comparisons

`adv_A_over_B` is oriented so positive = A better. `flip_single_seed` = chance that a one-seed-each comparison reverses the sign of the n-seed mean gap. `seeds_for_80pct_power` is a normal-approximation estimate.

| A | B | adv_A_over_B | pooled_sd | gap_over_sd | perm_p | perm_p_holm | flip_single_seed | single_gap_p05 | single_gap_p95 | seeds_for_80pct_power |
|---|---|---|---|---|---|---|---|---|---|---|
| dyadic2 | harpq | -2.778 | 2.889 | -0.9618 | 0.6667 | 1 | 0.25 | -6.305 | 0.7475 | 17 |
| dyadic2 | nearest | 1.359 | 6.912 | 0.1965 | 1 | 1 | 0.5 | -6.525 | 9.241 | 407 |
| harpq | nearest | 4.137 | 7.102 | 0.5825 | 1 | 1 | 0.5 | -4.209 | 12.48 | 47 |


## Claim templates


A 'matches' claim is an equivalence claim: judge it from the paired CI in `pairwise.csv`, not from p.
