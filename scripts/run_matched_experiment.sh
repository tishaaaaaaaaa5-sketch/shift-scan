#!/usr/bin/env bash
# Four-condition matched study. Requires a CUDA GPU and Hugging Face access.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/shift_scan_legacy"
OUT="$ROOT/results/matched_mixed_precision"
RUNS="$ROOT/runs/matched_mixed_precision"
PRETRAINED="${PRETRAINED:-state-spaces/mamba2-130m}"
TOKENS="${TOKENS:-2000000}"
SEEDS="${SEEDS:-0 1 2}"
DTYPE="${DTYPE:-fp16}"
mkdir -p "$OUT" "$RUNS"
cd "$SRC"

python prepare_layer_selection.py \
  --pretrained "$PRETRAINED" --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
  --seq-len 512 --micro-bs 1 --val-seqs 64 --val-docs 20000 --val-modulo 100 \
  --num-protect 2 --seed 2026 --dtype "$DTYPE" \
  --out "$OUT/layer_selection.json"

SENS_LAYERS="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["sensitivity_layers"])))' "$OUT/layer_selection.json")"
RAND_LAYERS="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["random_control_layers"])))' "$OUT/layer_selection.json")"

for SEED in $SEEDS; do
  COMMON=(--pretrained "$PRETRAINED" --decay exp --ternary 1 --group-size 128 \
    --activation-aware 0 --state-weight 0.05 --state-loss-type hybrid \
    --tokens "$TOKENS" --seq-len 512 --micro-bs 1 --accum 4 \
    --lr 0.0001 --dtype "$DTYPE" --val-seqs 64 --val-docs 20000 --val-modulo 100 \
    --eval-every 100 --save-every 250 --log-every 10 --seed "$SEED" --out "$RUNS")

  # A: full-precision student trained with the same distillation schedule.
  python train_qat.py --name "A_full_precision_student_s$SEED" "${COMMON[@]}" --ternary 0 --fp-layers ""
  # B: uniform ternary projections.
  python train_qat.py --name "B_uniform_ternary_s$SEED" "${COMMON[@]}" --fp-layers ""
  # C: two validation-sensitivity-selected blocks retain FP projections.
  python train_qat.py --name "C_sensitivity_mixed_s$SEED" "${COMMON[@]}" --fp-layers "$SENS_LAYERS"
  # D: random blocks, matched by protected block and projection parameter count.
  python train_qat.py --name "D_random_mixed_s$SEED" "${COMMON[@]}" --fp-layers "$RAND_LAYERS"

  python eval_lm.py --ckpt "$RUNS/A_full_precision_student_s$SEED/last.pt" --dtype "$DTYPE" \\
    --out "$OUT/A_full_precision_student_s$SEED.json"
  python eval_lm.py --ckpt "$RUNS/B_uniform_ternary_s$SEED/last.pt" --dtype "$DTYPE" \
    --out "$OUT/B_uniform_ternary_s$SEED.json"
  python eval_lm.py --ckpt "$RUNS/C_sensitivity_mixed_s$SEED/last.pt" --dtype "$DTYPE" \
    --out "$OUT/C_sensitivity_mixed_s$SEED.json"
  python eval_lm.py --ckpt "$RUNS/D_random_mixed_s$SEED/last.pt" --dtype "$DTYPE" \
    --out "$OUT/D_random_mixed_s$SEED.json"
done

python "$ROOT/scripts/summarize_matched_experiment.py" --input-dir "$OUT" --output "$OUT/summary.json"
echo "Matched experiment complete. Results: $OUT"
