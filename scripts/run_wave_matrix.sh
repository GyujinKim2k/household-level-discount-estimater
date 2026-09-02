#!/usr/bin/env bash
# 7 vs 10 waves, 5 training seeds each, age envelopes matched.
#
# Replaces the single-seed comparison that produced SIMULATOR_SPEC 6.1.2.
# That decision does not survive replication: at 7 waves, holding the data,
# the panel split and the SBC draws fixed and varying only network init, the
# KS p-values moved over two to three orders of magnitude (seed 1 passed all
# three parameters, seed 2 failed all three). A single p-value cannot separate
# wave count from seed luck -- under perfect calibration p is Uniform(0,1) by
# construction, so one draw is a maximally noisy statistic.
#
# Two design fixes over the earlier runs:
#
#   1. AGE ENVELOPE MATCHED. Previously every arm started in U{25..40}, so the
#      longer arm also reached older ages -- length was confounded with how far
#      into life the window ran, the same defect already flagged against 15w.
#      Both arms now occupy ages 25-59 exactly:
#          10 waves (20 yrs): start U{25..40} -> ends 44-59
#           7 waves (14 yrs): start U{25..46} -> ends 38-59
#      The start distributions necessarily differ (16 vs 22 choices); that is
#      unavoidable at different lengths, and the envelope is the confound that
#      matters because it carries the retirement crossing and the mortality-free
#      forward pass (SIMULATOR_SPEC 4).
#
#   2. COVERAGE IS THE PRIMARY METRIC, not ks_p. Across four seeds coverage_90
#      moved by ~0.02 while ks_p moved 1000x. Decide on coverage; report the
#      per-seed KS pass rate beside it as a secondary, replication-aware figure.
#
# Batch 1024 / lr 1e-3 comes from outputs/batch_pilot: against batch 256 it
# costs ~0.10 nats of held-out log q and runs 2.9x faster (1h51m vs 5h20m),
# with coverage identical to within 0.007 -- far inside the seed noise. Both
# arms share the config, so any residual optimization penalty is common to
# both and cannot favour either. Production models should revisit batch 256.
#
# Usage:  nohup setsid scripts/run_wave_matrix.sh > logs/wave_matrix.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
BATCH="${BATCH:-1024}"
LR="${LR:-1e-3}"
N_SBC="${N_SBC:-1000}"
GROUP="${GROUP:-5}"        # concurrent processes; see the RAM note below
STAGGER="${STAGGER:-90}"
CACHE="outputs/window_comparison/sbc_sims.pt"

# 7 waves starts later so both arms end at 59. Keep these paired.
declare -A START_HIGH=( [7]=46 [10]=40 )

[ -f "$CACHE" ] || { echo "no SBC cache at $CACHE -- refusing"; exit 1; }

# Each process holds the full windowed set in host RAM while building
# (458752 x waves x 5 float32 plus the panels it is cut from). Five at once fit
# in 22 GB; ten do not. The stagger keeps the build peaks from coinciding.
launch() {   # launch <waves> <seed>
    local w=$1 s=$2 out="outputs/wave_matrix/w${w}_s${s}"
    mkdir -p "$out"
    .venv/bin/python scripts/compare_windows.py \
        --windows "$w" \
        --start_low 25 --start_high "${START_HIGH[$w]}" \
        --train_seed "$s" --batch_size "$BATCH" --learning_rate "$LR" \
        --n_sbc "$N_SBC" --sbc_cache "$CACHE" \
        --out "$out" > "logs/matrix_w${w}_s${s}.log" 2>&1 &
    echo "$(date -u '+%F %T') w=${w} seed=${s} pid=$! -> $out"
}

for w in 7 10; do
    echo "=== $(date -u '+%F %T') group: ${w} waves, start 25-${START_HIGH[$w]} ==="
    pids=()
    for s in $SEEDS; do
        launch "$w" "$s"; pids+=($!)
        sleep "$STAGGER"
    done
    for p in "${pids[@]}"; do
        wait "$p" || echo "$(date -u '+%F %T') pid $p exited nonzero"
    done
    echo "=== $(date -u '+%F %T') ${w}-wave group done ==="
done

echo "$(date -u '+%F %T') matrix complete"
