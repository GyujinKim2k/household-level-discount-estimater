#!/usr/bin/env bash
# Windows per panel: does more augmentation fix the undercoverage?
#
# k is the number of random-age windows cut from each simulated panel. It is
# currently 8 (configs/npe/phase3.yaml, window.windows_per_panel). Note k adds
# NO information about theta -- the effective independent sample stays the
# panel count either way (SIMULATOR_SPEC 6.1.1). What it changes is how much
# the network sees of the age mapping, and how many gradient steps an epoch
# contains. Both plausibly affect how well the flow is fitted, and a badly
# fitted flow is exactly what the coverage shortfall looks like.
#
# So this is not a search for more data. It is a second attempt at the same
# target as the ensemble: reduce approximation error. If neither moves coverage
# off ~0.87, the residual is bias rather than variance and the next lever is
# capacity or architecture.
#
# Run at 7 waves: that is the window PSID can actually supply (PSID_DATA.md),
# so it is the one whose calibration has to be fixed for Phase 4 to proceed.
# Start range 25-46 keeps the age envelope at 25-59, matching the 10-wave arm.
#
# 5 seeds per k so the result is read the same replication-aware way as the
# wave matrix: coverage mean and spread first, per-seed KS pass rate second.
# A single p-value settles nothing here -- that error is what produced the
# retracted decision in SIMULATOR_SPEC 6.1.2.
#
# Usage:  nohup setsid scripts/run_k_sweep.sh > logs/k_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

KS="${KS:-5 8 10}"
SEEDS="${SEEDS:-0 1 2 3 4}"
WAVES="${WAVES:-7}"
START_LOW=25
START_HIGH=46
BATCH="${BATCH:-1024}"
LR="${LR:-1e-3}"
N_SBC="${N_SBC:-1000}"
STAGGER="${STAGGER:-90}"
CACHE="outputs/window_comparison/sbc_sims.pt"
WAIT_FOR="${WAIT_FOR:-}"

[ -f "$CACHE" ] || { echo "no SBC cache at $CACHE -- refusing"; exit 1; }

if [ -n "$WAIT_FOR" ]; then
    echo "$(date -u '+%F %T') waiting for pids: $WAIT_FOR"
    for p in $WAIT_FOR; do
        while kill -0 "$p" 2>/dev/null; do sleep 60; done
    done
    echo "$(date -u '+%F %T') all waited pids gone"
fi

# k=10 holds 573440 windows in host RAM while building, against 458752 at k=8
# and 286720 at k=5. Five concurrent builds of the largest fit in 22 GB only
# because they are staggered; do not raise the group size without checking.
for k in $KS; do
    echo "=== $(date -u '+%F %T') group: k=${k} windows/panel, ${WAVES} waves ==="
    pids=()
    for s in $SEEDS; do
        out="outputs/k_sweep/k${k}_s${s}"
        mkdir -p "$out"
        .venv/bin/python scripts/compare_windows.py \
            --windows "$WAVES" \
            --start_low "$START_LOW" --start_high "$START_HIGH" \
            --k "$k" \
            --train_seed "$s" --batch_size "$BATCH" --learning_rate "$LR" \
            --n_sbc "$N_SBC" --sbc_cache "$CACHE" \
            --out "$out" > "logs/ksweep_k${k}_s${s}.log" 2>&1 &
        pids+=($!)
        echo "$(date -u '+%F %T') k=${k} seed=${s} pid=$! -> $out"
        sleep "$STAGGER"
    done
    for p in "${pids[@]}"; do
        wait "$p" || echo "$(date -u '+%F %T') pid $p exited nonzero"
    done
    echo "=== $(date -u '+%F %T') k=${k} group done ==="
done

echo "$(date -u '+%F %T') k sweep complete"
