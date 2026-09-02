#!/usr/bin/env bash
# Is the 7-wave SBC failure real, or optimization scatter?
#
# The 6/7/8/9 comparison gives one seed per arm, and 7w's worst p-value
# (crra 0.0091, delta 0.0292) is close enough to the 0.05 line that a single
# draw cannot distinguish structure from noise. This trains 7 waves three more
# times, varying *only* network init and batch order: the windowed dataset, the
# panel split and the 1000 SBC draws all carry their own fixed seeds, so any
# spread across these runs is optimization noise.
#
# Read it against the seed-0 result already in outputs/window_comparison_short:
#   beta 0.0690   delta 0.0292   crra 0.0091
# If the three seeds scatter across the 0.05 line, the wave-count decision in
# SIMULATOR_SPEC 6.1.2 rests on noise and has to be redone with replication.
# If they cluster below it, 7 waves is genuinely miscalibrated and PSID's
# 7-wave ceiling (PSID_DATA.md) is a real obstacle.
#
# Run CONCURRENTLY, not in sequence. Training is latency-bound, not
# compute-bound -- a single run holds the V100 at ~11% and 1.5 GB -- so three
# processes finish in roughly the wall time of one. Batch size and the scoring
# path are deliberately left alone: they are the experiment's control, and
# changing either would confound the very question being asked.
#
# Usage:  nohup setsid scripts/queue_seed_replication.sh > logs/queue_seeds.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

WAIT_PID="${WAIT_PID:-708425}"
SEEDS="${SEEDS:-1 2 3}"
WAVES="${WAVES:-7}"
N_SBC="${N_SBC:-1000}"
CACHE="outputs/window_comparison/sbc_sims.pt"

if [ ! -f "$CACHE" ]; then
    echo "$(date -u '+%F %T') no SBC cache at $CACHE -- refusing."
    echo "Without it each seed re-simulates 1000 draws (3.46 h of GPU) for"
    echo "identical results, and the seeds would no longer share one SBC set."
    exit 1
fi

echo "$(date -u '+%F %T') waiting for pid $WAIT_PID (6/7/8/9 comparison)"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 120
done
echo "$(date -u '+%F %T') pid $WAIT_PID exited; launching seeds $SEEDS"

pids=()
for s in $SEEDS; do
    out="outputs/seed_replication/seed${s}"
    mkdir -p "$out"
    .venv/bin/python scripts/compare_windows.py \
        --windows $WAVES \
        --train_seed "$s" \
        --n_sbc "$N_SBC" \
        --sbc_cache "$CACHE" \
        --out "$out" > "logs/seed${s}.log" 2>&1 &
    pids+=($!)
    echo "$(date -u '+%F %T') seed $s -> pid $! -> $out"
    sleep 20   # stagger: three simultaneous dataset builds thrash the page cache
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || { echo "$(date -u '+%F %T') pid $p exited nonzero"; fail=1; }
done
echo "$(date -u '+%F %T') all seeds done (fail=$fail)"
