#!/usr/bin/env bash
# Run the full-dataset wave-count comparison once generation releases the GPU.
#
# The wave count (5 / 10 / 15) is the one observation-model choice still open:
# configs/npe/phase3.yaml carries n_waves 10 provisionally, decided on the
# 8192-draw prefix under the old fixed-window design. This settles it on all
# 65536 draws under the random-age pipeline, on calibration as well as fit.
#
# Replaces the earlier queue_sbc.sh, which ran plain SBC against the 8192-draw
# posterior. That posterior has seq_len 5 and the analysis window is now 10
# (configs/npe/phase3.yaml), so it would fail on shape alone -- and the useful
# question is no longer "is this one posterior calibrated" but "which window
# gives a posterior that is both sharp and calibrated on the full dataset".
#
# The SBC simulations are the expensive part (~12.4 s per draw, GPU) and cannot
# share the card with generation: the recorded theta_batch=16, chunk=16 peaks at
# 10.9 GiB against the V100's 16 GiB. Hence the wait.
#
# Usage:  nohup setsid scripts/queue_post_generation.sh > logs/queue_post_gen.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

GEN_PID="${GEN_PID:-75501}"
GEN_LOG="${GEN_LOG:-logs/generate_phase3_v100_panel.log}"
OUT_DIR="${OUT_DIR:-outputs/window_comparison}"
WINDOWS="${WINDOWS:-5 10 15}"
N_SBC="${N_SBC:-1000}"

echo "$(date -u '+%F %T') waiting for generation (pid $GEN_PID) to release the GPU"
while kill -0 "$GEN_PID" 2>/dev/null; do
    sleep 300
done
echo "$(date -u '+%F %T') pid $GEN_PID exited"

# Exiting is not finishing. A crashed run should be restarted, not analysed:
# a truncated Sobol prefix is still a valid dataset, but silently comparing
# windows on 40% of the draws and reporting it as the full-dataset result is
# how a wrong number gets into a paper.
if ! grep -q "All shards complete" "$GEN_LOG"; then
    echo "$(date -u '+%F %T') generation did NOT complete -- refusing to run."
    echo "Last log line: $(tail -n 1 "$GEN_LOG")"
    echo "Restart generation, or re-run this with --train_n set to the prefix"
    echo "that actually exists if a partial comparison is what you want."
    exit 1
fi

# Build the canonical training set from the stored panels: random start ages,
# 8 windows per panel, per-wave age channel. Generation writes a fixed 5-wave
# `x` into each shard, which is not the analysis window. Costs seconds, no GPU.
echo "$(date -u '+%F %T') building the random-age windowed dataset"
.venv/bin/python scripts/build_windowed_dataset.py \
    --out data/processed/phase3_windowed.pt || exit 1

echo "$(date -u '+%F %T') generation complete; starting window comparison"
exec .venv/bin/python scripts/compare_windows.py \
    --windows $WINDOWS \
    --n_sbc "$N_SBC" \
    --out "$OUT_DIR"
