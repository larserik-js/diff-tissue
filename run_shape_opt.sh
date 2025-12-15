#!/usr/bin/env bash

# Run shape optimization
if ! uv run python shape_opt.py "$@"; then
    echo ""
    echo "shape_opt.py failed - skipping plotting scripts."
    echo ""
    exit 0
fi

# If succeeds, run the plotting scripts in parallel
uv run python plot_final_tissues.py "$@" &
PID2=$!

uv run python plot_best_growth.py "$@" &
PID3=$!

wait $PID2
wait $PID3
