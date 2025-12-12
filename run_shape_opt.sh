#!/usr/bin/env bash

# Run shape optimization
if ! python3 shape_opt.py "$@"; then
    echo ""
    echo "shape_opt.py failed - skipping plotting scripts."
    echo ""
    exit 0
fi

# If succeeds, run the plotting scripts in parallel
python3 plot_final_tissues.py "$@" &
PID2=$!

python3 plot_best_growth.py "$@" &
PID3=$!

wait $PID2
wait $PID3
