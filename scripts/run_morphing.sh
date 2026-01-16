#!/usr/bin/env bash

# Run morph.py
if ! uv run python morph.py "$@"; then
    echo ""
    echo "morph.py failed - skipping plotting script."
    echo ""
    exit 0
fi

# If succeeds, run the plotting script
uv run python plot_morphing.py "$@"
