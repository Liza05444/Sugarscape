#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
PYTHON=python3
if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
fi
export MPLBACKEND=Agg
exec "$PYTHON" -m streamlit run start.py --browser.gatherUsageStats false
