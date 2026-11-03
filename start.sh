#!/bin/sh
# One container, two processes: headless paper-trading worker + dashboard (shared SQLite file).
set -e
python worker.py &
exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
