#!/bin/bash
# Execute a final notebook headless (no cell timeout) in its own run directory and print the sha256 of the
# submission it writes. Each notebook writes <rundir>/dicefest_nb/ (work files + submission.csv), so the full and the
# inference notebook can run at the same time without touching each other's files.
# usage: bash scripts/run_notebook.sh final/DICEFEST2026_<TEAM>.ipynb final/nbrun_full
set -euo pipefail
cd /root/Dicefest
NB=${1:?notebook path}
RUNDIR=${2:?run directory}
mkdir -p "$RUNDIR"
cp "$NB" "$RUNDIR/"
NAME=$(basename "$NB")
export DICEFEST_DATA_DIR=/root/Dicefest/data/raw
export DICEFEST_ARTIFACTS=/root/Dicefest/final/artifacts
echo "[$(TZ=Asia/Jakarta date '+%F %H:%M')] start $RUNDIR/$NAME"
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 --output "${NAME%.ipynb}_executed.ipynb" "$RUNDIR/$NAME"
echo "[$(TZ=Asia/Jakarta date '+%F %H:%M')] done"
sha256sum "$RUNDIR/dicefest_nb/submission.csv"
