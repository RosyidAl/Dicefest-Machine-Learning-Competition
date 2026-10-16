#!/bin/bash
# GPU chain B: after full-data p1_gpm (PID 2631651): p1_gpgea, then p1_epa
cd /root/Dicefest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 2631651 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_v9_nn.py p1_gpgea > exp/v9_nn_gpgea.log 2>&1
.venv/bin/python -u src/exp_v9_nn.py p1_epa > exp/v9_nn_epa.log 2>&1
