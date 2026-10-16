#!/bin/bash
# GPU chain A: after full-data p1_epm_wM (PID 2624332): p1_lgpm, then p1_leui_wA
cd /root/Dicefest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 2624332 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_v9_nn.py p1_lgpm > exp/v9_nn_lgpm.log 2>&1
.venv/bin/python -u src/exp_v9_nn.py p1_leui_wA > exp/v9_nn_leui.log 2>&1
