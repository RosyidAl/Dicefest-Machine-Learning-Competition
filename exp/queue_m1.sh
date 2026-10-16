#!/bin/bash
# stop chain D after t1_epm_wM (t1_gpgea is likely redundant: t1_gpm was not selected in v13), then run the
# multi-task member once chain C has finished
cd /root/Dicefest
until [ -f exp/pred/t1_epm_wM.npy ]; do sleep 60; done
sleep 20
kill 2793211 2>/dev/null
until [ -f exp/pred/p2_eui_wA.npy ]; do sleep 60; done
sleep 30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u src/exp_v14_mt.py m1 > exp/v14_mt.log 2>&1
