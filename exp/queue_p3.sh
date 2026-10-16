#!/bin/bash
# tuned PLR elec members after the fold-0 sweep has finished (RAM budget)
cd /root/Dicefest
until grep -q "^done5" /tmp/claude-0/-root-Dicefest/e1b793d8-9670-4e08-acb7-8bf572563c54/scratchpad/sweep.log; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u src/exp_v11_nn.py p3_lepm,p3_epm_wM > exp/v11_nn_e.log 2>&1
