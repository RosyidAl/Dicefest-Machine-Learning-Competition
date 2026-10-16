#!/bin/bash
# 1) stop the holdout PLR run (PID 2607411) once p1_gpm is done (p1_epm_wM goes straight to full data)
# 2) after full-data p1_lepm (PID 2607370) start full-data p1_epm_wM
cd /root/Dicefest
done_h=0; done_f=0
while [ $done_h -eq 0 ] || [ $done_f -eq 0 ]; do
  if [ $done_h -eq 0 ] && grep -q "^\[p1_gpm\]" exp/hold_plr2.log; then kill 2607411; done_h=1; fi
  if [ $done_h -eq 0 ] && ! kill -0 2607411 2>/dev/null; then done_h=1; fi
  if [ $done_f -eq 0 ] && ! kill -0 2607370 2>/dev/null; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u src/exp_v9_nn.py p1_epm_wM > exp/v9_nn_b.log 2>&1 &
    done_f=1
  fi
  sleep 30
done
wait
