#!/bin/bash
# GPU chain C: after xa_gpm (PID 2653104) and the v11 build (keeps RAM free for it), the other XGB + aggregate members
cd /root/Dicefest
while kill -0 2653104 2>/dev/null || ! grep -q done exp/sub_v11ncr.log 2>/dev/null; do sleep 60; done
.venv/bin/python -u src/exp_v10_agg.py xa_gpgea,xa_lgpm,xa_lepa,xa_epm_wM > exp/v10_agg_b.log 2>&1
