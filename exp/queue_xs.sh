#!/bin/bash
# no-subzone-TE XGB members, after the LGB chain (RAM budget)
cd /root/Dicefest
while kill -0 2736048 2>/dev/null; do sleep 60; done
.venv/bin/python -u src/exp_v13_xs.py xs_gpm,xs_epm_wM > exp/v13_xs.log 2>&1
