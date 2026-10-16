#!/bin/bash
cd /root/Dicefest
while kill -0 2511853 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_hk20.py x4_gpm,x4_lepa > exp/hk20.log 2>&1
