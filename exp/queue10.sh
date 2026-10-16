#!/bin/bash
cd /root/Dicefest
while kill -0 2487051 2>/dev/null; do sleep 15; done
.venv/bin/python -u src/exp_hold_nnb.py > exp/hold_nnb.log 2>&1
