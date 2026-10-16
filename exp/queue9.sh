#!/bin/bash
cd /root/Dicefest
while kill -0 2475555 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_hold.py xgbrev > exp/hold_xgbrev.log 2>&1
