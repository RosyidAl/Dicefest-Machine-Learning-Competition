#!/bin/bash
cd /root/Dicefest
until grep -q "^\[xgb_epm_wM_tewc\]\|Error" exp/cell1.log 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_k10.py > exp/k10.log 2>&1
