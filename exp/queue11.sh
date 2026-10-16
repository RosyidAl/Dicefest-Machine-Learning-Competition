#!/bin/bash
cd /root/Dicefest
while kill -0 2470746 2>/dev/null || kill -0 2485542 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_htune.py > exp/htune.log 2>&1
