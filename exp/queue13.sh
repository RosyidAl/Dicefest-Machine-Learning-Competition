#!/bin/bash
cd /root/Dicefest
while kill -0 2518944 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_htune.py gpu > exp/htune_gpu2.log 2>&1
