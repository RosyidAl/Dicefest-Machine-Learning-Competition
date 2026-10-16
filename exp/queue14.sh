#!/bin/bash
cd /root/Dicefest
while kill -0 2470790 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_htune.py cpu > exp/htune_cpu.log 2>&1
