#!/bin/bash
cd /root/Dicefest
while kill -0 2307221 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_zoo6.py > exp/zoo6.log 2>&1
