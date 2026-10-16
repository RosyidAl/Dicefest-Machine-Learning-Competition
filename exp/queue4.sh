#!/bin/bash
cd /root/Dicefest
while kill -0 2306320 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_zoo5.py > exp/zoo5.log 2>&1
