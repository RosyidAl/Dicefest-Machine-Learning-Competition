#!/bin/bash
cd /root/Dicefest
while kill -0 2277159 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_zoo2.py > exp/zoo2.log 2>&1
.venv/bin/python -u src/exp_zoo3.py > exp/zoo3.log 2>&1
