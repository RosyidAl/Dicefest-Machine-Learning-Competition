#!/bin/bash
cd /root/Dicefest
while kill -0 2287439 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_zoo4.py > exp/zoo4.log 2>&1
