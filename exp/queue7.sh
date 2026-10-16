#!/bin/bash
cd /root/Dicefest
while kill -0 2335373 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_v4_nn.py > exp/v4_nn.log 2>&1
