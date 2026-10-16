#!/bin/bash
cd /root/Dicefest
while kill -0 2442823 2>/dev/null || kill -0 2468200 2>/dev/null; do sleep 30; done
.venv/bin/python -u src/exp_hold.py nn > exp/hold_nn.log 2>&1
