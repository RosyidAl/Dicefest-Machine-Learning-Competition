#!/bin/bash
cd /root/Dicefest
while kill -0 2277281 2>/dev/null; do sleep 20; done
.venv/bin/python -u src/exp_lgb2.py > exp/lgb2.log 2>&1
