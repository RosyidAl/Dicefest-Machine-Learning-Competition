#!/bin/bash
# run a determinism check script twice in fresh processes and diff the outputs
cd /root/Dicefest
s=$1; out=logs/repro_$(basename $s .py)
.venv/bin/python -u $s > $out.run1 2>&1
.venv/bin/python -u $s > $out.run2 2>&1
if diff <(grep -v Warn $out.run1) <(grep -v Warn $out.run2) > /dev/null; then echo "IDENTICAL" > $out.result; else echo "DIFFERENT" > $out.result; fi
grep -v Warn $out.run1 >> $out.result; echo --- >> $out.result; grep -v Warn $out.run2 >> $out.result
