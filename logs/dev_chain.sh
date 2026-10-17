#!/bin/bash
# Generic chain of development member runs: each job is "script:mode:names" (e.g. src/exp_v18_knn2.py:full:xk2_gpm).
# Runs the jobs one after another; stops the chain if MemAvailable drops below 3 GB (protects the final run #1).
# usage: bash logs/dev_chain.sh <guard_log> job1 job2 ...
cd /root/Dicefest
G=${1:?guard log}; shift
ts() { TZ=Asia/Jakarta date '+%F %H:%M'; }
for job in "$@"; do
  script=${job%%:*}; rest=${job#*:}; mode=${rest%%:*}; names=${rest#*:}
  log=logs/dev_$(basename "$script" .py)_${mode}_${names//,/_}.log
  OMP_NUM_THREADS=4 .venv/bin/python -u "$script" "$mode" "$names" > "$log" 2>&1 &
  pid=$!
  echo "[$(ts)] start $script $mode $names pid $pid log $log" >> "$G"
  while kill -0 $pid 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail" -lt 3000000 ]; then
      echo "[$(ts)] low memory ($avail kB): kill $script $mode $names pid $pid, chain stopped" >> "$G"
      kill $pid; wait $pid; exit 1
    fi
    sleep 10
  done
  wait $pid; rc=$?
  echo "[$(ts)] end $script $mode $names rc=$rc" >> "$G"
  [ $rc -eq 0 ] || { echo "[$(ts)] chain stopped after failure" >> "$G"; exit 1; }
done
echo "[$(ts)] chain done" >> "$G"
