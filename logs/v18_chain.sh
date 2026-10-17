#!/bin/bash
# v18 FB07b members (src/exp_v18_knn2.py), full + holdout each; stops if MemAvailable < 3 GB (protects run #1).
cd /root/Dicefest
G=logs/v18_guard.log
ts() { TZ=Asia/Jakarta date '+%F %H:%M'; }
for job in "$@"; do
  mode=${job%%:*}; name=${job#*:}
  OMP_NUM_THREADS=4 .venv/bin/python -u src/exp_v18_knn2.py $mode $name > logs/v18_${mode}_${name}.log 2>&1 &
  pid=$!
  echo "[$(ts)] start $mode $name pid $pid" >> $G
  while kill -0 $pid 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail" -lt 3000000 ]; then
      echo "[$(ts)] low memory ($avail kB): kill $mode $name pid $pid, chain stopped" >> $G
      kill $pid; wait $pid; exit 1
    fi
    sleep 10
  done
  wait $pid; echo "[$(ts)] end $mode $name rc=$?" >> $G
done
echo "[$(ts)] chain done" >> $G
