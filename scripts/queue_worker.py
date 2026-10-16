"""Two-lane job queue (MASTER_PROMPT 3.4). Items are files in queue/<lane>/pending/, oldest first:
  *.yaml -> python src/train_cv.py --config <item>
  *.sh   -> bash <item>   (for jobs outside the generic runner)
The item moves pending -> running -> done | failed; the job's stdout/stderr go to logs/queue_<lane>_<item>.log.
The worker exits when the file queue/<lane>/STOP exists, and otherwise polls every 30 s.

usage: nohup setsid .venv/bin/python scripts/queue_worker.py --lane gpu > logs/worker_gpu.log 2>&1 &
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tracking import now_wib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lane", required=True, choices=["gpu", "cpu"])
a = ap.parse_args()
q = f"queue/{a.lane}"
for s in ("pending", "running", "done", "failed"):
    os.makedirs(f"{q}/{s}", exist_ok=True)
with open("logs/workers.pid", "a") as f:
    f.write(f"{a.lane} {os.getpid()} {now_wib()}\n")
env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "6"))
while not os.path.exists(f"{q}/STOP"):
    items = sorted(os.listdir(f"{q}/pending"), key=lambda n: os.path.getmtime(f"{q}/pending/{n}"))
    if not items:
        time.sleep(30)
        continue
    it = items[0]
    shutil.move(f"{q}/pending/{it}", f"{q}/running/{it}")
    path = f"{q}/running/{it}"
    cmd = [sys.executable, "-u", "src/train_cv.py", "--config", f"configs/{it}" if os.path.exists(f"configs/{it}") else path] if it.endswith(".yaml") else ["bash", path]
    print(f"[{now_wib()}] start {it}", flush=True)
    with open(f"logs/queue_{a.lane}_{os.path.splitext(it)[0]}.log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    dst = "done" if rc == 0 else "failed"
    shutil.move(path, f"{q}/{dst}/{it}")
    print(f"[{now_wib()}] {dst} {it} rc={rc}", flush=True)
print(f"[{now_wib()}] STOP file found, exiting", flush=True)
