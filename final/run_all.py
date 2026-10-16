"""One entry point for the final DICE FEST 2026 submission: data -> models -> blend -> corrections -> submission.

usage:
  python final/run_all.py --work final/work/run1 --out final/output/submission_run1.csv --clean
  python final/run_all.py --stage members --lane gpu --work ...   (one lane only; used internally)

Environment: DICEFEST_DATA_DIR = folder with train.csv, test.csv, sample_submission.csv (default data/raw).
--clean deletes the work folder first (reproduction from scratch). Without it, finished members are reused.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def log_to(path):
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work", "run"))
    ap.add_argument("--out", default=os.path.join(HERE, "output", "submission.csv"))
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "members", "adoption", "assemble"])
    ap.add_argument("--lane", choices=["gpu", "cpu"])
    ap.add_argument("--only", default="", help="comma-separated member names (members stage, for tests)")
    a = ap.parse_args()
    import pipeline
    if a.stage == "members":
        pipeline.run_members(a.lane, a.work, only=[m for m in a.only.split(",") if m] or None)
        return
    if a.clean and a.stage == "all" and os.path.exists(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work, exist_ok=True)
    log = log_to(os.path.join(a.work, "run_all.log"))
    t0 = time.time()
    if a.stage == "all":
        log(f"start, work {a.work}")
        procs = {}
        for lane in ("gpu", "cpu"):
            lf = open(os.path.join(a.work, f"lane_{lane}.log"), "a")
            env = dict(os.environ, OMP_NUM_THREADS="4" if lane == "gpu" else "12")
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--stage", "members", "--lane", lane, "--work", a.work]
            if a.only:
                cmd += ["--only", a.only]
            procs[lane] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            log(f"lane {lane} pid {procs[lane].pid}")
        for lane, p in procs.items():
            rc = p.wait()
            log(f"lane {lane} finished rc={rc} ({(time.time() - t0) / 3600:.2f} h)")
            if rc != 0:
                raise SystemExit(f"lane {lane} failed, see {a.work}/lane_{lane}.log")
    if a.stage in ("all", "adoption"):
        info = pipeline.run_adoption(a.work)
        log(f"adoption done {info}")
    if a.stage in ("all", "assemble", "adoption"):
        sha = pipeline.assemble(a.work, a.out, log=log)
        log(f"done in {(time.time() - t0) / 3600:.2f} h, sha256 {sha}")


if __name__ == "__main__":
    main()
