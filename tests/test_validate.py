import os
import subprocess
import sys


def test_sample_submission_format_passes():
    r = subprocess.run([sys.executable, "scripts/validate_submission.py", "data/raw/sample_submission.csv", "--sample",
                        "--no-log"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_best_current_passes():
    if os.path.exists("submissions/BEST_CURRENT.csv"):
        r = subprocess.run([sys.executable, "scripts/validate_submission.py", "submissions/BEST_CURRENT.csv",
                            "--no-log"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
