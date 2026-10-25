# Abubakar Iqbal
# NAVTTC Weeks 7 to 12 - Master Test Runner

import os
import sys
import time
import subprocess
import pandas as pd

def run_suite():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weeks = ["Week7", "Week8", "Week9", "Week10", "Week11", "Week12"]

    venv_py = os.path.abspath(os.path.join(base_dir, "../../.venv/bin/python3"))
    py_exec = venv_py if os.path.exists(venv_py) else sys.executable

    print("NAVTTC Weeks 7 to 12 Test Suite - Abubakar Iqbal")
    print(f"Interpreter: {py_exec}")
    print("-" * 70)

    records = []
    t_start = time.time()

    for w in weeks:
        w_dir = os.path.join(base_dir, w)
        if not os.path.isdir(w_dir):
            continue
        scripts = sorted([f for f in os.listdir(w_dir) if f.endswith(".py")])
        for s in scripts:
            s_path = os.path.join(w_dir, s)
            t0 = time.time()
            res = subprocess.run([py_exec, s_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0
            status = "PASSED" if res.returncode == 0 else "FAILED"
            records.append({
                "Week": w,
                "Script": s,
                "Status": status,
                "Time (s)": round(elapsed, 2)
            })
            print(f"[{status}] {w} -> {s} ({elapsed:.2f}s)")
            if res.returncode != 0:
                print(f"  Error:\n{res.stderr[:200]}")

    total_time = time.time() - t_start
    df = pd.DataFrame(records)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    passed = sum(df["Status"] == "PASSED")
    total = len(df)
    print("-" * 70)
    print(f"Total: {total} | Passed: {passed} | Failed: {total - passed} | Runtime: {total_time:.2f}s")

    if passed == total:
        print("All Week 7 to 12 tasks executed successfully.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
