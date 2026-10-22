# Master Test Runner & Verification Suite for NAVTTC Weeks 7 to 12
# Author: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

import os
import sys
import time
import subprocess
import pandas as pd

def run_suite():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weeks = ["Week7", "Week8", "Week9", "Week10", "Week11", "Week12"]

    print("=" * 80)
    print("NAVTTC WEEKS 7 TO 12 COMPREHENSIVE VERIFICATION & TEST SUITE")
    print("Author: Abubakar Iqbal | Program: NAVTTC / NIAI (NETSOL Technologies)")
    print("=" * 80)

    records = []
    total_start = time.time()

    for week in weeks:
        week_dir = os.path.join(base_dir, week)
        if not os.path.isdir(week_dir):
            continue

        scripts = sorted([f for f in os.listdir(week_dir) if f.endswith(".py")])
        for script in scripts:
            script_path = os.path.join(week_dir, script)
            t0 = time.time()

            # Execute via current python interpreter
            res = subprocess.run([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0

            status = "PASSED" if res.returncode == 0 else "FAILED"
            records.append({
                "Week": week,
                "Script": script,
                "Status": status,
                "Duration (s)": round(elapsed, 2)
            })

            print(f"[{status}] {week} -> {script} ({elapsed:.2f}s)")
            if res.returncode != 0:
                print(f"  ERROR OUTPUT:\n{res.stderr[:300]}")

    total_time = time.time() - total_start
    df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(df.to_string(index=False))

    passed_count = sum(df["Status"] == "PASSED")
    total_count = len(df)
    print("\n" + "-" * 80)
    print(f"Total Tasks Executed: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")
    print(f"Total Execution Runtime: {total_time:.2f} seconds")
    print("-" * 80)

    if passed_count == total_count:
        print("ALL TASKS COMPLETED AND VALIDATED WITH 100% SUCCESS.")
    else:
        print("WARNING: Some tasks encountered issues.")
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
