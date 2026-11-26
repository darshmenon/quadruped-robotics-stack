#!/usr/bin/env python3
"""n, reached, falls and Wilson 95% intervals for the NMPC flat-world runs, only from the CSVs.

    python3 scripts/nmpc_wilson_report.py

Groups: main CSV rows 17-26 (lightly loaded 10 runs, data rows counted from 1) and
docs/data/gazebo_benchmark_nmpc_rep2.csv (the replication), then the combined set.
"""
import csv
import math
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(path, lo=None, hi=None):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["controller"] == "nmpc"]
    return rows[lo:hi] if lo is not None else rows


def line(name, rows):
    n = len(rows)
    reached = sum(int(r["reached"]) for r in rows)
    falls = sum(int(r["fell"]) for r in rows)
    lo, hi = wilson(reached, n)
    flo, fhi = wilson(falls, n)
    print(f"{name}: n={n} reached={reached} ({lo:.1f}-{hi:.1f}% Wilson 95%) falls={falls} ({max(flo, 0):.1f}-{fhi:.1f}%)")


main = os.path.join(D, "gazebo_benchmark.csv")
rep2 = os.path.join(D, "gazebo_benchmark_nmpc_rep2.csv")
first10 = load(main)[13:23]   # NMPC rows 14-23 of the 23 NMPC rows = the 10 lightly loaded runs
line("lightly loaded 10 (main CSV)", first10)
if os.path.exists(rep2):
    r2 = load(rep2)
    line("replication (rep2 CSV)", r2)
    line("combined", first10 + r2)
else:
    print("rep2 CSV not present yet")
