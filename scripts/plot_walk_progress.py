#!/usr/bin/env python3
"""Plot how the trot-prior PPO walker improved and how well it tracks speed commands.

Reads the CSVs written by training/eval_checkpoints.py --csv (fixed-difficulty, clean env):
    docs/data/walk_progress_sweep.csv   one command, many checkpoints  -> left panel
    docs/data/walk_tracking.csv         one checkpoint, many commands  -> right panel

Usage:
    python3 scripts/plot_walk_progress.py
    python3 scripts/plot_walk_progress.py --out docs/images/walk_progress.png
"""
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def steps_of(model):
    m = re.search(r"_(\d+)_steps\.zip$", model)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--progress", default=os.path.join(ROOT, "docs/data/walk_progress_sweep.csv"))
    ap.add_argument("--tracking", default=os.path.join(ROOT, "docs/data/walk_tracking.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/images/walk_progress.png"))
    args = ap.parse_args()

    prog = sorted((steps_of(r["model"]), float(r["vx"]), float(r["fall_pct"]), float(r["cmd"]))
                  for r in read(args.progress) if steps_of(r["model"]))
    track = sorted((float(r["cmd"]), float(r["vx"]), float(r["fall_pct"])) for r in read(args.tracking))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = [p[0] / 1e6 for p in prog]
    a.plot(x, [p[1] for p in prog], "o-", color="#2a7ab9", lw=2, label="achieved forward speed")
    a.axhline(prog[0][3], ls="--", color="gray", lw=1, label=f"command ({prog[0][3]:.2f} m/s)")
    a.set(xlabel="PPO steps (millions)", ylabel="forward speed (m/s)", title="Walking speed vs training steps")
    for xi, p in zip(x, prog):
        if p[2] > 0:
            a.annotate(f"{p[2]:.0f}% falls", (xi, p[1]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    a.set_ylim(bottom=0)
    a.grid(alpha=0.3)
    a.legend(loc="upper left")

    cmd = [t[0] for t in track]
    b.plot(cmd, [t[1] for t in track], "o-", color="#d9622b", lw=2, label="achieved")
    lim = max(cmd + [t[1] for t in track]) * 1.1
    b.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="perfect tracking")
    b.set(xlabel="commanded vx (m/s)", ylabel="achieved vx (m/s)", title="Speed-command tracking (best checkpoint)",
          xlim=(0, lim), ylim=(0, lim))
    b.grid(alpha=0.3)
    b.legend(loc="upper left")
    falls = max(t[2] for t in track)
    b.text(0.97, 0.05, f"worst-case fall rate: {falls:.0f}%", transform=b.transAxes, ha="right", fontsize=9)

    fig.suptitle("Go2 PPO + trot prior, flat ground, 20 s episodes, clean env", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
