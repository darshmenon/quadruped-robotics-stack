#!/usr/bin/env python3
"""Plot SB3 EvalCallback reward curves from every training/logs/*/evaluations.npz
found in the repo, as a small-multiples comparison grid (one panel per run,
since reward scales aren't comparable across tasks/reward functions).

Usage:
    python3 scripts/plot_eval_comparison.py
    python3 scripts/plot_eval_comparison.py --out docs/images/eval_comparison.png
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# (glob pattern relative to repo root, display label)
RUNS = [
    ("training/logs/mujoco/evaluations.npz", "flat walk"),
    ("training/logs/mujoco_fresh/evaluations.npz", "flat walk (fresh)"),
    ("training/logs/mujoco_gated_fresh/evaluations.npz", "walk + arm reach (gated)"),
    ("training/logs/stairs/evaluations.npz", "blind stairs"),
    ("training/logs/recovery/evaluations.npz", "fall recovery"),
    ("training/logs/vision_compare/blind/evaluations.npz", "rough terrain, blind"),
    ("training/logs/vision_compare/sighted/evaluations.npz", "rough terrain, sighted"),
    ("training/logs/mujoco_curriculum/flat/evaluations.npz", "curriculum: flat stage"),
    ("training/logs/mujoco_curriculum/rough/evaluations.npz", "curriculum: rough stage"),
    ("training/logs/mujoco_curriculum/stairs/evaluations.npz", "curriculum: stairs stage"),
    ("training/logs/parkour/evaluations.npz", "parkour"),
    ("training/logs/gazebo/evaluations.npz", "gazebo backend"),
]


def load_runs(repo_root):
    found = []
    for rel_path, label in RUNS:
        path = os.path.join(repo_root, rel_path)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        timesteps = data["timesteps"]
        results = data["results"]
        if len(timesteps) == 0:
            continue
        found.append(
            {
                "label": label,
                "path": rel_path,
                "timesteps": timesteps,
                "mean": results.mean(axis=1),
                "std": results.std(axis=1),
            }
        )
    return found


def plot(runs, out_path):
    n = len(runs)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

    for i, run in enumerate(runs):
        ax = axes[i // ncols][i % ncols]
        x = run["timesteps"]
        mean = run["mean"]
        std = run["std"]
        ax.plot(x, mean, marker="o", color="#2563eb")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#2563eb")
        ax.set_title(run["label"], fontsize=10)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("eval mean reward")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.3)

    # hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("SB3 EvalCallback reward curves (per-task, real training/logs/*/evaluations.npz)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({n} runs plotted)")
    for run in runs:
        print(f"  - {run['label']:<28} {run['path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/images/eval_comparison.png")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = load_runs(repo_root)
    if not runs:
        print("No evaluations.npz files found under training/logs/ — train something first.")
        return
    plot(runs, os.path.join(repo_root, args.out))


if __name__ == "__main__":
    main()
