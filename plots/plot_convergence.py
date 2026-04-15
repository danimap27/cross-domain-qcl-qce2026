"""
plot_convergence.py — Figure 3: Task A loss convergence by initialization strategy.

Reads history.json files and plots the cross-entropy loss during Task A training
for three initialization strategies (scratch, synthetic_gaussian, mobilenetv2)
using the TTN ansatz under ideal simulation. Produces figure3_convergence.png.

Usage:
    python plots/plot_convergence.py
    python plots/plot_convergence.py --results-dir ./results --out paper/figure3_convergence.png
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


SOURCE_CONFIG = {
    "scratch":            ("Random Init (Scratch)",    "#d62828", "--",  2.0, "s"),
    "synthetic_gaussian": ("QTL Synthetic Gaussian",   "#003049",  "-",  2.5, "o"),
    "mobilenetv2":        ("QTL MobileNet-V2",         "#2a9d8f",  "-",  2.0, "^"),
}


def load_histories(results_dir: str, ansatz: str, noise_model: str) -> dict:
    """
    Load per-epoch loss_a for each source strategy.

    Returns {source: list of lists (one per seed)}.
    """
    pattern = os.path.join(results_dir, "*", "history.json")
    data: dict[str, list] = {s: [] for s in SOURCE_CONFIG}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                h = json.load(f)
        except Exception:
            continue
        if h.get("ansatz") != ansatz:
            continue
        if h.get("noise_model") != noise_model:
            continue
        src = h.get("source")
        if src in data and h.get("loss_a"):
            data[src].append(h["loss_a"])
    return data


def plot_convergence(results_dir: str, out_path: str, ansatz: str = "ttn",
                     noise_model: str = "ideal"):
    data = load_histories(results_dir, ansatz=ansatz, noise_model=noise_model)
    n_epochs = max(
        (len(v[0]) for v in data.values() if v), default=10
    )
    epochs = list(range(1, n_epochs + 1))

    fig, ax = plt.subplots(figsize=(7, 5))

    for source, (label, color, ls, lw, marker) in SOURCE_CONFIG.items():
        runs = data.get(source, [])
        if not runs:
            continue
        arr = np.array(runs)          # (n_seeds, n_epochs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        ax.plot(epochs, mean, marker=marker, label=label,
                color=color, linestyle=ls, linewidth=lw, markersize=5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ansatz_label = {"strongly_entangling": "SEL", "basic_entangler": "Basic", "ttn": "TTN"}.get(ansatz, ansatz)
    noise_tag = "Ideal simulation" if noise_model == "ideal" else "IBM Heron r2 noise"
    ax.set_xlabel("Task A Training Epoch", fontsize=12)
    ax.set_ylabel(r"Cross-Entropy Loss $\downarrow$", fontsize=12)
    ax.set_title(f"Task A Convergence by Initialization ({ansatz_label}, {noise_tag})", fontsize=13)
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--out", default="paper/figure3_convergence.png")
    parser.add_argument("--ansatz", default="ttn")
    parser.add_argument("--noise", default="ideal",
                        choices=["ideal", "ibm_heron_r2"])
    args = parser.parse_args()
    plot_convergence(args.results_dir, args.out, args.ansatz, args.noise)


if __name__ == "__main__":
    main()
