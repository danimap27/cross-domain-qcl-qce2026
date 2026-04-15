"""
plot_forgetting_curves.py — Figure 2: Task A accuracy during Task B training.

Reads history.json files from results/ and plots the forgetting curve
for each ansatz (SEL, Basic Entangler, TTN) under ideal simulation,
scratch initialization. Produces figure2_ansatz_decay.png.

Usage:
    python plots/plot_forgetting_curves.py
    python plots/plot_forgetting_curves.py --results-dir ./results --out paper/figure2_ansatz_decay.png
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


ANSATZ_CONFIG = {
    "strongly_entangling": ("SEL",          "#e63946", "o",  2.5),
    "basic_entangler":     ("Basic Entangler","#f4a261", "s",  2.0),
    "ttn":                 ("TTN",           "#457b9d", "^",  2.5),
}


def load_histories(results_dir: str, noise_model: str, source: str) -> dict:
    """
    Load per-epoch forgetting_history for each ansatz.

    Returns {ansatz: list of lists (one per seed)}.
    """
    pattern = os.path.join(results_dir, "*", "history.json")
    data: dict[str, list] = {a: [] for a in ANSATZ_CONFIG}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                h = json.load(f)
        except Exception:
            continue
        if h.get("noise_model") != noise_model:
            continue
        if h.get("source") != source:
            continue
        ans = h.get("ansatz")
        if ans in data and h.get("forgetting_history"):
            data[ans].append(h["forgetting_history"])
    return data


def plot_forgetting_curves(results_dir: str, out_path: str, noise_model: str = "ideal"):
    data = load_histories(results_dir, noise_model=noise_model, source="scratch")
    n_epochs = max(
        (len(v[0]) for v in data.values() if v), default=10
    )
    epochs = list(range(1, n_epochs + 1))

    fig, ax = plt.subplots(figsize=(7, 5))

    for ansatz, (label, color, marker, lw) in ANSATZ_CONFIG.items():
        runs = data.get(ansatz, [])
        if not runs:
            continue
        arr = np.array(runs)          # (n_seeds, n_epochs)
        mean = arr.mean(axis=0) * 100
        std = arr.std(axis=0) * 100

        ax.plot(epochs, mean, marker=marker, label=label,
                color=color, linewidth=lw, markersize=5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    noise_tag = "Ideal simulation" if noise_model == "ideal" else "IBM Heron r2 noise"
    ax.set_xlabel("Task B Training Epoch", fontsize=12)
    ax.set_ylabel(r"Task A Accuracy (\%) $\downarrow$", fontsize=12)
    ax.set_title(f"Task A Accuracy During Task B Training ({noise_tag})", fontsize=13)
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.set_ylim(0, 105)
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
    parser.add_argument("--out", default="paper/figure2_ansatz_decay.png")
    parser.add_argument("--noise", default="ideal",
                        choices=["ideal", "ibm_heron_r2"])
    args = parser.parse_args()
    plot_forgetting_curves(args.results_dir, args.out, args.noise)


if __name__ == "__main__":
    main()
