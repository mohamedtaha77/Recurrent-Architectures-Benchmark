# -*- coding: utf-8 -*-
"""
Compare the original WikiText-2 benchmark checkpoints against the
WikiText-103 showcase checkpoints: same 3 architectures, same 5-length
sweep, different dataset scale and training setup. See the main README's
"WikiText-103 showcase" section for exactly what changed between them.

Usage (from this folder, after both sweeps have been trained):
    python compare_wt103.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARCHS = ["rnn", "gru", "lstm"]
LENGTHS = [10, 25, 50, 100, 200]
COLORS = {"rnn": "#1f77b4", "gru": "#ff7f0e", "lstm": "#2ca02c"}


def load(suffix):
    out = {}
    for arch in ARCHS:
        for L in LENGTHS:
            path = os.path.join(RESULTS_DIR, f"{arch}_{L}{suffix}.json")
            with open(path, encoding="utf-8") as f:
                out[(arch, L)] = json.load(f)
    return out


def plot_curves(new):
    """Training curves for the showcase runs only (the old runs' curves are
    already in curves.png from compare.py). Early stopping means each run
    has a different epoch count, so each line simply runs however long
    that particular run did."""
    fig, axes = plt.subplots(len(LENGTHS), 2, figsize=(11, 3.6 * len(LENGTHS)))
    for row, L in enumerate(LENGTHS):
        for arch in ARCHS:
            h = new[(arch, L)]["history"]
            epochs = [e["epoch"] for e in h]
            axes[row][0].plot(epochs, [e["train_loss"] for e in h], color=COLORS[arch], linestyle="--", alpha=0.6)
            axes[row][0].plot(epochs, [e["val_loss"] for e in h], color=COLORS[arch], label=arch.upper())
            axes[row][1].plot(epochs, [e["train_acc"] for e in h], color=COLORS[arch], linestyle="--", alpha=0.6)
            axes[row][1].plot(epochs, [e["val_acc"] for e in h], color=COLORS[arch], label=arch.upper())
        axes[row][0].set_title(f"seq_len={L}: loss (solid=val, dashed=train)")
        axes[row][0].set_xlabel("epoch"); axes[row][0].set_ylabel("cross-entropy loss")
        axes[row][0].legend(fontsize=8)
        axes[row][1].set_title(f"seq_len={L}: next-word accuracy")
        axes[row][1].set_xlabel("epoch"); axes[row][1].set_ylabel("accuracy")
        axes[row][1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "wt103_curves.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot_sweep_comparison(old, new):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for arch in ARCHS:
        old_ppl = [old[(arch, L)]["test_perplexity"] for L in LENGTHS]
        new_ppl = [new[(arch, L)]["test_perplexity"] for L in LENGTHS]
        ax.plot(LENGTHS, old_ppl, marker="o", linestyle="--", color=COLORS[arch], alpha=0.5,
                label=f"{arch.upper()} — WikiText-2 (old)")
        ax.plot(LENGTHS, new_ppl, marker="o", linestyle="-", color=COLORS[arch], linewidth=2.4,
                label=f"{arch.upper()} — WikiText-103 (new)")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("test perplexity (lower is better)")
    ax.set_title("Old (WikiText-2) vs new (WikiText-103) perplexity, by context length")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "wt103_sweep_comparison.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def plot_improvement_bars(old, new):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.25
    x = range(len(LENGTHS))
    for i, arch in enumerate(ARCHS):
        improvements = []
        for L in LENGTHS:
            o = old[(arch, L)]["test_perplexity"]
            n = new[(arch, L)]["test_perplexity"]
            improvements.append(100 * (o - n) / o)
        ax.bar([xi + i * width for xi in x], improvements, width=width, color=COLORS[arch], label=arch.upper())
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("perplexity improvement (%)")
    ax.set_title("Showcase model's improvement over the original benchmark, by length")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "wt103_improvement_bars.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def comparison_table(old, new):
    rows = ["| Seq len | RNN: old → new | GRU: old → new | LSTM: old → new |",
            "|---|---|---|---|"]
    for L in LENGTHS:
        cells = []
        for arch in ARCHS:
            o = old[(arch, L)]["test_perplexity"]
            n = new[(arch, L)]["test_perplexity"]
            pct = 100 * (o - n) / o
            cells.append(f"{o:.1f} → **{n:.1f}** ({pct:+.0f}%)")
        rows.append(f"| {L} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(rows)


def main():
    old = load("")
    new = load("_wt103")
    plot_curves(new)
    plot_sweep_comparison(old, new)
    plot_improvement_bars(old, new)
    print()
    print(comparison_table(old, new))
    print()
    old_params = old[("rnn", 10)]["n_params"], old[("gru", 10)]["n_params"], old[("lstm", 10)]["n_params"]
    new_params = new[("rnn", 10)]["n_params"], new[("gru", 10)]["n_params"], new[("lstm", 10)]["n_params"]
    old_vocab = old[("rnn", 10)]["vocab_size"]
    new_vocab = new[("rnn", 10)]["vocab_size"]
    print(f"old params (rnn/gru/lstm): {old_params[0]:,} / {old_params[1]:,} / {old_params[2]:,}")
    print(f"new params (rnn/gru/lstm): {new_params[0]:,} / {new_params[1]:,} / {new_params[2]:,}")
    print(f"old vocab: {old_vocab:,}   new vocab: {new_vocab:,}")


if __name__ == "__main__":
    main()
