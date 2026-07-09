#!/usr/bin/env python3
"""Generate report figures from real training / label data (no AI imagery)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LOG = ROOT / "train_log_full.csv"
CSV = ROOT / "ffhq_all_results.csv"
ARCH_LOG = ROOT / "archive_epoch40_dropout_on_log" / "train_log_full.csv"


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def fig_train_val_mae():
    df = pd.read_csv(LOG)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["epoch"], df["train_mae"], "o-", color="#1f4e79", lw=1.6, ms=3.5, label="Train MAE (dropout off)")
    ax.plot(df["epoch"], df["val_mae"], "s-", color="#c45c26", lw=1.6, ms=3.5, label="Val MAE")
    best = int(df["val_mse"].idxmin())
    ax.axvline(df.loc[best, "epoch"], color="#666666", ls="--", lw=1.0, label=f"Lowest val MSE (epoch {int(df.loc[best, 'epoch'])}, ref.)")
    ax.axvline(40, color="#2e7d32", ls=":", lw=1.2, label="Saved checkpoint (epoch 40)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (native OFIQ units)")
    ax.set_title("Training progression — MAE (clean logging)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(1, 40)
    fig.tight_layout()
    fig.savefig(OUT / "fig_train_val_mae.png")
    plt.close(fig)


def fig_train_val_mse():
    df = pd.read_csv(LOG)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["epoch"], df["train_mse"], "o-", color="#1f4e79", lw=1.6, ms=3.5, label="Train MSE (scaled)")
    ax.plot(df["epoch"], df["val_mse"], "s-", color="#c45c26", lw=1.6, ms=3.5, label="Val MSE (scaled)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (min–max scaled target space)")
    ax.set_title("Training progression — MSE")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_train_val_mse.png")
    plt.close(fig)


def fig_logging_artifact():
    """Compare inverted (dropout-on) vs clean train/val MAE gap."""
    clean = pd.read_csv(LOG)
    old = pd.read_csv(ARCH_LOG) if ARCH_LOG.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    axes[0].plot(clean["epoch"], clean["train_mae"], color="#1f4e79", label="Train")
    axes[0].plot(clean["epoch"], clean["val_mae"], color="#c45c26", label="Val")
    axes[0].set_title("After fix: dropout-off train logging")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend(fontsize=8)
    if old is not None:
        axes[1].plot(old["epoch"], old["train_mae"], color="#1f4e79", label="Train (dropout on)")
        axes[1].plot(old["epoch"], old["val_mae"], color="#c45c26", label="Val")
        axes[1].set_title("Before fix: dropout-on train logging")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Archive log unavailable", ha="center", va="center")
    fig.suptitle("Measurement artifact vs true train/val relationship", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_logging_artifact.png", bbox_inches="tight")
    plt.close(fig)


def fig_overfit_comparison():
    labels = ["ResNet18\n(~11M params)", "SmallResNet\n(~0.33M params)"]
    train = [1.7, 1.25]
    val = [8.7, 1.32]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - w / 2, train, w, label="Train MAE", color="#1f4e79")
    ax.bar(x + w / 2, val, w, label="Val MAE", color="#c45c26")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (native OFIQ units)")
    ax.set_title("Overfitting comparison: capacity vs generalization")
    ax.legend()
    for i, (t, v) in enumerate(zip(train, val)):
        ax.text(i - w / 2, t + 0.15, f"{t:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_overfit_comparison.png")
    plt.close(fig)


def fig_label_histogram():
    df = pd.read_csv(CSV, sep=";")
    col = "UnifiedQualityScore.native"
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(df[col], bins=50, color="#1f4e79", edgecolor="white", linewidth=0.4)
    ax.axvline(df[col].mean(), color="#c45c26", ls="--", lw=1.4, label=f"Mean = {df[col].mean():.2f}")
    ax.set_xlabel("UnifiedQualityScore.native")
    ax.set_ylabel("Count (images)")
    ax.set_title("Ground-truth quality label distribution (FFHQ, N=70,000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_label_histogram.png")
    plt.close(fig)


def fig_metrics_summary():
    # Two panels: error metrics carry native score units, correlation metrics
    # are unitless on [0, 1]. Mixing them on one axis visually understates the
    # correlation values (0.87 next to a 2.68 baseline looks small).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    err_names = ["MAE", "RMSE"]
    err_vals = [1.32, 1.67]
    bars = ax1.bar(err_names, err_vals, color="#1f4e79", edgecolor="#333333", linewidth=0.5, width=0.5)
    ax1.axhline(2.68, color="#c45c26", ls="--", lw=1.2, label="Baseline MAE = 2.68 (guess the mean)")
    for b, v in zip(bars, err_vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax1.set_ylabel("Native score units (lower is better)")
    ax1.set_title("Error metrics")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0, 3.1)

    corr_names = ["Pearson r", "Spearman ρ", "R²"]
    corr_vals = [0.866, 0.867, 0.738]
    bars = ax2.bar(corr_names, corr_vals, color="#2e7d32", edgecolor="#333333", linewidth=0.5, width=0.5)
    ax2.axhline(1.0, color="#888888", ls=":", lw=1.0, label="Perfect = 1.0")
    for b, v in zip(bars, corr_vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("Unitless (higher is better)")
    ax2.set_title("Correlation / agreement metrics")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(0, 1.1)

    fig.suptitle("Held-out evaluation (7,000 images, checkpoint epoch 40)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_metrics_summary.png")
    plt.close(fig)


def fig_architecture_schematic():
    """Simple block diagram of SmallResNet drawn with matplotlib patches."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, 1.6, 1.6, "Input\n224×224×3"),
        (2.1, 1.2, 1.6, 1.6, "Stem\n16 ch"),
        (4.0, 1.2, 1.6, 1.6, "Res×1\n16"),
        (5.9, 1.2, 1.6, 1.6, "Res×1\n32"),
        (7.8, 1.2, 1.6, 1.6, "Res×1\n64"),
        (9.7, 1.2, 1.6, 1.6, "Res×1\n128"),
        (11.6, 1.2, 2.0, 1.6, "GAP+Head\n→ score"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor="#e8eef5",
                edgecolor="#1f4e79",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
    ax.set_title("SmallResNet forward path (~0.33M parameters)", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture_schematic.png")
    plt.close(fig)


def fig_lr_schedule():
    df = pd.read_csv(LOG)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.step(df["epoch"], df["lr"], where="post", color="#1f4e79", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("ReduceLROnPlateau schedule (AdamW)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_lr_schedule.png")
    plt.close(fig)


def main():
    style()
    fig_train_val_mae()
    fig_train_val_mse()
    fig_logging_artifact()
    fig_overfit_comparison()
    fig_label_histogram()
    fig_metrics_summary()
    fig_architecture_schematic()
    fig_lr_schedule()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("fig_*.png")):
        print(" ", p.name, f"{p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
