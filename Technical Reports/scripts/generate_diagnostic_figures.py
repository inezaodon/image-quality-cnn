#!/usr/bin/env python3
"""Diagnostic figures for the regression-to-the-mean problem.

Consumes the per-image predictions CSV that the patched evaluate.py now writes
(<model>_val_preds.csv with columns Filename, true, pred). Produces:

  fig_binned_residuals.png       mean residual +/- std per true-score bin;
                                 tail compression shows up as positive bias at
                                 low scores and negative bias at high scores.
  fig_calibration.png            least-squares fit of pred vs true against the
                                 ideal y = x; slope < 1 quantifies the
                                 compression a GAN critic would feel.
  fig_component_correlation.png  correlation of the residual with every OFIQ
                                 .native component measure -- which quality
                                 factors the model systematically misses.

If the predictions CSV does not exist yet, the script prints how to create it
and exits cleanly (evaluate.py must run on a machine that has the images).
"""
from __future__ import annotations

import os
import sys
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

PREDS_CSV = ROOT / "best_model_full_val_preds.csv"
OFIQ_CSV = ROOT / "ffhq_all_results.csv"


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


def fig_binned_residuals(df):
    true, resid = df["true"].to_numpy(), (df["pred"] - df["true"]).to_numpy()
    bins = np.linspace(true.min(), true.max(), 15)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, stds, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (true >= lo) & (true < hi)
        counts.append(int(mask.sum()))
        if mask.sum() >= 5:
            means.append(resid[mask].mean())
            stds.append(resid[mask].std())
        else:
            means.append(np.nan)
            stds.append(np.nan)
    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color="#888888", lw=1)
    ax.errorbar(centers, means, yerr=stds, fmt="o-", ms=4, capsize=3,
                color="#1f4e79", ecolor="#9dbcd4", label="mean residual ± std")
    for c, n, m in zip(centers, counts, means):
        if not np.isnan(m):
            ax.annotate(f"n={n}", (c, m), textcoords="offset points",
                        xytext=(0, 10), fontsize=6, ha="center", color="#666666")
    ax.set_xlabel("True OFIQ score (native units)")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Binned residuals — positive at low scores / negative at high scores\n= regression to the mean")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_binned_residuals.png")
    plt.close(fig)


def fig_calibration(df):
    true, pred = df["true"].to_numpy(), df["pred"].to_numpy()
    slope, intercept = np.polyfit(true, pred, 1)
    lo, hi = true.min(), true.max()
    xs = np.array([lo, hi])

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(true, pred, s=5, alpha=0.2, color="#1f4e79")
    ax.plot(xs, xs, "r--", lw=1.2, label="ideal y = x (slope 1.0)")
    ax.plot(xs, slope * xs + intercept, "-", lw=1.8, color="#c45c26",
            label=f"fitted: slope {slope:.2f}, intercept {intercept:.1f}")
    ax.set_xlabel("True OFIQ score (native units)")
    ax.set_ylabel("Predicted score (native units)")
    ax.set_title("Calibration — slope < 1 means the prediction range is compressed")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.png")
    plt.close(fig)
    print(f"  calibration slope = {slope:.3f} (1.0 = perfectly calibrated)")


def fig_component_correlation(df, top_k=15):
    if not OFIQ_CSV.exists():
        print(f"  (skipping component correlation: {OFIQ_CSV} not found)")
        return
    ofiq = pd.read_csv(OFIQ_CSV, sep=";")
    ofiq.columns = [c.strip() for c in ofiq.columns]
    comps = [c for c in ofiq.columns
             if c.endswith(".native") and c != "UnifiedQualityScore.native"]
    merged = df.merge(ofiq[["Filename"] + comps], on="Filename", how="inner")
    resid = (merged["pred"] - merged["true"]).to_numpy()

    corrs = {}
    for c in comps:
        vals = merged[c].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() > 100 and np.nanstd(vals[mask]) > 0:
            corrs[c.replace(".native", "")] = np.corrcoef(vals[mask], resid[mask])[0, 1]
    if not corrs:
        print("  (no component columns usable for correlation)")
        return
    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    names = [k for k, _ in ranked][::-1]
    vals = [v for _, v in ranked][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = ["#c45c26" if v > 0 else "#1f4e79" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0, color="#888888", lw=1)
    ax.set_xlabel("Correlation of OFIQ component with residual (pred − true)")
    ax.set_title(f"Top {len(names)} component measures correlated with the model's errors\n"
                 "(large |corr| = quality factor the model fails to capture)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_component_correlation.png")
    plt.close(fig)


def main():
    if not PREDS_CSV.exists():
        print(f"Predictions CSV not found: {PREDS_CSV}\n"
              "Generate it first (needs the images, e.g. on the cluster):\n"
              "    python evaluate.py --model best_model_full.pt --plot eval_scatter_full.png\n"
              "The patched evaluate.py writes best_model_full_val_preds.csv automatically.")
        sys.exit(0)
    style()
    df = pd.read_csv(PREDS_CSV)
    print(f"Loaded {len(df)} per-image predictions from {PREDS_CSV.name}")
    fig_binned_residuals(df)
    fig_calibration(df)
    fig_component_correlation(df)
    print(f"Diagnostic figures written to {OUT}")


if __name__ == "__main__":
    main()
