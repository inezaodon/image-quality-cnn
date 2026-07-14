#!/usr/bin/env python3
"""Regenerate training-curve PNG from an existing train_log CSV."""
import argparse
import pandas as pd
from train_quality import plot_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="train_log_full.csv")
    ap.add_argument("--out", default="train_curve_full.png")
    args = ap.parse_args()
    rows = pd.read_csv(args.log).to_dict("records")
    plot_curves(rows, args.out)


if __name__ == "__main__":
    main()
