#!/usr/bin/env python
"""
diagnose_overfitting.py
-----------------------
Read a training log (the per-epoch CSV that train_quality.py writes) and print
concrete, quantified signs of overfitting -- the kind of evidence you can put
straight into a report.

It flags, per run:
  - the generalisation gap (val MAE - train MAE) and how it grows
  - the val/train MSE ratio (how much harder unseen data is than seen data)
  - the epoch val performance actually peaked at (and how many epochs were
    then wasted training past it)
  - whether the val loss started trending UP while train kept falling

Usage:
    python diagnose_overfitting.py train_log_full.csv
    python diagnose_overfitting.py train_log.csv train_log_resnet.csv train_log_full.csv
"""

import csv
import sys

# stdlib only -- runs on the login node without loading the pytorch module.


def analyse(path):
    need = {"epoch", "train_mse", "train_mae", "val_mse", "val_mae"}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[{path}] empty -- skipping.")
        return
    if not need.issubset(rows[0].keys()):
        print(f"[{path}] missing columns {need - set(rows[0].keys())} -- skipping.")
        return

    def col(name):
        return [float(r[name]) for r in rows]

    epoch, train_mse, train_mae = col("epoch"), col("train_mse"), col("train_mae")
    val_mse, val_mae = col("val_mse"), col("val_mae")

    # Rank by val MSE, not val MAE: MAE is noisier epoch-to-epoch, so picking
    # "best" by MAE can land on a different (less representative) epoch than
    # the true minimum of the MSE curve.
    best_i = min(range(len(val_mse)), key=lambda i: val_mse[i])
    best_epoch = int(epoch[best_i])
    best_val = val_mae[best_i]
    best_val_mse = val_mse[best_i]
    final_epoch = int(epoch[-1])

    gap = val_mae[-1] - train_mae[-1]
    rel_gap = gap / max(val_mae[-1], 1e-9)                    # gap as fraction of val MAE
    ratio = val_mse[-1] / max(train_mse[-1], 1e-12)
    wasted = final_epoch - best_epoch
    diverging = (len(val_mse) - best_i) > 3 and val_mse[-1] > val_mse[best_i]
    # how much val improved AFTER the early plateau (epoch <=5)
    early_cut = min(5, final_epoch)
    early_min = min(val_mae[i] for i in range(len(epoch)) if epoch[i] <= early_cut)
    val_gain_after_early = early_min - best_val
    last = {"train_mae": train_mae[-1], "val_mae": val_mae[-1]}

    print(f"\n================ {path} ================")
    print(f"  epochs run         : {final_epoch}")
    print(f"  best val MSE       : {best_val_mse:.5f}  (val MAE {best_val:.3f})  at epoch {best_epoch}")
    print(f"  final train MAE    : {float(last['train_mae']):.3f}")
    print(f"  final val   MAE    : {float(last['val_mae']):.3f}")
    print(f"  generalisation gap : {gap:.3f}  ({rel_gap*100:.0f}% of val MAE)")
    print(f"  val/train MSE ratio: {ratio:.1f}x")
    print(f"  epochs after best  : {wasted}")
    print(f"  val MAE gained after epoch 5 : {val_gain_after_early:.3f}")

    flags = []
    if ratio > 3:
        flags.append(f"val MSE is {ratio:.0f}x the train MSE -> the model is memorising the training set")
    if rel_gap > 0.5:
        flags.append(f"val MAE is {rel_gap*100:.0f}% larger than train MAE -> large train/val divergence")
    if wasted > 5:
        flags.append(f"val peaked at epoch {best_epoch} but training ran {wasted} more epochs -> "
                     f"early stopping would save time and the over-trained weights")
    if abs(val_gain_after_early) < 0.1 * best_val:
        flags.append("val MAE barely improved after epoch 5 while train kept dropping -> "
                     "extra epochs only fit the training set")
    if diverging:
        flags.append("val loss trended UP after its best epoch -> over-training")

    if flags:
        print("  >> SIGNS OF OVERFITTING:")
        for f in flags:
            print(f"     - {f}")
        print("  >> Suggested fixes: smaller model (try --arch resnet_small), "
              "weight decay, spatial dropout, stronger augmentation, early stopping.")
    else:
        print("  >> No strong overfitting signals: val tracks train reasonably well.")


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(1)
    for p in paths:
        analyse(p)
    print()


if __name__ == "__main__":
    main()
