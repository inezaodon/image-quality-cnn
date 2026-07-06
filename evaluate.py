#!/usr/bin/env python
"""
evaluate.py
-----------
Answer the question: "did the trained model actually work?"

It runs a saved model on the held-out validation images (the ones it never
trained on) and reports the numbers that actually tell you whether it learned:

  MAE / RMSE   - how far off the predicted score is, in native score units
  Pearson r    - do predicted and true scores move together (linear)?
  Spearman rho - does the model RANK images by quality correctly?  <-- the big one
  R^2          - fraction of score variance the model explains
  baseline MAE - error you'd get by always guessing the mean (the bar to beat)

It also prints a handful of example predictions so you can eyeball them, and
(by default) saves BOTH a validation scatter AND a training-set scatter — put
side by side, a tight train cloud next to a loose val cloud is overfitting you
can literally see.

Usage:
    module load pytorch/2.9.1
    python evaluate.py --model best_model.pt --plot eval_scatter.png
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_quality import FFHQQualityDataset, build_model


def run_model(model, df, root, target_col, tf, lo, hi, device, batch_size, workers):
    """Run the model over a dataframe; return (preds, trues) in NATIVE units."""
    ds = FFHQQualityDataset(df, root, target_col, tf, lo, hi)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    span = hi - lo
    preds, trues = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs.to(device)).cpu().numpy().ravel()
            preds.append(out)
            trues.append(targets.numpy().ravel())
    # undo the [0,1] scaling -> back to the target's native units
    preds = np.concatenate(preds) * span + lo
    trues = np.concatenate(trues) * span + lo
    return preds, trues


def metrics(preds, trues):
    err = preds - trues
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    pearson = np.corrcoef(preds, trues)[0, 1]
    rp = pd.Series(preds).rank().values
    rt = pd.Series(trues).rank().values
    spearman = np.corrcoef(rp, rt)[0, 1]  # Spearman = Pearson on the ranks
    ss_res = (err ** 2).sum()
    ss_tot = ((trues - trues.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    baseline_mae = np.abs(trues - trues.mean()).mean()
    return dict(mae=mae, rmse=rmse, pearson=pearson, spearman=spearman,
                r2=r2, baseline_mae=baseline_mae)


def scatter(trues, preds, m, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(could not make plot {path}: {e})")
        return
    lo, hi = float(min(trues.min(), preds.min())), float(max(trues.max(), preds.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, s=6, alpha=0.3)
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel("True OFIQ score (native)"); plt.ylabel("Predicted score (native)")
    plt.title(f"{title}\nr={m['pearson']:.3f}  rho={m['spearman']:.3f}  MAE={m['mae']:.2f}")
    plt.tight_layout(); plt.savefig(path, dpi=120)
    print(f"Scatter plot saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_model.pt")
    ap.add_argument("--csv", default="ffhq_all_results.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--plot", default="", help="path to save the VALIDATION scatter PNG")
    ap.add_argument("--train-plot", default="auto",
                    help="path for the TRAINING-set scatter PNG; 'auto' = <plot>_train.png, ''=skip")
    ap.add_argument("--train-sample", type=int, default=7000,
                    help="how many training images to score for the train scatter (speed cap)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "simple")
    target_col = ckpt.get("target", "UnifiedQualityScore.native")
    img_size = ckpt.get("img_size", 224)
    # scaling: new checkpoints carry the train-set (lo, hi); old ones used /100.
    lo = ckpt.get("target_lo", 0.0)
    hi = ckpt.get("target_hi", 100.0)
    print(f"Loaded {args.model}: arch={arch} target={target_col} img_size={img_size} "
          f"scale=({lo:.3f},{hi:.3f}) (trained to epoch {ckpt.get('epoch','?')})", flush=True)

    # ---- labels --------------------------------------------------------- #
    df = pd.read_csv(args.csv, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df[["Filename", target_col]].dropna()

    # ---- split into the exact val set + the training remainder ---------- #
    split_path = os.path.splitext(args.model)[0] + ".val_files.txt"
    if os.path.exists(split_path):
        val_files = set(pd.read_csv(split_path, header=None)[0].tolist())
        exists = df["Filename"].apply(
            lambda p: os.path.exists(p if os.path.isabs(p) else os.path.join(args.root, p)))
        df = df[exists].reset_index(drop=True)
        val_df = df[df["Filename"].isin(val_files)].reset_index(drop=True)
        train_df = df[~df["Filename"].isin(val_files)].reset_index(drop=True)
        print(f"Using saved val split ({len(val_df)} val / {len(train_df)} train) -> clean, no leakage.",
              flush=True)
    else:
        print("WARNING: no saved val split found; reconstructing with seed=42 (approximate).", flush=True)
        exists = df["Filename"].apply(
            lambda p: os.path.exists(p if os.path.isabs(p) else os.path.join(args.root, p)))
        df = df[exists].reset_index(drop=True)
        np.random.seed(42)
        idx = np.random.permutation(len(df))
        n_val = int(len(df) * 0.1)
        val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
        train_df = df.iloc[idx[n_val:]].reset_index(drop=True)

    # ---- model ---------------------------------------------------------- #
    if arch == "resnet18":
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(), norm])

    model = build_model(arch, head_drop=ckpt.get("head_drop")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- evaluate on held-out validation ------------------------------- #
    preds, trues = run_model(model, val_df, args.root, target_col, tf, lo, hi,
                             device, args.batch_size, args.workers)
    m = metrics(preds, trues)

    print("\n================ EVALUATION (held-out val) ================")
    print(f"  images evaluated : {len(trues)}")
    print(f"  MAE              : {m['mae']:6.2f}   (lower better, native units)")
    print(f"  baseline MAE     : {m['baseline_mae']:6.2f}   (always-guess-mean; beat this!)")
    print(f"  RMSE             : {m['rmse']:6.2f}")
    print(f"  Pearson r        : {m['pearson']:6.3f}        (1.0 = perfect linear)")
    print(f"  Spearman rho     : {m['spearman']:6.3f}        (ranking quality; the key one)")
    print(f"  R^2              : {m['r2']:6.3f}        (1.0 = explains all variance)")
    print("===========================================================\n")

    # ---- a few concrete examples ---------------------------------------- #
    n = min(args.examples, len(trues))
    show = np.linspace(0, len(trues) - 1, n).astype(int)
    print(f"{'Filename':<28}{'TRUE':>8}{'PRED':>8}{'err':>8}")
    for i in show:
        print(f"{val_df['Filename'].iloc[i]:<28}{trues[i]:>8.1f}{preds[i]:>8.1f}{preds[i]-trues[i]:>+8.1f}")

    # ---- validation scatter --------------------------------------------- #
    if args.plot:
        scatter(trues, preds, m, args.plot, "Validation (held-out)")

    # ---- training-set scatter (so overfitting is visible) --------------- #
    train_plot = args.train_plot
    if train_plot == "auto":
        train_plot = (os.path.splitext(args.plot)[0] + "_train.png") if args.plot else ""
    if train_plot:
        sample = train_df
        if args.train_sample and len(train_df) > args.train_sample:
            sample = train_df.sample(args.train_sample, random_state=0).reset_index(drop=True)
        tp, tt = run_model(model, sample, args.root, target_col, tf, lo, hi,
                           device, args.batch_size, args.workers)
        tm = metrics(tp, tt)
        print("\n---- TRAINING-set fit (for comparison; expect it to look better) ----")
        print(f"  train MAE {tm['mae']:.2f}  vs  val MAE {m['mae']:.2f}   "
              f"(big gap => overfitting)")
        print(f"  train r   {tm['pearson']:.3f}  vs  val r   {m['pearson']:.3f}")
        scatter(tt, tp, tm, train_plot, "Training set (model has seen these)")


if __name__ == "__main__":
    main()
