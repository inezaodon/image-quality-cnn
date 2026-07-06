#!/usr/bin/env python
"""
train_quality.py
----------------
Train a CNN to predict a face-image quality score directly from the pixels,
using the scores OFIQ produced as the ground-truth labels.

  inputs  (X) : FFHQ face images           (ffhq_all/00000.png ... 69999.png)
  labels  (Y) : a column from the OFIQ CSV (default: UnifiedQualityScore.native)

This is a supervised *regression* problem: image in -> quality score out.
The model is a fast learned approximation of OFIQ.

Why train on UnifiedQualityScore.native (not .scalar)?
  .scalar is a non-linear (sigmoid-like) re-mapping of the raw measure onto
  0-100. .native is the raw, linear quantity OFIQ actually computes. Regressing
  the linear quantity keeps the target undistorted and the error interpretable,
  which is why we default to it. The label is min-max scaled to [0,1] using the
  TRAIN split's range only (stored in the checkpoint) so the model -> score
  mapping is exact and works for ANY target column, whatever its range.

Anti-overfitting toolbox (all on by default, tune via flags):
  - spatial dropout (Dropout2d) inside the conv stack
  - SE (Squeeze-and-Excitation) channel attention in every conv block
  - weight decay (L2) via AdamW
  - data augmentation (random crop / flip / colour jitter)
  - a deliberately SMALL ResNet (resnet_small) as an alternative to resnet18
  - early stopping on val MAE
  - gradient clipping (--grad-clip 1.0) to prevent training spikes
  - combined MSE + Pearson-correlation loss (--loss combined) to directly
    optimise the correlation metrics the evaluator measures
  - per-epoch overfitting diagnostics + a training-curve plot

Run it on the GPU node via JOB_train_quality_full.sh, or directly:
    module load pytorch/2.9.1
    python train_quality.py --arch resnet_small --target UnifiedQualityScore.native --epochs 40
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# --------------------------------------------------------------------------- #
# Dataset: join each CSV row to its image BY FILENAME (never by row order!)    #
# Targets are min-max scaled to [0,1] with the (lo, hi) computed from TRAIN.   #
# --------------------------------------------------------------------------- #
class FFHQQualityDataset(Dataset):
    def __init__(self, dataframe, root, target_col, transform, lo, hi):
        # dataframe already filtered to rows whose image exists
        self.files = dataframe["Filename"].tolist()
        raw = dataframe[target_col].astype("float32").to_numpy()
        # scale target -> [0,1] using the train-set range, so Sigmoid output matches.
        # (lo, hi) carry a small margin so values don't sit on the saturating ends.
        self.targets = ((raw - lo) / (hi - lo)).tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Filename already carries the "ffhq_all/" prefix; join to root just in case
        path = self.files[idx]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([self.targets[idx]], dtype=torch.float32)
        return img, target


# --------------------------------------------------------------------------- #
# Model 1: a simple CNN from scratch, now regularised to fight overfitting.    #
#   - Dropout2d (spatial dropout) after each conv block: drops whole feature   #
#     maps, which is far more effective on conv layers than ordinary dropout.  #
#   - two Dropout layers in the head.                                          #
# Built from scratch -> no weight download -> runs on the offline node.        #
# --------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention at near-zero parameter cost.
    Lets the network up-weight feature maps that carry quality signal and
    suppress those that carry noise -- especially useful for regression."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


class SimpleCNN(nn.Module):
    def __init__(self, drop2d=0.10, head_drop=0.40):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop2d),     # <-- spatial dropout
                SEBlock(cout),            # <-- channel attention
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # 224 -> 112
            block(32, 64),    # 112 -> 56
            block(64, 128),   # 56  -> 28
            block(128, 256),  # 28  -> 14
            nn.AdaptiveAvgPool2d(1),  # -> 256 x 1 x 1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_drop),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_drop),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # keep output in [0, 1] to match the scaled target
        )

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------- #
# Model 2: the SMALLEST sensible ResNet, built from scratch.                   #
# torchvision's smallest is resnet18 (~11M params); on 70k images with strong  #
# labels it has way more capacity than needed and overfits hard. This custom   #
# ResNet uses ONE residual block per stage and narrow widths (~0.5M params),   #
# plus spatial dropout inside each block. Smaller capacity = less overfitting. #
# No pretrained weights -> works on the offline compute node.                  #
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, drop=0.10):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.se = SEBlock(cout)           # <-- channel attention per block
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        out = self.se(out)                # channel attention before residual add
        out = out + self.short(x)
        return self.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), drop=0.10, head_drop=0.30):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, w0, 3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(w0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
        )
        self.layer1 = BasicBlock(w0, w0, stride=1, drop=drop)  # 56
        self.layer2 = BasicBlock(w0, w1, stride=2, drop=drop)  # 56 -> 28
        self.layer3 = BasicBlock(w1, w2, stride=2, drop=drop)  # 28 -> 14
        self.layer4 = BasicBlock(w2, w3, stride=2, drop=drop)  # 14 -> 7
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_drop),
            nn.Linear(w3, w3 // 2),      # extra layer: more expressive without much overfit
            nn.ReLU(inplace=True),
            nn.Dropout(head_drop),
            nn.Linear(w3 // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x))


def build_model(arch, head_drop=None):
    if arch == "simple":
        return SimpleCNN(head_drop=0.40 if head_drop is None else head_drop)
    if arch == "resnet_small":
        return SmallResNet(head_drop=0.30 if head_drop is None else head_drop)
    if arch == "resnet18":
        # pretrained weights need internet/cache; download on the front-end first.
        m = models.resnet18(weights="IMAGENET1K_V1")
        # add dropout before the final layer so the big pretrained net regularises.
        m.fc = nn.Sequential(
            nn.Dropout(0.40 if head_drop is None else head_drop),
            nn.Linear(m.fc.in_features, 1),
            nn.Sigmoid(),
        )
        return m
    raise ValueError(f"unknown arch: {arch}")


# --------------------------------------------------------------------------- #
# Loss function                                                                #
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """MSE + (1 - Pearson r): optimises both score magnitude AND ranking.

    Why combined?  MSELoss alone minimises prediction error in absolute terms
    but gives no gradient signal for correlation.  The Pearson term directly
    pushes the model to rank images in the right order -- which is exactly what
    the evaluator measures with Spearman rho and Pearson r.

    pearson_w=0.4 means 60% MSE (magnitude) + 40% correlation (ranking).
    """
    def __init__(self, pearson_w=0.4):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson_w = pearson_w
        self.mse_w = 1.0 - pearson_w

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(preds, targets)
        p = preds.squeeze() - preds.squeeze().mean()
        t = targets.squeeze() - targets.squeeze().mean()
        r = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
        return self.mse_w * mse_loss + self.pearson_w * (1.0 - r)


# --------------------------------------------------------------------------- #
# Train / evaluate                                                             #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, span, optimizer=None, grad_clip=0.0):
    """span = (hi - lo): converts the scaled MAE back into native score units.

    Returns (loss, mse, mae):
      - loss = whatever `criterion` computes (e.g. combined MSE+Pearson) -- the
        quantity actually being optimised / fed to the LR scheduler.
      - mse  = plain MSE in scaled [0,1] units, ALWAYS, regardless of criterion.
        Logging this separately matters because when --loss=combined (the
        default), `loss` is NOT pure MSE -- conflating the two mislabels the
        training curves and corrupts MSE-based model selection.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, total_sq_err, total_abs_err, n = 0.0, 0.0, 0.0, 0
    torch.set_grad_enabled(train)
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, targets)
        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        diff = (preds - targets).detach()
        total_sq_err += diff.pow(2).sum().item()
        # MAE back in the ORIGINAL score units, so it's interpretable
        total_abs_err += diff.abs().sum().item() * span
        n += bs
    return total_loss / n, total_sq_err / n, total_abs_err / n


def diagnose_overfitting(log_rows, span, requested_epochs=None, patience=None):
    """Inspect the per-epoch log and print concrete signs of overfitting.

    Shared logic with diagnose_overfitting.py so the same verdict appears both
    at the end of training and when re-analysing an old train_log later.

    Best epoch is selected by val MSE, not val MAE: MAE is the noisier of the
    two metrics epoch-to-epoch, so ranking by it can pick a different (and
    less representative) epoch than the one that's actually lowest on the
    loss curve people are looking at.
    """
    if not log_rows:
        return
    df = pd.DataFrame(log_rows)
    best_i = int(df["val_mse"].idxmin())
    best_epoch = int(df["epoch"].iloc[best_i])
    best_val_mse = float(df["val_mse"].iloc[best_i])
    best_val_mae = float(df["val_mae"].iloc[best_i])
    last = df.iloc[-1]
    n = len(df)
    last_epoch = int(last["epoch"])

    # generalisation gap = how much worse val is than train (native MAE units)
    gap = float(last["val_mae"] - last["train_mae"])
    ratio = float(last["val_mse"] / max(last["train_mse"], 1e-12))
    # epochs of training spent AFTER val stopped improving = wasted/overfitting
    wasted = int(last["epoch"] - best_epoch)
    # did val loss trend UP while train kept falling? (divergence)
    tail = df.iloc[best_i:]
    diverging = bool(len(tail) > 3 and tail["val_mse"].iloc[-1] > tail["val_mse"].iloc[0])

    print("\n---------------- OVERFITTING DIAGNOSIS ----------------")
    if requested_epochs and last_epoch < requested_epochs:
        print(f"  epochs run        : {last_epoch}/{requested_epochs} requested "
              f"-> stopped early (no val-MSE gain for {patience} epochs after epoch {best_epoch})")
    print(f"  best val MSE      : {best_val_mse:.5f} (val MAE {best_val_mae:.2f}) "
          f"at epoch {best_epoch}/{last_epoch}")
    print(f"  final train MAE   : {float(last['train_mae']):.2f}   "
          f"final val MAE: {float(last['val_mae']):.2f}")
    print(f"  generalisation gap: {gap:.2f} score points (val - train MAE)")
    print(f"  val/train MSE ratio: {ratio:.1f}x  (1x = no overfit; big = memorising)")
    print(f"  epochs after best : {wasted}  (training continued with no val gain)")
    flags = []
    if ratio > 3:
        flags.append(f"val MSE is {ratio:.0f}x the train MSE -> the model memorises the train set")
    if gap > 0.25 * span:
        flags.append("train MAE is far below val MAE -> classic train/val divergence")
    if wasted > 5:
        flags.append(f"val peaked at epoch {best_epoch} but training ran {wasted} more epochs -> use early stopping")
    if diverging:
        flags.append("val loss trended UP after the best epoch -> over-training")
    if flags:
        print("  >> SIGNS OF OVERFITTING DETECTED:")
        for f in flags:
            print(f"     - {f}")
    else:
        print("  >> No strong overfitting signals (val tracks train).")
    print("-------------------------------------------------------\n", flush=True)


def plot_curves(log_rows, path, requested_epochs=None, patience=None):
    """Plot the TRAINING curves: train vs val MSE and MAE per epoch.

    Best epoch (the dashed vertical line) is chosen by val MSE, not val MAE --
    MAE is noisy epoch-to-epoch, so a MAE-based marker can land visibly off
    the true minimum of the MSE curve it's drawn on. Both panels use the same
    MSE-based best epoch so the two plots agree with each other.
    """
    import textwrap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(could not import matplotlib for training plot: {e})", flush=True)
        return
    df = pd.DataFrame(log_rows)
    best_i = int(df["val_mse"].idxmin())
    best_epoch = int(df["epoch"].iloc[best_i])
    last_epoch = int(df["epoch"].iloc[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    ax1.plot(df["epoch"], df["train_mse"], "-o", ms=3, label="train MSE")
    ax1.plot(df["epoch"], df["val_mse"], "-o", ms=3, label="val MSE")
    ax1.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best val MSE (ep {best_epoch})")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE (scaled)"); ax1.set_title("Loss curve (true MSE)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(df["epoch"], df["train_mae"], "-o", ms=3, label="train MAE")
    ax2.plot(df["epoch"], df["val_mae"], "-o", ms=3, label="val MAE")
    ax2.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best val MSE (ep {best_epoch})")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("MAE (score points)"); ax2.set_title("Error curve")
    ax2.legend(); ax2.grid(alpha=0.3)

    title = f"Training curves — best epoch selected by val MSE, not val MAE (epoch {best_epoch})"
    if requested_epochs and last_epoch < requested_epochs:
        title += (f"\nRan {last_epoch}/{requested_epochs} requested epochs — stopped early: "
                  f"no val-MSE improvement for {patience} epochs after epoch {best_epoch}.")
    note = (
        "Val MAE/MSE can sit below train's because dropout is ON during training (intentionally "
        "crippling predictions) but OFF during validation — expected, not a labelling error."
    )
    title += "\n" + "\n".join(textwrap.wrap(note, width=100))
    n_lines = title.count("\n") + 1
    fig.tight_layout()
    top = max(0.97 - 0.05 * n_lines, 0.72)
    fig.subplots_adjust(top=top)
    fig.suptitle(title, fontsize=9, y=0.995)
    fig.savefig(path, dpi=120)
    print(f"Training-curve plot saved -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ffhq_all_results.csv")
    ap.add_argument("--root", default=".", help="dir the Filename paths are relative to")
    # default target is now the raw, linear .native score (professor's recommendation)
    ap.add_argument("--target", default="UnifiedQualityScore.native")
    ap.add_argument("--arch", default="resnet_small",
                    choices=["simple", "resnet_small", "resnet18"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 regularisation via AdamW (key anti-overfit lever)")
    ap.add_argument("--head-drop", type=float, default=-1.0,
                    help="override head dropout (>=0); -1 uses the arch default")
    ap.add_argument("--augment", default="strong", choices=["none", "basic", "strong"],
                    help="train-time data augmentation strength")
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs with no val-MAE gain (0=off)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0=all); for quick smoke tests")
    ap.add_argument("--out", default="best_model.pt")
    ap.add_argument("--log", default="train_log.csv")
    ap.add_argument("--curve", default="train_curve.png",
                    help="path for the training-curve PNG ('' to skip)")
    ap.add_argument("--loss", default="combined",
                    choices=["mse", "huber", "combined"],
                    help="loss: mse=MSELoss, huber=SmoothL1Loss, "
                         "combined=MSE+PearsonCorr (directly optimises the "
                         "correlation metric the evaluator measures)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient clipping max norm (0=off); prevents spikes "
                         "in early epochs")
    args = ap.parse_args()

    head_drop = None if args.head_drop < 0 else args.head_drop

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs visible: {torch.cuda.device_count()}", flush=True)

    # ---- load labels, keep only rows whose image actually exists ----------- #
    df = pd.read_csv(args.csv, sep=";")
    df.columns = [c.strip() for c in df.columns]
    if args.target not in df.columns:
        avail = [c for c in df.columns if c.endswith(".native") or c.endswith(".scalar")][:6]
        raise SystemExit(f"target column '{args.target}' not in CSV. Available e.g.: {avail} ...")
    df = df[["Filename", args.target]].dropna()
    exists = df["Filename"].apply(
        lambda p: os.path.exists(p if os.path.isabs(p) else os.path.join(args.root, p))
    )
    df = df[exists].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    print(f"Usable labelled images: {len(df)}", flush=True)

    # ---- reproducible train/val split (by row, after filtering) ------------ #
    idx = np.random.permutation(len(df))
    n_val = int(len(df) * args.val_frac)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}", flush=True)

    # ---- target scaling from the TRAIN split only (no val leakage) --------- #
    y = train_df[args.target].astype("float32").to_numpy()
    y_min, y_max = float(y.min()), float(y.max())
    margin = 0.05 * (y_max - y_min)          # keep targets off the Sigmoid's saturating ends
    lo, hi = y_min - margin, y_max + margin
    span = hi - lo
    print(f"Target '{args.target}': train range [{y_min:.3f}, {y_max:.3f}] "
          f"-> scaled with (lo={lo:.3f}, hi={hi:.3f}); MAE reported in native units.",
          flush=True)

    # Save the exact validation filenames next to the model, so evaluate.py can
    # score the model on precisely the images it never trained on (no leakage).
    split_path = os.path.splitext(args.out)[0] + ".val_files.txt"
    val_df["Filename"].to_csv(split_path, index=False, header=False)
    print(f"Saved val split -> {split_path}", flush=True)

    if args.arch == "resnet18":
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # ---- data augmentation: more variety in training = less overfitting ---- #
    aug = []
    if args.augment == "strong":
        aug = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomRotation(10),
        ]
    elif args.augment == "basic":
        aug = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
        ]
    else:  # none
        aug = [transforms.Resize((args.img_size, args.img_size))]

    train_tf = transforms.Compose(aug + [transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        norm,
    ])

    train_ds = FFHQQualityDataset(train_df, args.root, args.target, train_tf, lo, hi)
    val_ds = FFHQQualityDataset(val_df, args.root, args.target, eval_tf, lo, hi)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- model ------------------------------------------------------------- #
    model = build_model(args.arch, head_drop=head_drop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch} | parameters: {n_params/1e6:.2f}M", flush=True)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:  # combined: MSE + Pearson correlation (default)
        criterion = CombinedLoss(pearson_w=0.4)
    print(f"Loss: {args.loss} | grad_clip: {args.grad_clip}", flush=True)

    # AdamW = Adam with proper (decoupled) weight decay -> the L2 regularisation
    # your professor asked for. Adam still adapts the per-parameter learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # ---- training loop ----------------------------------------------------- #
    # Model selection (checkpointing + early stopping) tracks val MSE, not val
    # MAE: MSE is the smoother of the two metrics epoch-to-epoch, so it's the
    # more reliable "did this actually get better" signal (MAE alone can flip
    # the answer by a few epochs just from noise).
    best_val_mse = float("inf")
    epochs_since_best = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mse, tr_mae = run_epoch(model, train_loader, criterion, device, span, optimizer,
                                            grad_clip=args.grad_clip)
        va_loss, va_mse, va_mae = run_epoch(model, val_loader, criterion, device, span, None)
        scheduler.step(va_loss)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        gap = va_mae - tr_mae
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train MSE {tr_mse:.4f} MAE {tr_mae:5.2f} | "
              f"val MSE {va_mse:.4f} MAE {va_mae:5.2f} | "
              f"{args.loss} loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"gap {gap:+5.2f} | lr {lr_now:.1e} | {dt:.0f}s", flush=True)
        log_rows.append(dict(epoch=epoch, train_loss=tr_loss, train_mse=tr_mse, train_mae=tr_mae,
                             val_loss=va_loss, val_mse=va_mse, val_mae=va_mae, lr=lr_now, seconds=dt))
        pd.DataFrame(log_rows).to_csv(args.log, index=False)

        if va_mse < best_val_mse - 1e-9:
            best_val_mse = va_mse
            epochs_since_best = 0
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model_state": state, "arch": args.arch, "target": args.target,
                        "img_size": args.img_size, "val_mse": va_mse, "val_mae": va_mae,
                        "epoch": epoch, "target_lo": lo, "target_hi": hi, "head_drop": head_drop},
                       args.out)
            print(f"  -> saved {args.out} (best val MSE {va_mse:.5f}, val MAE {va_mae:.2f} "
                  f"in native units)", flush=True)
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"  -> early stop: no val-MSE gain for {args.patience} epochs "
                      f"(best {best_val_mse:.5f}). Stopping at epoch {epoch}/{args.epochs} requested.",
                      flush=True)
                break

    print(f"\nDone. Best val MSE: {best_val_mse:.5f} (native-unit MAE at that epoch is in the log).",
          flush=True)

    # ---- training-set plot + overfitting diagnostics ----------------------- #
    if args.curve:
        plot_curves(log_rows, args.curve, requested_epochs=args.epochs, patience=args.patience)
    diagnose_overfitting(log_rows, span, requested_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
