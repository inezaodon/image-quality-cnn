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
  - early stopping on val MSE (the smoother metric; see the training loop)
  - gradient clipping (--grad-clip 1.0) to prevent training spikes
  - combined MSE + Pearson-correlation loss (--loss combined) to directly
    optimise the correlation metrics the evaluator measures
  - optional inverse-frequency tail weighting (--tail-weight 0.5) so rare
    extreme-quality images get proportionally more gradient -- the fix for
    regression-to-the-mean at the score extremes
  - optional linear output head (--head linear) to remove sigmoid saturation
    at the extremes of the score range
  - optional multi-task auxiliary head (--aux 10): also predicts the K OFIQ
    component measures most correlated with the unified score (sharpness,
    occlusion, margins, ...). Forces the backbone to learn WHY an image is
    low quality, not just that it is -- consistently improves the main task.
    The aux head is a training-time regulariser only; inference still returns
    the unified score.
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
    def __init__(self, dataframe, root, target_col, transform, lo, hi,
                 aux_cols=None, aux_lo=None, aux_hi=None):
        # dataframe already filtered to rows whose image exists
        self.files = dataframe["Filename"].tolist()
        raw = dataframe[target_col].astype("float32").to_numpy()
        # scale target -> [0,1] using the train-set range, so Sigmoid output matches.
        # (lo, hi) carry a small margin so values don't sit on the saturating ends.
        self.targets = ((raw - lo) / (hi - lo)).tolist()
        self.root = root
        self.transform = transform
        # optional auxiliary targets (OFIQ component measures), each min-max
        # scaled to [0,1] with its own train-split (lo, hi) -- multi-task learning.
        self.aux = None
        if aux_cols:
            raw_aux = dataframe[aux_cols].astype("float32").to_numpy()
            self.aux = (raw_aux - aux_lo) / (aux_hi - aux_lo)

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
        if self.aux is not None:
            return img, target, torch.tensor(self.aux[idx], dtype=torch.float32)
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


def _head_activation(head_act):
    """'sigmoid' bounds the output to [0,1] (matches the scaled target) but
    saturates at the extremes -> contributes to regression-to-the-mean.
    'linear' leaves the output unbounded; evaluate.py clamps at inference."""
    return nn.Sigmoid() if head_act == "sigmoid" else nn.Identity()


class SimpleCNN(nn.Module):
    def __init__(self, drop2d=0.10, head_drop=0.40, head_act="sigmoid", n_aux=0):
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
            _head_activation(head_act),
        )
        # multi-task: predict n_aux OFIQ component measures from the same features
        self.aux_head = None
        if n_aux > 0:
            self.aux_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(head_drop),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_aux),
                _head_activation(head_act),
            )

    def forward(self, x):
        feat = self.features(x)
        out = self.head(feat)
        if self.aux_head is not None:
            return out, self.aux_head(feat)
        return out


# --------------------------------------------------------------------------- #
# Model 2: the SMALLEST sensible ResNet, built from scratch.                   #
# torchvision's smallest is resnet18 (~11M params); on 70k images with strong  #
# labels it has way more capacity than needed and overfits hard. This custom   #
# ResNet uses ONE residual block per stage and narrow widths (~0.33M params),  #
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
    def __init__(self, widths=(16, 32, 64, 128), drop=0.10, head_drop=0.30,
                 head_act="sigmoid", n_aux=0):
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
            _head_activation(head_act),
        )
        # multi-task: predict n_aux OFIQ component measures from the same features
        self.aux_head = None
        if n_aux > 0:
            self.aux_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(head_drop),
                nn.Linear(w3, w3 // 2),
                nn.ReLU(inplace=True),
                nn.Linear(w3 // 2, n_aux),
                _head_activation(head_act),
            )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.pool(x)
        out = self.head(feat)
        if self.aux_head is not None:
            return out, self.aux_head(feat)
        return out


def build_model(arch, head_drop=None, head_act="sigmoid", n_aux=0):
    if arch == "simple":
        return SimpleCNN(head_drop=0.40 if head_drop is None else head_drop,
                         head_act=head_act, n_aux=n_aux)
    if arch == "resnet_small":
        return SmallResNet(head_drop=0.30 if head_drop is None else head_drop,
                           head_act=head_act, n_aux=n_aux)
    if arch == "resnet18":
        if n_aux > 0:
            raise ValueError("--aux is only supported for arch simple/resnet_small")
        # pretrained weights need internet/cache; download on the front-end first.
        m = models.resnet18(weights="IMAGENET1K_V1")
        # add dropout before the final layer so the big pretrained net regularises.
        m.fc = nn.Sequential(
            nn.Dropout(0.40 if head_drop is None else head_drop),
            nn.Linear(m.fc.in_features, 1),
            _head_activation(head_act),
        )
        return m
    raise ValueError(f"unknown arch: {arch}")


# --------------------------------------------------------------------------- #
# Loss function                                                                #
# --------------------------------------------------------------------------- #
def make_tail_weight_fn(scaled_targets, n_bins=20, power=0.5, max_w=5.0):
    """Build a per-sample weight function from the TRAIN label distribution.

    The label histogram is heavily concentrated in the mid-range (only ~0.4% of
    scores below 15 and ~1% above 30 in native units), so plain MSE barely sees
    the tails and the model regresses extreme predictions toward the mean.
    Weight w(y) ~ freq(y)^-power up-weights rare scores; power=0.5 is a gentle
    inverse-sqrt reweighting. Weights are normalised so their expectation over
    the data distribution is 1 (loss scale unchanged), then capped at max_w so
    a handful of ultra-rare images can't dominate a batch.
    """
    hist, edges = np.histogram(scaled_targets, bins=n_bins, range=(0.0, 1.0))
    freq = hist / max(hist.sum(), 1)
    w = (freq + 1e-6) ** (-power)
    w = w / (freq * w).sum()          # E[w] = 1 over the train distribution
    w = np.clip(w, None, max_w)
    w_t = torch.tensor(w, dtype=torch.float32)
    inner_edges = torch.tensor(edges[1:-1], dtype=torch.float32)

    def weight_fn(targets: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(targets.detach().reshape(-1).float().cpu(), inner_edges)
        return w_t[idx].reshape(targets.shape).to(targets.device)

    return weight_fn


class CombinedLoss(nn.Module):
    """MSE + (1 - Pearson r): optimises both score magnitude AND ranking.

    Why combined?  MSELoss alone minimises prediction error in absolute terms
    but gives no gradient signal for correlation.  The Pearson term directly
    pushes the model to rank images in the right order -- which is exactly what
    the evaluator measures with Spearman rho and Pearson r.

    pearson_w=0.4 means 60% MSE (magnitude) + 40% correlation (ranking).

    weight_fn (optional): maps targets -> per-sample weights; used to
    up-weight rare extreme-quality images in the MSE term (see
    make_tail_weight_fn). The Pearson term is left unweighted -- it is a
    batch-level ranking statistic, not a per-sample error.
    """
    def __init__(self, pearson_w=0.4, weight_fn=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson_w = pearson_w
        self.mse_w = 1.0 - pearson_w
        self.weight_fn = weight_fn

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight_fn is not None:
            w = self.weight_fn(targets)
            mse_loss = (w * (preds - targets) ** 2).sum() / w.sum().clamp(min=1e-8)
        else:
            mse_loss = self.mse(preds, targets)
        if self.pearson_w == 0.0:
            return self.mse_w * mse_loss
        p = preds.squeeze() - preds.squeeze().mean()
        t = targets.squeeze() - targets.squeeze().mean()
        # clamp (not +eps) the denominator: on a low-variance batch the old
        # 1e-8 epsilon still let the gradient of the Pearson term blow up,
        # which is the likely cause of the periodic val-MAE spikes seen in
        # the training curves. Clamping bounds the gradient magnitude while
        # leaving normal batches (denominator >> 1e-4) untouched.
        denom = (p.norm() * t.norm()).clamp(min=1e-4)
        r = (p * t).sum() / denom
        return self.mse_w * mse_loss + self.pearson_w * (1.0 - r)


# --------------------------------------------------------------------------- #
# Train / evaluate                                                             #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, span, optimizer=None, grad_clip=0.0,
              aux_w=0.0):
    """span = (hi - lo): converts the scaled MAE back into native score units.

    Returns (loss, mse, mae):
      - loss = whatever `criterion` computes (e.g. combined MSE+Pearson) -- the
        quantity actually being optimised / fed to the LR scheduler.
      - mse  = plain MSE in scaled [0,1] units, ALWAYS, regardless of criterion.
        Logging this separately matters because when --loss=combined (the
        default), `loss` is NOT pure MSE -- conflating the two mislabels the
        training curves and corrupts MSE-based model selection.

    Multi-task note: only the TRAINING loader yields (img, target, aux) batches;
    the aux MSE (weighted by aux_w) is added to the optimised loss there. The
    val / train-eval loaders yield (img, target) and the model's aux output is
    ignored, so all LOGGED losses and metrics stay main-task-only and remain
    comparable to runs without --aux.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, total_sq_err, total_abs_err, n = 0.0, 0.0, 0.0, 0
    torch.set_grad_enabled(train)
    for batch in loader:
        imgs, targets = batch[0], batch[1]
        aux_targets = batch[2] if len(batch) == 3 else None
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out = model(imgs)
        preds, aux_preds = out if isinstance(out, tuple) else (out, None)
        loss = criterion(preds, targets)
        if aux_targets is not None and aux_preds is not None and aux_w > 0:
            aux_targets = aux_targets.to(device, non_blocking=True)
            loss = loss + aux_w * nn.functional.mse_loss(aux_preds, aux_targets)
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

    The dashed vertical line marks the epoch with the lowest val MSE, shown
    for reference only -- it is NOT necessarily the epoch saved to disk.
    The checkpoint written by this script is always the LAST epoch actually
    run (see the save call after the training loop in main()), so the two
    can differ when val performance ticks up again after its low point.
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
    ax1.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"lowest val MSE (ep {best_epoch}, reference only)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE (scaled)"); ax1.set_title("Loss curve (true MSE)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(df["epoch"], df["train_mae"], "-o", ms=3, label="train MAE")
    ax2.plot(df["epoch"], df["val_mae"], "-o", ms=3, label="val MAE")
    ax2.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"lowest val MSE (ep {best_epoch}, reference only)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("MAE (score points)"); ax2.set_title("Error curve")
    ax2.legend(); ax2.grid(alpha=0.3)

    title = (f"Training curves — checkpoint saved = final epoch ({last_epoch}); "
             f"dashed line = lowest-val-MSE epoch ({best_epoch}), shown for reference only, not the saved epoch")
    if requested_epochs and last_epoch < requested_epochs:
        title += (f"\nRan {last_epoch}/{requested_epochs} requested epochs — stopped early: "
                  f"no val-MSE improvement for {patience} epochs after epoch {best_epoch}.")
    note = (
        "Both curves are measured with dropout off and no augmentation, on their respective "
        "image sets, so they are directly comparable epoch to epoch."
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
                    help="early-stop after this many epochs with no val-MSE gain (0=off)")
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
    ap.add_argument("--tail-weight", type=float, default=0.0,
                    help="inverse-frequency label weighting power (0=off). "
                         "0.5 = inverse-sqrt up-weighting of rare extreme "
                         "scores; the fix for regression-to-the-mean at the "
                         "tails. Applies to the MSE term of 'mse'/'combined' "
                         "losses (huber is left unweighted).")
    ap.add_argument("--head", default="sigmoid", choices=["sigmoid", "linear"],
                    help="output activation: sigmoid bounds [0,1] but "
                         "saturates at the extremes; linear removes that "
                         "saturation (evaluate.py clamps at inference)")
    ap.add_argument("--aux", type=int, default=0,
                    help="multi-task: also predict the K OFIQ .native component "
                         "measures most correlated with the target (0=off). "
                         "Teaches the backbone WHY an image is low quality.")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight of the auxiliary component-prediction MSE "
                         "added to the main loss during training")
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
    # keep the aux candidate columns too when multi-task is on
    aux_candidates = []
    if args.aux > 0:
        aux_candidates = [c for c in df.columns
                          if c.endswith(".native") and c != args.target]
    df = df[["Filename", args.target] + aux_candidates].dropna(
        subset=["Filename", args.target])
    for c in aux_candidates:  # aux NaNs (rare) get the column median, rows kept
        df[c] = df[c].fillna(df[c].median())
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

    # ---- multi-task: pick the K components most correlated with the target -- #
    # Selection and per-column scaling both use the TRAIN split only.
    aux_cols, aux_lo, aux_hi = [], None, None
    if args.aux > 0:
        yt = train_df[args.target].astype("float64")
        corrs = {}
        for c in aux_candidates:
            v = train_df[c].astype("float64")
            if v.std() > 0:
                corrs[c] = abs(v.corr(yt))
        aux_cols = sorted(corrs, key=corrs.get, reverse=True)[:args.aux]
        a = train_df[aux_cols].astype("float32").to_numpy()
        a_min, a_max = a.min(axis=0), a.max(axis=0)
        a_margin = 0.05 * (a_max - a_min)
        aux_lo, aux_hi = a_min - a_margin, a_max + a_margin
        aux_hi = np.where(aux_hi - aux_lo < 1e-6, aux_lo + 1.0, aux_hi)  # constant-col guard
        print(f"Multi-task aux targets ({len(aux_cols)}, weight {args.aux_weight}):", flush=True)
        for c in aux_cols:
            print(f"  |corr|={corrs[c]:.3f}  {c}", flush=True)

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

    # only the TRAINING dataset carries the aux targets: the aux head is a
    # training-time regulariser, and keeping val main-only means the logged
    # val loss / early stopping stay comparable to runs without --aux.
    train_ds = FFHQQualityDataset(train_df, args.root, args.target, train_tf, lo, hi,
                                  aux_cols=aux_cols, aux_lo=aux_lo, aux_hi=aux_hi)
    val_ds = FFHQQualityDataset(val_df, args.root, args.target, eval_tf, lo, hi)
    # Same training images, but no augmentation and (via run_epoch's optimizer=None)
    # no dropout -- lets the logged train_mse/train_mae be measured the same way as
    # val, so the per-epoch curves are a fair train-vs-val comparison.
    train_eval_ds = FFHQQualityDataset(train_df, args.root, args.target, eval_tf, lo, hi)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)

    # ---- model ------------------------------------------------------------- #
    model = build_model(args.arch, head_drop=head_drop, head_act=args.head,
                        n_aux=len(aux_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch} | parameters: {n_params/1e6:.2f}M", flush=True)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    # optional tail weighting: computed from TRAIN scaled targets only
    weight_fn = None
    if args.tail_weight > 0:
        weight_fn = make_tail_weight_fn(np.asarray(train_ds.targets, dtype=np.float32),
                                        power=args.tail_weight)
        print(f"Tail weighting ON: inverse-frequency power {args.tail_weight}", flush=True)

    if args.loss == "mse":
        # weighted MSE = CombinedLoss with the Pearson term switched off
        criterion = CombinedLoss(pearson_w=0.0, weight_fn=weight_fn) if weight_fn \
            else nn.MSELoss()
    elif args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:  # combined: MSE + Pearson correlation (default)
        criterion = CombinedLoss(pearson_w=0.4, weight_fn=weight_fn)
    print(f"Loss: {args.loss} | grad_clip: {args.grad_clip} | head: {args.head}", flush=True)

    # AdamW = Adam with proper (decoupled) weight decay -> the L2 regularisation
    # your professor asked for. Adam still adapts the per-parameter learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # ---- training loop ----------------------------------------------------- #
    # Early stopping still tracks val MSE (the smoother of the two metrics
    # epoch-to-epoch) to decide when to stop -- but the checkpoint saved to
    # disk is always the LAST epoch actually run, not the best-val-MSE epoch:
    # simpler, more defensible model selection (Spencer's request).
    best_val_mse = float("inf")
    epochs_since_best = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # dropout on, augmented: this pass is what actually updates the weights.
        # Its returned metrics are not logged -- they're not comparable to val
        # (dropout + augmentation both inflate the error), so logging them was
        # what made the train/val curves look backwards.
        run_epoch(model, train_loader, criterion, device, span, optimizer,
                  grad_clip=args.grad_clip, aux_w=args.aux_weight if aux_cols else 0.0)
        # dropout off, unaugmented: this is what gets logged, so train and val
        # are measured the same way and are actually comparable epoch to epoch.
        tr_loss, tr_mse, tr_mae = run_epoch(model, train_eval_loader, criterion, device, span, None)
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
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"  -> early stop: no val-MSE gain for {args.patience} epochs "
                      f"(best {best_val_mse:.5f}). Stopping at epoch {epoch}/{args.epochs} requested.",
                      flush=True)
                break

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"model_state": state, "arch": args.arch, "target": args.target,
                "img_size": args.img_size, "val_mse": va_mse, "val_mae": va_mae,
                "epoch": epoch, "target_lo": lo, "target_hi": hi, "head_drop": head_drop,
                "head_act": args.head, "tail_weight": args.tail_weight,
                "aux_cols": aux_cols, "aux_weight": args.aux_weight if aux_cols else 0.0},
               args.out)
    print(f"Saved final-epoch checkpoint -> {args.out} (epoch {epoch}, val MSE {va_mse:.5f}, "
          f"val MAE {va_mae:.2f} in native units)", flush=True)

    print(f"\nDone. Best val MSE seen during training: {best_val_mse:.5f} "
          f"(native-unit MAE at that epoch is in the log).", flush=True)

    # ---- training-set plot + overfitting diagnostics ----------------------- #
    if args.curve:
        plot_curves(log_rows, args.curve, requested_epochs=args.epochs, patience=args.patience)
    diagnose_overfitting(log_rows, span, requested_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
