#!/usr/bin/env python
"""
model_architecture.py
----------------------
Standalone copy of the model classes needed to load best_model_full.pt.

best_model_full.pt was trained with arch="resnet_small", so the only classes
required to reconstruct it are SEBlock, BasicBlock, and SmallResNet below
(copied unmodified from train_quality.py). No other file or class is needed.

HOW TO LOAD THE CHECKPOINT
--------------------------
    import torch
    from model_architecture import SmallResNet

    ckpt = torch.load("best_model_full.pt", map_location="cpu", weights_only=False)
    model = SmallResNet()          # head_drop doesn't matter for inference --
    model.load_state_dict(ckpt["model_state"])   # dropout has no learned weights
    model.eval()                                 # and is disabled in eval() anyway

WHAT ELSE IS IN THE CHECKPOINT (all plain Python values, not model code)
-------------------------------------------------------------------------
    ckpt["arch"]       -> "resnet_small" (confirms which architecture to use)
    ckpt["target"]      -> "UnifiedQualityScore.native" (the OFIQ column it predicts)
    ckpt["img_size"]    -> 224 (resize input images to this before feeding them in)
    ckpt["epoch"]       -> 40 (the final epoch trained; the checkpoint is always saved
                           from the last epoch actually run, not from whichever epoch
                           had the lowest validation MSE -- see train_quality.py's
                           training loop for the save call)
    ckpt["val_mse"]/["val_mae"] -> the validation error at that epoch

The model's raw output is a single number in [0, 1] (from the final Sigmoid).
To convert it back to a real OFIQ score, undo the min-max scaling used at
training time, stored right here in the checkpoint:

    native_score = model_output * (ckpt["target_hi"] - ckpt["target_lo"]) + ckpt["target_lo"]

RUNNING INFERENCE ON A NEW IMAGE
---------------------------------
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = tf(Image.open("some_face.png").convert("RGB")).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        out = model(img).item()
    score = out * (ckpt["target_hi"] - ckpt["target_lo"]) + ckpt["target_lo"]
"""

import torch
import torch.nn as nn


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


class BasicBlock(nn.Module):
    """One residual block: conv-bn-relu-conv-bn -> SE attention -> add the
    original input back in (the 'residual'/skip connection) -> relu."""
    def __init__(self, cin, cout, stride=1, drop=0.10):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.se = SEBlock(cout)
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
        out = self.se(out)
        out = out + self.short(x)
        return self.relu(out)


class SmallResNet(nn.Module):
    """The architecture best_model_full.pt was trained with (~0.33M params).
    stem -> 4 residual stages (16 -> 32 -> 64 -> 128 channels) -> pooled head."""
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
            nn.Linear(w3, w3 // 2),
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
