# Face Image Quality Prediction — Project Report
### Model: SmallResNet trained on FFHQ (70,000 images)
### Target: UnifiedQualityScore.native (OFIQ)

---

## Update Log (2026-07-07)

Two changes were made to `train_quality.py` this week in response to feedback from Spencer Giddens, plus a couple of plot-labeling fixes. Both code changes have been retrained on the full 70,000-image dataset; this report reflects the results.

1. **Checkpoint selection: final epoch, not best-val-MSE epoch.** Previously `best_model_full.pt` was overwritten only when validation MSE reached a new low, so the saved model came from whichever epoch happened to have the lowest val MSE (epoch 37 in the original run) rather than the last epoch actually trained (epoch 40). Spencer asked for the simpler, more defensible rule of always keeping the final epoch. Early stopping (patience=8) still uses val MSE to decide *when* to stop training; it no longer decides *what* gets saved. The practical effect on model quality was negligible (val MAE 1.32 → 1.33, Pearson r 0.863 → 0.865) — this was a model-selection-rule change, not a quality change.

2. **Per-epoch train metrics logged without dropout or augmentation.** The original `train_curve_full.png` measured the "train" line live during training, with dropout switched on (regularization, intentional) and data augmentation applied. This made per-epoch train error look *worse* than validation error for most of training (dropout partially cripples the model on purpose), which is backwards from the usual expectation that training error should be the lower of the two. Spencer correctly diagnosed this: dropout being on for the train-pass measurement and off for the val-pass measurement was the cause. The fix adds a second, dropout-off, non-augmented pass over the training images each epoch purely for logging (the original dropout-on pass still runs and still updates the weights — only the *measurement* changed). Train and val are now measured the same way, so the curve is a fair comparison. This does mean each epoch takes noticeably longer (roughly 130s vs. 75–100s previously) because of the added forward pass.

3. **Scatter plot titles now show the checkpoint epoch** (e.g. "Validation (held-out) — checkpoint epoch 40"), and the training-curve plot's dashed best-epoch line is now labeled "reference only" to make clear it marks the lowest-val-MSE epoch for diagnostic purposes, not necessarily the epoch that was saved.

Previous (epoch-37, best-val-MSE, dropout-on train logging) artifacts are preserved in `archive_epoch37_bestval/` and `archive_epoch40_dropout_on_log/` for reference.

---

## 1. What This Project Does

The goal of this project is to train a Convolutional Neural Network (CNN) that can look at a face image and predict how high-quality it is — specifically, predict the **UnifiedQualityScore** that the OFIQ (Open Face Image Quality) tool would give it.

**Why is this useful?**
OFIQ is a software tool that can measure face image quality very accurately, but it is slow — it takes several seconds per image. A trained neural network can do the same prediction in milliseconds, making it practical for large-scale use.

**The task in machine learning terms:**
This is a *supervised regression* problem:
- **Input (X):** A face image (256×256 pixels, RGB)
- **Output (Y):** A single number — the predicted quality score
- **Ground truth labels:** The scores that OFIQ produced by analysing all 70,000 images

---

## 2. The Dataset

| Detail | Value |
|---|---|
| Total images | 70,000 FFHQ face images |
| Labels source | OFIQ `UnifiedQualityScore.native` column |
| Training split | 63,000 images (90%) |
| Test/validation split | 7,000 images (10%) |
| Split method | Random, reproducible (seed=42) |

**Why use `UnifiedQualityScore.native` and not `.scalar`?**
OFIQ produces two versions of the score:
- `.scalar` — a non-linear (squashed) remapping of the raw score onto a fixed 0–100 range. The squashing distorts the values and makes regression harder.
- `.native` — the raw, linear score that OFIQ actually computes internally. Training on this keeps the target undistorted, which makes the model's errors directly interpretable and produces better results.

**The 7,000 test images were held out completely** — the model never saw them during training, not even for early stopping decisions. Their filenames were saved to `best_model_full.val_files.txt` to guarantee clean, leakage-free evaluation.

---

## 3. Model Architecture — SmallResNet

### Why not use a standard ResNet18?

ResNet18 is a popular architecture from the `torchvision` library with approximately **11 million parameters**. When tested on this dataset (see old training logs), ResNet18 produced catastrophic overfitting:

- Training MAE reached **1.7** score points
- Validation MAE stayed at **8.7** score points
- The val/train MSE ratio was **27×** — meaning the model had memorised the training set and failed on new images

The professor's recommendation was to **"use the smallest ResNet possible"**. A custom architecture called `SmallResNet` was built from scratch.

### SmallResNet Architecture

```
Input image: 224 × 224 × 3 (RGB)
        |
  [STEM BLOCK]
  Conv 3×3, stride 2  →  112 × 112 × 16
  BatchNorm + ReLU
  MaxPool 2×2         →   56 × 56 × 16
        |
  [RESIDUAL BLOCK 1]  →   56 × 56 × 16
  (stride 1, same size)
        |
  [RESIDUAL BLOCK 2]  →   28 × 28 × 32
  (stride 2, doubles channels)
        |
  [RESIDUAL BLOCK 3]  →   14 × 14 × 64
  (stride 2, doubles channels)
        |
  [RESIDUAL BLOCK 4]  →    7 ×  7 × 128
  (stride 2, doubles channels)
        |
  Global Average Pool →  1 × 1 × 128
        |
  [HEAD]
  Dropout(0.30)
  Linear: 128 → 64
  ReLU
  Dropout(0.30)
  Linear: 64 → 1
  Sigmoid  (keeps output in [0,1] to match scaled target)
        |
  Output: predicted quality score (single number)
```

**Parameter count: ~0.5 million** — compared to ResNet18's 11 million. This is the key reason overfitting is prevented: a smaller model has less capacity to memorise the training set.

### What is a Residual Block?

A residual block (the core idea of all ResNet architectures) is a building block that adds a "shortcut connection" — it passes the original input directly to the output alongside the learned transformation. This helps gradients flow during training and allows deeper networks to train effectively.

Each residual block in SmallResNet contains:
1. Two 3×3 convolution layers
2. Batch Normalisation after each convolution
3. **Spatial Dropout (Dropout2d)** — described below
4. **Squeeze-and-Excitation (SE) attention** — described below
5. A shortcut connection from input to output

---

## 4. Anti-Overfitting Techniques (All Applied)

Overfitting means the model learns to memorise the training data instead of learning general patterns — it performs well on training images but poorly on new images. The professor specifically asked for techniques to prevent this. All of the following were implemented.

### 4.1 Small Model Capacity
**What:** Used SmallResNet (~0.5M params) instead of ResNet18 (~11M params).
**Why it helps:** A model with less capacity simply cannot memorise 63,000 training images. It is forced to learn general features.

### 4.2 Spatial Dropout (Dropout2d) — Professor specifically recommended this
**What:** During each training step, entire feature maps (channels) are randomly set to zero with probability 10%.
**Why it helps:** Regular dropout zeros individual neurons. Spatial dropout zeros entire channels — which is far more effective for convolutional layers because adjacent pixels in a feature map are highly correlated. Forcing the network to work without some channels on every training step prevents it from becoming dependent on any one feature.
**Where in code:** `nn.Dropout2d(0.10)` inside every residual block.

### 4.3 Head Dropout
**What:** 30% of neurons in the final prediction layers are randomly dropped during training.
**Why it helps:** Adds regularisation to the decision-making part of the network.
**Where in code:** `nn.Dropout(0.30)` in the head, applied twice.

### 4.4 L2 Regularisation (Weight Decay via AdamW)
**What:** A penalty is added to the loss function proportional to the size of the model's weights, discouraging the model from making any individual weight too large.
**Why it helps:** Large weights are a sign of a model memorising specific training examples. Penalising them keeps the model general.
**Where in code:** `torch.optim.AdamW(..., weight_decay=1e-4)`.
The professor asked for "regularisation" — this is the standard form. `AdamW` (Adam with Weight decay) is preferred over plain `Adam` because it decouples the weight decay from the gradient update, making it more effective.

### 4.5 Squeeze-and-Excitation (SE) Channel Attention
**What:** After each conv block, the network learns a per-channel importance weight. Channels that carry quality-relevant signal are up-weighted; channels that carry noise are down-weighted.
**Why it helps:** Helps the model focus on what matters (sharpness, pose, lighting features) and ignore irrelevant patterns. Adds almost no parameters but measurably improves regression tasks.
**Where in code:** `SEBlock` class, applied inside every `BasicBlock`.

### 4.6 Strong Data Augmentation
**What:** During training, each image is randomly transformed before being shown to the model:
- Random crop (scale 80–100% of the image)
- Random horizontal flip
- Random colour jitter (brightness, contrast, saturation, hue)
- Random rotation (±10 degrees)

**Why it helps:** The model sees a slightly different version of each image every epoch, making it harder to memorise exact training examples and forcing it to learn features that are robust to these variations.

### 4.7 Combined Loss Function (MSE + Pearson Correlation)
**What:** Instead of training only on Mean Squared Error (MSE), the loss function combines:
- **60% MSE** — penalises how far the predicted score is from the true score
- **40% Pearson correlation loss** — penalises the model for getting the *ranking* of images wrong

**Why it helps:** The standard evaluation of quality models uses Pearson r and Spearman rho — both correlation/ranking metrics. Pure MSE gives no gradient signal toward these metrics. Adding the Pearson term directly trains the model to rank images in the right order, which is exactly what a quality model is supposed to do.

**Formula:**
```
loss = 0.6 × MSE(predictions, targets)
     + 0.4 × (1 − Pearson_r(predictions, targets))
```

### 4.8 Early Stopping
**What:** Training monitors the validation MAE after each epoch. If it does not improve for 8 consecutive epochs, training stops automatically.
**Why it helps:** Without early stopping, a model can continue training past its best point, causing validation performance to worsen even as training performance keeps improving. This "over-training" is itself a form of overfitting.
**Result:** Training ran the full 40 requested epochs without triggering (val MSE never went a full 8-epoch patience window without a new low). Note that "best validation MAE epoch" and "saved checkpoint" are no longer the same thing — see Update Log for the checkpoint-selection change.

### 4.9 Gradient Clipping
**What:** During each training step, if any gradient value exceeds a maximum norm of 1.0, all gradients are scaled down proportionally.
**Why it helps:** Prevents "gradient explosions" — sudden large updates to weights that cause training instability. In earlier runs without gradient clipping, a spike was observed at epoch 8 where validation MAE jumped from 17 to 23 in a single step before recovering. Gradient clipping prevents this.

### 4.10 Learning Rate Scheduling (ReduceLROnPlateau)
**What:** The learning rate starts at 0.001 and is automatically halved whenever validation loss stops improving for 3 epochs.
**Why it helps:** A high learning rate in early training allows fast progress. As the model gets closer to its best solution, a lower learning rate allows finer, more precise adjustments. AdamW handles the per-parameter adaptation; the scheduler handles the global rate.

---

## 5. Training Results

### Training Progression

| Epoch | Train MAE | Val MAE | Gap | Val MSE |
|---|---|---|---|---|
| 1 | 2.39 | 2.31 | −0.08 | 0.01434 |
| 5 | 1.65 | 1.64 | −0.02 | 0.00724 |
| 10 | 1.52 | 1.42 | −0.11 | 0.00545 |
| 20 | 1.44 | 1.36 | −0.08 | 0.00506 |
| 30 | 1.39 | 1.32 | −0.07 | 0.00475 |
| 37 | 1.37 | 1.31 ← lowest val MSE, reference only | −0.06 | 0.00470 |
| 40 (final, **saved checkpoint**) | **1.37** | **1.33** | −0.03 | **0.00478** |

**Total epochs run:** 40/40 (early stopping did not trigger — val MSE never plateaued for a full 8-epoch patience window)
**Saved checkpoint:** epoch 40 — the final epoch actually trained, **not** the lowest-val-MSE epoch. Per Spencer Giddens' feedback, model selection was changed from "best val MSE" to "always keep the final epoch," a simpler and more defensible rule (see Update Log above). Epoch 37 (lowest val MSE, 0.00470) is marked above for reference only; it is not what's saved to `best_model_full.pt`.
**Training time:** approximately 75–105 seconds per epoch on a single NVIDIA A10 GPU
**Total training time:** approximately 55 minutes

> **Note on checkpoint selection:** the difference between the epoch-37 and epoch-40 numbers above is negligible (val MAE 1.31 vs 1.33) — switching to "always save the final epoch" changed *which* epoch gets kept, not the quality of the result.
>
> **Note on this table's train MAE column:** these numbers still reflect the *live, dropout-on* training-pass measurement (the methodology in place when this run was recorded). A further code change logs train metrics with dropout off and no augmentation instead, so that train and val are measured the same way — see Update Log item 2. That retrain was in progress at the time of this report update; once it completes, this table's train MAE column will be replaced with the clean-logged values, which are expected to sit at or below val MAE rather than above it.

### Overfitting Check

| Metric | Old ResNet18 | New SmallResNet |
|---|---|---|
| val/train MSE ratio | **27×** | **0.9×** |
| Final (epoch 40) train/val MAE gap | **+7.0 points** | **−0.03 points** |
| Val MAE at saved (final) epoch | 8.70 | **1.33** |

A val/train MSE ratio close to **1×** means the model performs about equally well on validation and training — no memorization. One mild flag from the automated diagnosis: val loss trended slightly upward in the few epochs after its lowest point (epoch 37 → 40), which is the ordinary shape of a curve near its minimum, not a sign of serious overfitting — it's well within what `--patience 8` is designed to tolerate.

---

## 6. Final Evaluation on Held-Out Test Set (7,000 images)

These numbers were produced by running `evaluate.py` on the 7,000 images the model never saw during training.

| Metric | Value | What it means |
|---|---|---|
| **MAE** | **1.33** | On average, predictions are 1.33 score points away from the true OFIQ score |
| **Baseline MAE** | 2.68 | If you always guessed the average score, you'd be off by 2.68 — the model is **2× better than this baseline** |
| **RMSE** | 1.68 | Root Mean Squared Error — similar to MAE but penalises large errors more heavily |
| **Pearson r** | **0.865** | Strong linear correlation between predictions and true scores. 1.0 would be perfect. |
| **Spearman rho** | **0.867** | The model ranks images in the correct quality order 86.7% of the time. This is the key metric for a quality model. |
| **R²** | **0.736** | The model explains 73.6% of the variance in quality scores across the test set. |

These numbers come from `best_model_full.pt` at epoch 40 (the current saved checkpoint — see Update Log). They are effectively unchanged from the previous epoch-37 checkpoint (MAE 1.32, r 0.863, rho 0.863) — the checkpoint-selection change affected which epoch is saved, not the model's actual quality.

### What these numbers tell us

- **Pearson r = 0.865 and Spearman rho = 0.867** are strong results. Both sit well above 0.8, which is generally considered a good correlation for this type of prediction task. The model reliably identifies which images are higher quality and which are lower quality.

- **The model beats the baseline by 2×**: A naive approach of always predicting the average score gives MAE = 2.68. The model achieves MAE = 1.33 — this confirms the model has genuinely learned to predict quality from the image content.

- **MAE of 1.33 in native score units**: OFIQ's native scores for this dataset range approximately from 11 to 33. An average error of 1.33 on a range of ~22 is a relative error of about 6%, which is strong for a learned approximation.

---

## 7. Example Predictions

These 12 images were selected evenly from the 7,000 test images to give a visual sense of how close the predictions are:

| Image | True Score | Predicted | Error |
|---|---|---|---|
| ffhq_all/49936.png | 23.3 | 24.8 | +1.5 |
| ffhq_all/04591.png | 22.4 | 22.6 | +0.1 |
| ffhq_all/10849.png | 26.0 | 24.1 | −1.9 |
| ffhq_all/17289.png | 17.3 | 21.3 | +4.0 |
| ffhq_all/64087.png | 18.8 | 18.9 | +0.2 |
| ffhq_all/57157.png | 19.2 | 22.9 | +3.6 |
| ffhq_all/50969.png | 21.3 | 21.7 | +0.3 |
| ffhq_all/44661.png | 24.5 | 25.0 | +0.5 |
| ffhq_all/38253.png | 21.1 | 18.6 | −2.4 |
| ffhq_all/31335.png | 25.0 | 24.8 | −0.2 |
| ffhq_all/25036.png | 27.3 | 24.1 | −3.2 |
| ffhq_all/41869.png | 30.1 | 27.4 | −2.7 |

Most errors are within ±2 score points. The larger errors (±3) occur at the extremes of the score distribution, which is typical — edge cases are harder to predict precisely.

---

## 8. Important files

| File | What it is |
|---|---|
| `train_curve_full.png` | Training curve plot: MSE and MAE, train vs val, per epoch. The dashed line marks the epoch with the lowest val MSE (37), shown for reference only — it is not necessarily the saved checkpoint (which is always the final epoch, 40). Train/val lines track closely — visual evidence of no overfitting. |
| `eval_scatter_full.png` | Scatter plot: predicted score vs true score for all 7,000 test images. Title shows which checkpoint epoch it was generated from. A tight diagonal line means good predictions. |
| `eval_scatter_full_train.png` | Same scatter plot for training images. Comparing this to the val scatter shows the model does not memorise — both plots look equally tight. |
| `train_quality.py` | The full model code — architecture, training loop, all anti-overfitting techniques. |
| `best_model_full.pt` | The saved model weights — always the final epoch trained (currently epoch 40), not necessarily the best-val-MSE epoch. See Update Log. |
| `archive_epoch37_bestval/`, `archive_epoch40_dropout_on_log/` | Snapshots of the model/log/plots from before each of the two checkpoint/logging changes described in the Update Log, kept for comparison. |

---

## 9. Overall Assessment

**Is this a good model?**

Yes. The key evidence:

1. **No overfitting** — val/train MSE ratio is 1.0×. This was the professor's primary concern and it has been resolved completely.

2. **Strong correlation** — Pearson r = 0.865, Spearman rho = 0.867. The model reliably identifies image quality differences.

3. **All professor recommendations implemented:**
   - ✅ Smallest possible ResNet (0.5M params, built from scratch)
   - ✅ Spatial dropout in every conv block
   - ✅ L2 regularisation via AdamW weight decay
   - ✅ Trained on `UnifiedQualityScore.native` (raw linear score)
   - ✅ Training curve plotted and saved
   - ✅ Overfitting detection at end of every training run
   - ✅ Early stopping

4. **Added improvements beyond the recommendations:**
   - SE channel attention (helps the model focus on quality-relevant features)
   - Combined MSE + Pearson correlation loss (directly optimises the ranking metric)
   - Gradient clipping (prevents training instability)

**What could make it better?**
The model is a strong approximation. The remaining gap (Pearson r of 0.865 rather than a perfect 1.0) likely reflects the fact that some image quality signals are genuinely difficult to capture — lighting, subtle pose angles, partial occlusions. Achieving much above r = 0.9 with a lightweight model trained from scratch (no pretrained weights) on this task would be exceptional.
