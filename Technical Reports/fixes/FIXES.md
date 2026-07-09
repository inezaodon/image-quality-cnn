# Problems Found and Fixes Applied

Each entry: the problem, the evidence, and the fix. Code fixes land in the repo root
(`train_quality.py`, `evaluate.py`, `REPORT.md`) and in this report folder's `scripts/`.

**Note on dropout.** The architecture *does* contain dropout — spatial dropout
(`Dropout2d`, p=0.10) inside every residual block and two head dropout layers
(p=0.30) — but dropout is active **only during the training (weight-update) passes**.
In eval mode (`model.eval()`) PyTorch disables all dropout layers automatically, so
validation, logged metrics, `evaluate.py`, and deployment inference all run
dropout-free. Wherever this document or the technical report says "dropout on/off",
it refers to this measurement/inference distinction, never to an architecture change:
every run trains with dropout and evaluates without it.

---

## 1. Regression to the mean at score extremes (model problem)

**Problem.** The validation scatter (`eval_scatter_full.png`) shows predictions pulled
toward the mean: low-quality images (true < 16) are over-predicted, high-quality
images (true > 29) are under-predicted. Root causes: only 0.37% of labels fall below
15 and 1.0% above 30, MSE is dominated by the mid-range bulk, and the sigmoid output
head saturates at the extremes.

**Fix (code, `train_quality.py`).**
- New `--tail-weight POWER` flag: inverse-frequency sample weighting of the MSE term.
  Weights are computed from the train-split label histogram (20 bins), normalized to
  mean 1, capped at 5x, so rare extreme scores contribute proportionally more gradient.
  Default `0.0` (off) preserves existing behavior.
- New `--head {sigmoid,linear}` flag: a linear output head removes sigmoid saturation
  at the extremes. The head type is stored in the checkpoint; `evaluate.py` clamps
  predictions to the valid scaled range at inference.

**Fix (post-hoc, `evaluate.py`).** New `--calibrate` flag: fits isotonic regression
(pure-numpy PAVA, no sklearn dependency) on one half of the validation set and reports
before/after metrics on the other half, so calibration is never fit and scored on the
same images.

---

## 2. Unexplained training spikes (training stability)

**Problem.** Val MAE jumps at epochs 15, 20, 22, 26, and 34 — including after the LR
was halved — visible in both train and val curves. Likely cause: the Pearson term of
the combined loss is computed per batch (64 images); when a batch happens to have low
target variance, the correlation denominator is tiny and its gradient explodes.

**Fix (code, `train_quality.py`).** The Pearson denominator is clamped
(`min=1e-4` instead of `+1e-8`), which bounds the gradient magnitude for
low-variance batches without changing behavior on normal batches.

---

## 3. False "no early-stopping leakage" claim (documentation)

**Problem.** `REPORT.md` states the 7,000 test images were "never seen during training,
not even for early stopping decisions." The code contradicts this: validation MSE
drives early stopping and the LR scheduler every epoch. This is standard practice
(model-selection signal, not gradient leakage) but the claim as written is wrong.

**Fix (docs, `REPORT.md`).** Rewritten to say the val images were never used for
gradient updates, and that they do drive early stopping and LR scheduling — the
standard, defensible formulation.

---

## 4. Spearman rho misinterpretation (documentation)

**Problem.** `REPORT.md` says rho = 0.867 means "the model ranks images in the correct
quality order 86.7% of the time." Spearman rho is a rank correlation coefficient, not
a percentage of correctly ordered pairs.

**Fix (docs, `REPORT.md`).** Reworded to describe rho as strong monotonic rank
agreement between predicted and true quality orderings.

---

## 5. Parameter count inconsistency (documentation)

**Problem.** `REPORT.md` and a code comment say "~0.5M parameters"; the runtime log
prints 0.33M.

**Fix (docs/code).** Both corrected to ~0.33M.

---

## 6. Early-stopping metric mismatch (documentation vs code)

**Problem.** The `train_quality.py` docstring and `--patience` help text say early
stopping monitors val MAE; the implementation monitors val MSE. `REPORT.md` 4.8
repeats the MAE claim.

**Fix (docs/code).** Docstring, help text, and REPORT.md corrected to val MSE.

---

## 7. Metrics figure mixes units (visualization)

**Problem.** `fig_metrics_summary.png` put MAE/RMSE (native score units) and
r/rho/R-squared (unitless, 0-1) on one axis, visually understating the correlation
metrics.

**Fix (code, `scripts/generate_figures.py`).** Split into two panels: error metrics
in native units (with the baseline MAE line) and correlation metrics on a 0-1 axis.
Figure regenerated and the report PDF recompiled.

---

## 8. Missing diagnostic visualizations (visualization gap)

**Problem.** No figure exposes the tail bias directly (binned residuals), the
calibration slope, or which OFIQ component measures correlate with the model's
errors. These require per-image predictions, which `evaluate.py` never saved.

**Fix (code).**
- `evaluate.py` now writes `<model>_val_preds.csv` (Filename, true, pred) by default,
  and `<model>_train_preds.csv` for the train-scatter sample.
- New `scripts/generate_diagnostic_figures.py` builds three figures from that CSV:
  1. `fig_binned_residuals.png` — mean residual +/- std per true-score bin;
  2. `fig_calibration.png` — fitted prediction-vs-truth slope against the ideal y=x;
  3. `fig_component_correlation.png` — correlation of the residual with each OFIQ
     `.native` component measure (what the model fails to see).
  The script exits with instructions if the predictions CSV does not exist yet
  (it requires rerunning `evaluate.py` on a machine with the images).

---

## 9. Single-task training wastes the component labels (model improvement)

**Problem.** The OFIQ CSV contains 27 component `.native` measures per image
(occlusion, eye visibility, margins, compression artifacts, luminance, ...), but the
model only ever saw the single unified score. It learns *that* an image is low
quality without being pushed to learn *why* — weaker features, especially for the
rare defect types that drive the tails.

**Fix (code, `train_quality.py`).** Multi-task learning via `--aux K --aux-weight W`
(job script uses `--aux 10 --aux-weight 0.3`):

- At startup the K components most correlated with the unified score are selected
  on the train split (top of the ranking on real data: FaceOcclusionPrevention
  |corr|=0.39, EyesVisible 0.27, MarginBelow 0.25, HeadSize 0.20, ...), each
  min-max scaled to [0,1] with train-split ranges.
- Both architectures gain an auxiliary head off the same pooled features
  (`Linear(w3, w3/2) -> ReLU -> Linear(w3/2, K)`); training loss becomes
  `main_loss + W * MSE(aux_pred, aux_true)`.
- The aux head is a **training-time regulariser only**: val/train-eval loaders stay
  main-task-only, so logged curves, early stopping, and LR scheduling remain
  comparable to previous runs. `evaluate.py` and `model_architecture.py` load
  multi-task checkpoints via `ckpt["aux_cols"]` and take the unified output.

---

## 10. Training at 224 discards native image detail (model improvement)

**Problem.** FFHQ images are native 256x256, but training resized them to 224 —
throwing away exactly the fine detail that sharpness/compression-type quality cues
depend on, before the model ever saw them.

**Fix (config, `JOB_train_quality_full.sh`).** Train at native resolution with
`--img-size 256` (the flag already existed; adaptive pooling makes the architecture
resolution-agnostic). Costs roughly 30% more GPU time per epoch. `evaluate.py`
already reads `img_size` from the checkpoint, so evaluation follows automatically.

---

## Status summary

| # | Problem | Type | Status |
|---|---------|------|--------|
| 1 | Tail compression / regression to mean | Model | Fix implemented; requires retrain (`--tail-weight 0.5`, optional `--head linear`) + `--calibrate` available now |
| 2 | Training spikes | Stability | Fixed (Pearson denominator clamp) |
| 3 | Early-stopping leakage claim | Docs | Fixed in REPORT.md |
| 4 | Spearman misread | Docs | Fixed in REPORT.md |
| 5 | Param count 0.5M vs 0.33M | Docs | Fixed in REPORT.md + code comment |
| 6 | Early-stop MAE vs MSE wording | Docs/code | Fixed in docstring, help, REPORT.md |
| 7 | Mixed-unit metrics figure | Viz | Fixed; figure regenerated |
| 8 | Missing diagnostics | Viz | Scripts ready; figures generate once predictions CSV exists |
| 9 | Component labels unused | Model | Implemented; in job script (`--aux 10 --aux-weight 0.3`); takes effect on retrain |
| 10 | Trained below native resolution | Model | In job script (`--img-size 256`); takes effect on retrain |
