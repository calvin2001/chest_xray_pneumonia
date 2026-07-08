# Chest X-Ray Pneumonia Classification

*Read this in [한국어](./README.ko.md).*

A deep-learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, built as an end-to-end study of medical image classification with PyTorch. The focus is not just on hitting a number, but on **evaluating models the way a clinician would** — where a *missed pneumonia* is the mistake that matters most.

---

## Problem

Given a chest X-ray, decide whether it shows signs of pneumonia. This is a binary classification task with a real medical asymmetry: telling a sick patient they are healthy (a false negative) is far more dangerous than a false alarm. That asymmetry drives every evaluation decision in this project.

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle (~5,856 images).
- **Split as provided:** `train` (5,216) · `val` (16) · `test` (624).
- **Class imbalance:** pneumonia outnumbers normal roughly **3:1** in the training set (3,875 vs. 1,341), so accuracy is misleading and recall on the pneumonia class is tracked instead.
- **Validation caveat:** the stock `val` folder holds only **16 images**, far too few to be reliable, so a proper stratified validation set is carved out of `train` (see below).

## Approach

The project is structured as a progression, each step motivating the next:

1. **Data exploration** — how an image becomes a `[C, H, W]` tensor, class-imbalance check, visual comparison of normal vs. pneumonia lungs.
2. **Domain-aware augmentation** — random horizontal flip, small rotation (±10°), and mild brightness jitter. Vertical flips and large rotations are deliberately excluded because an upside-down or heavily rotated chest X-ray is not physically realistic. Augmentation is applied to **train only**; validation/test use fixed transforms so evaluation stays honest and reproducible.
3. **Baseline: a small CNN from scratch** — a deliberately weak baseline (limited data can't teach a network to "see" from zero) that makes the transfer-learning gain visible.
4. **Transfer learning with ResNet18 (ImageNet)** — reuse the pretrained feature extractor and retrain only the classifier head. Two regimes are compared: **feature extraction** (backbone frozen, train `fc` only) and **fine-tuning** (unfreeze all, small learning rate).
5. **Overfitting control & medical evaluation** — dropout, early stopping, best-model checkpointing, a rebuilt stratified validation set, loss-curve diagnosis, confusion-matrix interpretation in clinical terms, and a visual error analysis of the missed pneumonia cases.

## Why recall is the headline metric

With labels `NORMAL = 0`, `PNEUMONIA = 1`:

- **False Negative (missed pneumonia)** — a sick patient sent home untreated. The most expensive error clinically.
- **False Positive (false alarm)** — a healthy patient flagged as pneumonia. Causes extra tests and anxiety, but is the *safe* kind of error.

Minimising false negatives means **maximising recall**, so recall is the primary metric, with precision and F1 reported for balance.

## Results

Held-out test set (624 images: 234 normal + 390 pneumonia), 3 epochs each, same loaders:

| Model | Recall | Precision | F1 | Missed pneumonia (FN) | False alarms (FP) |
|---|---|---|---|---|---|
| Small CNN (from scratch)               | 0.99 | 0.69 | 0.82 | 3 | 172 |
| ResNet18 — feature extraction (freeze) | 0.98 | 0.85 | **0.91** | 8 | 65 |
| ResNet18 — fine-tuning                 | **1.00** | 0.75 | 0.86 | 1 | 128 |

After refining with a proper stratified validation set, dropout, and early stopping (see the notebook's Section 5), the from-scratch CNN's precision rises to ~0.72 (F1 ~0.83) and the fine-tuned model settles at recall ~1.00 / precision ~0.76 / F1 ~0.86.

*Numbers are from Colab (Tesla T4) runs; augmentation and dropout are stochastic, so re-runs vary slightly.*

## What the results say

- **Transfer learning is the workhorse.** With only ~5k images, borrowing ImageNet features beats training from scratch on precision and F1, at a fraction of the effort.
- **Every model reaches very high recall (~0.98–1.00).** The fine-tuned network misses essentially no pneumonia (1 of 390); its lower precision — more false alarms — is the acceptable trade-off in a medical setting.
- **Freeze vs. fine-tune is a genuine choice, not a formality.** The frozen feature extractor gives the best overall balance (F1 ≈ 0.91), while fine-tuning maximises recall. Which to ship depends on whether the clinical goal is "miss nothing" or "balanced triage".
- **Error analysis over raw numbers.** Inspecting the missed-pneumonia X-rays (rather than only reading metrics) is what turns a score into an understanding of where and why the model fails.

## How to run

Designed for Google Colab with a GPU runtime.

1. Open `chest_xray_pneumonia.ipynb` in Colab and set **Runtime → Change runtime type → GPU**.
2. Create a Kaggle API token (**Kaggle → Settings → API → Create New Token**) and upload the resulting `kaggle.json` into the Colab file browser.
3. Run the cells top to bottom. The setup cells install the Kaggle client, download and unzip the dataset, and verify the folder structure before training begins.

## Tech stack

Python · PyTorch · torchvision (ResNet18) · scikit-learn (metrics & stratified split) · matplotlib · Google Colab (Tesla T4 GPU).

## Repository

```
.
├── chest_xray_pneumonia.ipynb   # end-to-end notebook (data → CNN → transfer learning → evaluation)
├── README.md                    # this file
└── README.ko.md                 # Korean version
```
