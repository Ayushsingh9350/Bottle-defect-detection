# Manufacturing Defect Detection with Autoencoders

An unsupervised anomaly detection system that flags manufacturing defects in product images — trained **only on normal, defect-free images**, without ever seeing a labeled defect during training.

**[Live Demo](#)** &nbsp;|&nbsp; **[Kaggle Notebook](#)**

![Anomaly Heatmap Example](assets/heatmap_example1.png)

---

## Problem

Manufacturing quality control faces two hard constraints that make standard classification a poor fit:

- **Defects are rare.** In a working production line, the vast majority of items are normal — labeled defect data is scarce and heavily imbalanced.
- **Defects are unpredictable.** Scratches, contamination, breakage, and misalignment can look completely different from one another, and new defect types can appear that were never seen during data collection. A classifier trained on past defect types has no way to recognize an unfamiliar one.

This project uses a **convolutional autoencoder** trained exclusively on normal images. Because the model never sees defects during training, it learns to reconstruct normal patterns well — and reconstructs *anything unfamiliar* (regardless of what kind of defect it is) poorly. That reconstruction error becomes the anomaly signal, sidestepping both the imbalance and the "unknown defect type" problem at once.

## Dataset

[MVTec AD (Anomaly Detection) dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al.) — an industry-standard benchmark for unsupervised anomaly detection. This project uses the **bottle** category:

| Split | Count |
|---|---|
| Train (normal only) | 209 images |
| Test — normal | 20 images |
| Test — defective | 63 images (broken_large, broken_small, contamination) |

## Approach

1. **Architecture:** Convolutional autoencoder (encoder → bottleneck → decoder), trained to reconstruct input images, with input/target both set to normal images only.
2. **Anomaly scoring:** Per-image reconstruction error (pixel-wise MSE between input and output). Higher error = more anomalous.
3. **Localization:** Per-pixel reconstruction error maps double as heatmaps, showing *where* on the image the model struggled to reconstruct — not just whether an image is anomalous.
4. **Threshold selection:** Rather than using a fixed or arbitrary cutoff, the decision threshold was chosen from the precision-recall curve on the held-out test set, targeting a minimum 90% recall (prioritizing catching real defects over avoiding false alarms — appropriate for a QC setting where missed defects are costlier than false positives).

## Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.861** |
| Precision | 0.864 |
| Recall | 0.905 |
| F1 | 0.884 |
| Decision threshold | 0.00094 |

**Reconstruction error by category:**

| Defect type | Mean error | vs. normal baseline |
|---|---|---|
| Normal (good) | 0.000946 | — |
| broken_large | 0.001463 | 1.55x |
| contamination | 0.001464 | 1.55x |
| broken_small | 0.001394 | 1.47x |

All three defect categories separate clearly from the normal baseline, with large breaks and contamination showing the strongest signal.

## Limitations

- **Dark-on-dark contamination is harder to detect.** Contaminants that are tonally similar to the bottle's dark interior produce a weaker reconstruction-error signal than visually distinct defects (e.g. light-colored debris, threads), since pixel-wise MSE is less sensitive to low-contrast structural changes.
- **Attempted fix (SSIM + augmentation) did not help.** Swapping to a combined MSE+SSIM loss with rotation/shift augmentation was tested to address the above, but reduced ROC-AUC from ~0.86 to 0.68 — likely due to the augmentation strategy not suiting a training set of only 209 images, which introduced reconstruction artifacts. Reverted to the original MSE-based model, which was more reliable overall. This is a natural next direction — a feature-based approach (e.g. pretrained-CNN feature comparison, as in PaDiM) would likely generalize better than further tuning pixel-level reconstruction.
- **Small test set (83 images)** — results are indicative rather than statistically robust; a production system would need evaluation on a larger sample.

## Model

Lightweight convolutional autoencoder — **333,955 parameters (1.27 MB)**. Small enough for fast CPU inference and simple deployment without GPU requirements.

## Demo

Deployed with Streamlit — upload a bottle image and get a pass/fail verdict with an anomaly heatmap overlay.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

TensorFlow/Keras · NumPy · scikit-learn (metrics) · Streamlit (deployment) · Matplotlib

## Project Structure

```
├── app.py                  # Streamlit deployment app
├── requirements.txt
├── bottle_autoencoder.h5   # trained model
├── notebook.ipynb          # training + evaluation notebook

```
