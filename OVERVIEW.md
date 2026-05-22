# VOG-Based MCI Detection System
## Technical Overview

---

## 1. Problem Statement

The goal is to detect Mild Cognitive Impairment (MCI) non-invasively using Video-Oculography (VOG), which records eye position (degrees of visual angle) at ~120 Hz during structured saccade tasks. The superior colliculus and frontal eye fields — the brain structures governing saccadic control — are among the earliest affected in MCI-related neurodegeneration, making oculomotor dynamics a sensitive, fast (~2 min), and objective biomarker.

Given a time-series gaze recording, the system learns a binary classifier $f_\theta : \mathbf{x} \mapsto y \in \{0\ (\text{HC}),\ 1\ (\text{MCI})\}$ operating not on raw coordinates but on a **time-frequency representation of gaze error** — motivated by the clinical finding that MCI disrupts both saccade latency (non-stationary) and micro-saccadic tremor (frequency-specific).

---

## 2. Dataset

| Group | Subjects | Epochs (post-rejection) |
|-------|:---:|:---:|
| Healthy Control (HC) | 14 | 2,495 |
| MCI (incl. MCI+) | 23 | 3,241 |
| **Total** | **37** | **5,736** |

Each subject undergoes 8 structured saccade tasks per session (Saccade A/B/B-anti/R across Horizontal and Vertical axes). Each CSV records `Time`, `LH`, `RH`, `LV`, `RV`, `TargetH`, `TargetV`. For anti-saccade trials, the target signal is inverted so that gaze error is directionally consistent across all task types.

---

## 3. Signal Processing: Dual-Band CWT Pipeline

**Gaze error** is computed as signed difference (preserving hypometria/hypermetria) and baseline-corrected using the pre-stimulus mean. Epochs are extracted as $[-0.2, +0.8]$ s windows around each target transition; epochs where peak gaze error exceeds 30° are rejected as artifacts.

The **Continuous Wavelet Transform** (complex Morlet, $w=5.0$) is applied to both eyes independently. The output is a 4-channel binocular scalogram $\mathbf{Z} \in \mathbb{R}^{4 \times F \times T}$: $[\operatorname{Re}_L,\ \operatorname{Im}_L,\ \operatorname{Re}_R,\ \operatorname{Im}_R]$. Retaining the complex output preserves instantaneous phase, which distinguishes overshoot from undershoot and encodes oscillatory instability. Binocular channels allow the network to detect inter-ocular asymmetry.

**Dual-band frequency grid** ($F = 20$ bins, $T = 50$ time bins):
- Low band: 12 bins log-spaced from 0.5–10 Hz — cognitive latency and slow dynamics
- High band: 8 bins log-spaced from 30–60 Hz — micro-saccadic tremor
- 10–30 Hz excluded: cardiopulmonary and muscle fasciculation noise floor

Each of the 4 channels is independently Z-score normalized per epoch before being fed to the model.

---

## 4. Model: PureCNNClassifier

The current production model is a lightweight pure-CNN classifier (no attention), designed to minimize overfitting risk with N=37 subjects.

| Stage | Output Shape |
|-------|-------------|
| Input | `[B, 4, 20, 50]` |
| Conv(4→16) + BN + ReLU + MaxPool(2,2) | `[B, 16, 10, 25]` |
| Conv(16→32) + BN + ReLU + MaxPool(1,2) | `[B, 32, 10, 12]` — freq axis preserved |
| Conv(32→64) + BN + ReLU | `[B, 64, 10, 12]` — no pool |
| AdaptiveAvgPool2d(1) → Flatten | `[B, 64]` |
| Dropout(0.3) + FC(64→32) + ReLU + FC(32→2) | `[B, 2]` |

The asymmetric MaxPool in Block 2 (pooling only the time axis) preserves the 10-bin frequency resolution throughout the network, which is critical given the small $F=20$ dual-band axis.

---

## 5. Training Framework

**Subject-stratified split** (val_ratio=0.3): subjects are partitioned by class, shuffled, and split at the subject level using `round()` — guaranteeing no epoch-level data leakage. With N=37 this yields approximately HC=10, MCI=16 train and HC=4, MCI=7 validation subjects.

**SpecAugment** (p=0.25, f_mask≤3 bins, t_mask≤5 bins) is applied on-the-fly to training data only. Masks are small relative to the 20×50 scalogram to preserve clinically informative latency spikes.

**Loss:** `nn.CrossEntropyLoss` with inverse-frequency class weights ($w_c = N / (2 N_c)$, unnormalized). No Focal loss — gamma scaling was found to corrupt gradient magnitudes with this cohort size.

**Optimizer:** AdamW ($\alpha_0 = 10^{-3}$, $\lambda = 10^{-4}$) with Cosine Annealing LR schedule and early stopping (patience=25). Mixed-precision (FP16 AMP + GradScaler) on Jetson AGX Orin CUDA.

---

## 6. Evaluation: Stratified Monte Carlo Group CV

LOSO-CV (N=1 test set) was discarded due to extreme variance. Instead, **Stratified Monte Carlo Group Evaluation** runs 30 independent splits:

- Per iteration: randomly select 30% of HC subjects and 30% of MCI subjects as test set (using `round()`, seeded per iteration). Train a fresh `PureCNNClassifier` on the remaining 70% with `AugmentedSubset`.
- **Subject-level soft voting**: per-epoch MCI probabilities are averaged per subject before thresholding at 0.5.
- Report: Mean ± Std over 30 iterations for Accuracy, Sensitivity, Specificity, AUROC.

Stratification per class in every split is guaranteed, preventing test sets with zero HC or MCI subjects (which would invalidate AUROC and Specificity).

---

## 7. XAI: SPM-Style Difference Map

Before any ML, the pipeline computes a **group contrast map** as a model-agnostic validity check. Per-subject mean power scalograms are computed ($P = (\operatorname{Re}^2 + \operatorname{Im}^2)/2$, converted to dB), and a pixel-wise Welch t-test identifies time-frequency bins with significant MCI−HC power differences (p < 0.05). The masked difference map ($\Delta(f,t)$ where significant) is saved as `xai_difference_map.png`. Approximately 14.5% of pixels reach significance with the current cohort.

---

## 8. Inference

At inference time, a CSV file yields $K$ valid saccade epochs. Each epoch is transformed via the same CWT pipeline, and per-epoch MCI posteriors are averaged via **soft voting**:

$$\hat{p}(y=1) = \frac{1}{K} \sum_{k=1}^{K} p(y=1 \mid \mathbf{z}_k)$$

Variance decreases as $1/K$, providing a $\sqrt{K}$-fold noise reduction over single-epoch prediction. The `JetsonInferenceEngine` attempts to load a TensorRT engine (INT8+DLA preferred, FP16 secondary) and falls back to `PureCNNClassifier` PyTorch if no engine is found.
