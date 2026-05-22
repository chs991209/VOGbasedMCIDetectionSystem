# VOG-Based MCI Detection System

**Statistical Learning for AI Lab — Jetson AGX Orin Edge Deployment**

> **Quick links:** [Technical Overview →](OVERVIEW.md) | [Evaluation Results & Ablation Plan →](RESULT_SUMMARY.md)

---

## Project Overview

This project implements a complete end-to-end pipeline for **binary classification of Mild Cognitive Impairment (MCI) vs. Healthy Controls (HC)** using Video-Oculography (VOG) saccade recordings, deployed on NVIDIA Jetson AGX Orin.

The core premise is that MCI patients exhibit measurable degradation in oculomotor control — specifically, abnormal **gaze error dynamics** during visually-guided saccade tasks. By transforming the 1-D gaze error signal into the time-frequency domain via the Continuous Wavelet Transform (CWT), we obtain a 2-D scalogram that simultaneously encodes:
- **Reaction latency** (temporal axis): delayed saccade initiation in MCI
- **Micro-saccadic tremor** (frequency axis): high-frequency instability during post-saccadic fixation

### Pipeline Architecture

```
Raw VOG CSV  (Time, LH, RH, LV, RV, TargetH, TargetV @ ~120 Hz)
  │
  ▼
[Layer 1]  EventLockedCWTPipeline
           Trigger detection → Epoch extraction → Gaze error → Dual-band Complex Morlet CWT
           Output: [4, 20, 50] tensor  [Re_L, Im_L, Re_R, Im_R]
  │
  ├──────► [Layer 5]  XAIVisualizer
  │                   Per-subject mean power → SPM-style MCI−HC Difference Map
  │
  ▼
[Layer 2]  VOG_CWT_Dataset
           Per-channel Z-score normalization, subject-ID tracking
  │
  ▼
[Layer 3]  PureCNNClassifier  (Frequency-Preserving Asymmetric Pooling)
           Conv→BN→ReLU×3, MaxPool(2,2) → MaxPool(1,2) → no pool → AdaptiveAvgPool
  │
  ▼
[Layer 4a] ModelTrainer
           Subject-level stratified split (7:3), Weighted CE Loss, CosineAnnealingLR
  │
  ▼
[Layer 4b] JetsonInferenceEngine
           Per-epoch inference → Soft-voting ensemble → MCI probability
  │
  ▼
[Layer 6]  MonteCarloGroupEvaluator
           30 × Stratified 7:3 random splits → Mean ± Std metrics
```

### Dataset

| Group | Subjects | CSV Files | Epochs (post-rejection) | Label |
|-------|:---:|:---:|:---:|:---:|
| HC | 14 | ~116 | 2,495 | 0 |
| MCI (incl. MCI+) | 23 | ~184 | 3,241 | 1 |
| **Total** | **37** | **278 valid** | **5,736** | binary |

---

## Layer 1 — Event-Locked CWT Pipeline

### Clinical Signal: Gaze Error During Saccades

A **saccade** is a rapid, ballistic eye movement that re-foveates on a newly appeared target. In MCI, documented oculomotor deficits include:
- **Increased latency** ($\Delta t \in [50, 200]$ ms): delayed saccade initiation
- **Hypometria** (undershoot, $e > 0$): the eye falls short of the target
- **Hypermetria** (overshoot, $e < 0$): the eye moves past the target
- **Post-saccadic oscillation**: high-frequency instability during re-fixation

The **signed gaze error** is:

$$e(t) = \theta_{\mathrm{target}}(t) - \theta_{\mathrm{actual}}(t)$$

Sign is preserved deliberately — it distinguishes hypometria from hypermetria, which have different neural substrates and may carry distinct discriminative power.

### Epoching and Baseline Correction

A saccade event is detected at each discrete change in target position. Each epoch is a fixed window:

$$\mathcal{E}_k = e(t) \quad \text{for } t \in [t_k - 0.2\,\text{s},\; t_k + 0.8\,\text{s}]$$

**Baseline correction** removes the pre-stimulus DC offset:

$$\tilde{e}_k(t) = e_k(t) - \frac{1}{N_{\mathrm{pre}}} \sum_{\tau < t_k} e_k(\tau)$$

This is equivalent to the demeaning step in ERP analysis — the CWT then captures *deviation from baseline* rather than absolute gaze position.

**Artifact rejection:** Any epoch where peak gaze error exceeds 30° in either eye is discarded before CWT computation, removing blinks and tracking failures.

**Anti-saccade correction:** For anti-saccade trials, the target signal is inverted ($\theta_{\mathrm{target}} \leftarrow -\theta_{\mathrm{target}}$) so gaze error is directionally consistent across all task types.

### Dual-Band Frequency Grid

Rather than a single continuous frequency range, the pipeline uses a **dual-band** design that explicitly excludes the 10–30 Hz noise floor (cardiopulmonary rhythm ~1 Hz harmonics, muscle fasciculation):

| Band | Range | Bins | Clinical target |
|------|-------|:----:|-----------------|
| Low | 0.5 – 10 Hz | 12 | Cognitive latency, slow dynamics |
| High | 30 – 60 Hz | 8 | Micro-saccadic tremor |
| *Excluded* | *10 – 30 Hz* | *—* | *Somatic noise floor* |

Both bands use logarithmic spacing within their range. The bin split scales with `freq_bins`:

```
low_bins  = int(freq_bins * 0.6)   # → 12
high_bins = freq_bins - low_bins   # →  8
```

This ensures the proportional clinical representation is preserved regardless of `freq_bins`.

### Complex Morlet CWT

The CWT uses the **Complex Morlet wavelet** (`cmor5.0-1.0`):

$$\psi_{\mathrm{cmor}}(t) = \frac{1}{\sqrt{\pi B}}\, e^{2\pi i f_c t}\, e^{-t^2/B}, \quad B=1.0,\; f_c=5.0$$

Superiority over STFT: the CWT adaptively uses short windows at high frequencies (precise tremor localization) and long windows at low frequencies (precise latency characterization) — exactly the trade-off required by the dual-band design.

### 4-Channel Binocular Output

The full complex output is retained for both eyes, yielding a 4-channel tensor per epoch:

$$\mathbf{Z}_k = \bigl[\operatorname{Re}(W_L),\; \operatorname{Im}(W_L),\; \operatorname{Re}(W_R),\; \operatorname{Im}(W_R)\bigr] \in \mathbb{R}^{4 \times 20 \times 50}$$

Retaining phase (Re/Im) allows the network to distinguish overshoot from undershoot via instantaneous phase $\phi = \arctan(\operatorname{Im}/\operatorname{Re})$. Binocular channels allow detection of inter-ocular asymmetry, a known MCI-related conjugate gaze deficit.

---

## Layer 2 — Dataset Bridge

### Normalization

Each of the 4 channels is independently Z-score normalized per epoch:

$$\hat{Z}_k^{(c)} = \frac{Z_k^{(c)} - \mu_k^{(c)}}{\sigma_k^{(c)} + \varepsilon}, \quad \varepsilon = 10^{-8}$$

Per-channel normalization prevents the larger-magnitude real channels from suppressing the imaginary channels during gradient updates.

### SpecAugment (Training Only)

`AugmentedSubset` wraps training subsets and applies on-the-fly masking with probability $p=0.25$:
- **Frequency masking:** zero up to 3 contiguous frequency bins at a random position
- **Time masking:** zero up to 5 contiguous time bins at a random position

Masks are kept small relative to the 20×50 scalogram to preserve clinically informative latency spikes. Validation and inference inputs are never augmented.

### Subject-ID Tracking

Every sample carries a `subject_id` tag. This enables subject-level splitting — without it, epoch-level random splits allow the model to exploit **within-subject correlations** (a distinctive fixation tremor, systematic bias) to inflate validation accuracy. All evaluation splits in this pipeline operate at the subject level.

---

## Layer 3 — PureCNNClassifier

The production model is a lightweight pure-CNN (no attention modules), designed to minimize overfitting risk at N=37 subjects. The key design constraint is that MaxPool must not collapse the 20-bin dual-band frequency axis — hence the asymmetric pooling in Block 2.

| Stage | Operation | Output Shape |
|-------|-----------|:------------:|
| Input | — | `[B, 4, 20, 50]` |
| Block 1 | Conv(4→16, 3×3) + BN + ReLU + MaxPool(2,2) | `[B, 16, 10, 25]` |
| Block 2 | Conv(16→32, 3×3) + BN + ReLU + MaxPool(1,2) | `[B, 32, 10, 12]` |
| Block 3 | Conv(32→64, 3×3) + BN + ReLU | `[B, 64, 10, 12]` |
| GAP | AdaptiveAvgPool2d(1) → Flatten | `[B, 64]` |
| Head | Dropout(0.3) + FC(64→32) + ReLU + FC(32→2) | `[B, 2]` |

**Why asymmetric pooling in Block 2?**
`MaxPool(2,2)` twice would reduce 20 freq bins → 5, destroying the dual-band structure. `MaxPool(1,2)` pools only the time axis, preserving 10 freq bins through Blocks 2 and 3. The frequency axis then spans low-band (bins 0–7) and high-band (bins 8–11) feature maps simultaneously throughout the network.

> **Note:** The `EdgeCWTClassifier` (CNN–CBAM hybrid with depthwise-separable convolutions and residual connections) is retained in the codebase as a reference architecture but is not used in the current training pipeline.

---

## Layer 4a — Model Trainer

### Subject-Level Stratified Split (val_ratio = 0.3)

Subjects are partitioned by class, independently shuffled, and split using `round()`:

```
hc_val_n  = max(1, round(len(hc_subjects)  * 0.3))   # → 4
mci_val_n = max(1, round(len(mci_subjects) * 0.3))   # → 7
```

`round()` is used instead of `int()` to prevent truncation error at small cohort sizes (e.g., `int(14 * 0.3) = 4` but `int(14 * 0.2) = 2`, which undershoots). This yields approximately HC=10/MCI=16 train and HC=4/MCI=7 validation subjects.

### Class-Weighted Cross-Entropy Loss

The dataset has a ~1:1.6 (HC:MCI) class imbalance. Inverse-frequency weights are computed as:

$$w_c = \frac{N}{2 \cdot N_c}$$

These are applied **unnormalized** to `nn.CrossEntropyLoss`. Normalizing the weights was found to cause HC-bias (MCI sensitivity drops to ~30%) because it reduces the effective penalty on the majority class.

### Optimization

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW ($\alpha_0 = 10^{-3}$, $\lambda = 10^{-4}$) |
| LR Schedule | CosineAnnealingLR ($T_{\max}$ = epochs) |
| Early stopping | patience = 25 epochs (val loss) |
| Mixed precision | FP16 AMP + GradScaler on CUDA |

---

## Layer 4b — Jetson Inference Engine

At inference time, a CSV file yields $K$ valid saccade epochs. Per-epoch MCI posteriors are aggregated via **soft voting**:

$$\hat{p}(y=1) = \frac{1}{K} \sum_{k=1}^{K} p(y=1 \mid \mathbf{z}_k;\,\hat{\theta})$$

Variance decreases as $\sigma^2/K$, providing a $\sqrt{K}$-fold noise reduction over single-epoch classification.

The engine attempts to load a TensorRT engine (INT8+DLA preferred, FP16 secondary) and falls back to `PureCNNClassifier` PyTorch if no engine is found — verified to operate correctly in fallback mode on Jetson AGX Orin.

---

## Layer 5 — XAI: SPM-Style Difference Map

Before any machine learning, the pipeline generates a group contrast map as a **model-agnostic validity check**.

Per-subject mean power scalograms are computed ($P = (\operatorname{Re}^2 + \operatorname{Im}^2)/2$, in dB), and a pixel-wise **Welch t-test** identifies time-frequency bins with significant MCI−HC power differences ($p < 0.05$):

$$\Delta(f, t) = \overline{\mathrm{CWT}}_{\mathrm{MCI}}(f, t)_{\mathrm{dB}} - \overline{\mathrm{CWT}}_{\mathrm{HC}}(f, t)_{\mathrm{dB}}$$

| Color | Meaning |
|-------|---------|
| Red ($\Delta > 0$) | MCI shows higher gaze error power — oscillatory instability, prolonged undershoot |
| Blue ($\Delta < 0$) | HC shows higher power — more vigorous corrective saccades |

**Current result:** ~14.5% of time-frequency pixels reach significance ($n_\mathrm{HC}=14$, $n_\mathrm{MCI}=23$), saved as `xai_difference_map.png`.

---

## Layer 6 — Stratified Monte Carlo Group Evaluator

LOSO-CV (N=1 test set) was discarded due to prohibitively high variance at N=37. The `MonteCarloGroupEvaluator` replaces it with 30 independent stratified random splits.

### Algorithm

For each of 30 iterations (seeded by `split_num × 42`):

1. Shuffle HC subjects and MCI subjects independently
2. Select test set: `round(N_HC × 0.3)` HC + `round(N_MCI × 0.3)` MCI subjects → always **4 HC + 7 MCI**
3. Train a fresh `PureCNNClassifier` on remaining 70% with `AugmentedSubset`
4. Subject-level soft voting on test set
5. Record Accuracy, Sensitivity, Specificity, AUROC

Per-class stratification in every split is **guaranteed** — no fold can have zero HC or zero MCI subjects, which would invalidate AUROC and Specificity.

### Baseline Results (Phase 1 — PureCNN + CE Loss)

| Metric | Mean ± Std |
|--------|:----------:|
| Accuracy | 0.530 ± 0.139 |
| Sensitivity | 0.568 ± 0.253 |
| Specificity | 0.493 ± 0.302 |
| AUROC | 0.543 ± 0.127 |

The ±13.9% standard deviation in accuracy (ranging from 20% to 80% across splits) confirms the PI's assessment that single-split evaluation is statistically unreliable for this cohort size. The mean accuracy of 53.0% — below the 62% majority-class baseline — indicates the model has not yet learned a stable decision boundary. See `RESULT_SUMMARY.md` for the full per-split breakdown and the ablation plan.

---

## Outputs

| File | Description |
|------|-------------|
| `xai_difference_map.png` | SPM-style MCI−HC contrast (dual-band, binocular power) |
| `src/best_edge_cwt_model.pth` | Best PureCNN checkpoint from single-split training |
| `src/int8_calibration_data.npy` | 200 CWT tensors for TensorRT INT8 calibration |
| `pipeline_run_new.log` | Full stdout log of the latest run |
| `RESULT_SUMMARY.md` | Phase 1 evaluation report and ablation plan |

---

## Running the Pipeline

```bash
cd ~/AIResearch/VOGBasedDetectionSystem/src
~/AIResearch/VOGBasedDetectionSystem/jetson-env/bin/python3 -u \
    wavelet_transform_detection.py > ../pipeline_run_new.log 2>&1 &
tail -f ../pipeline_run_new.log
```

A parallel experimental environment is available at `~/AIResearch/VOGBasedDetectionSystemExperimental/`, sharing the same data via symlink but with an isolated source tree for ablation studies.
