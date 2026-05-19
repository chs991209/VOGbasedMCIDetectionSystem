# VOG-Based MCI Detection System
## Technical Overview: A Statistical & Machine Learning Perspective

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset & Experimental Design](#2-dataset--experimental-design)
3. [Signal Processing: Event-Locked CWT Pipeline](#3-signal-processing-event-locked-cwt-pipeline)
4. [Feature Representation: 2-Channel Complex Scalogram](#4-feature-representation-2-channel-complex-scalogram)
5. [Explainability: SPM-Style Group Contrast Maps](#5-explainability-spm-style-group-contrast-maps)
6. [Model Architecture: CNN–CBAM Hybrid](#6-model-architecture-cnncbam-hybrid)
7. [Statistical Learning Framework](#7-statistical-learning-framework)
8. [Inference: Epoch-Level Ensemble](#8-inference-epoch-level-ensemble)
9. [Limitations & Future Directions](#9-limitations--future-directions)

---

## 1. Problem Statement

**Clinical objective:** Detect Mild Cognitive Impairment (MCI) non-invasively through Video-Oculography (VOG), a technique that records eye position over time (degrees of visual angle) at ~120 Hz during structured saccade tasks.

**Statistical framing:** Given a time-series recording $\mathbf{x} \in \mathbb{R}^{T}$ from a saccade task, learn a binary classifier

$$f_\theta : \mathbf{x} \mapsto y \in \{0\ (\text{HC}),\ 1\ (\text{MCI})\}$$

where the mapping is not performed on raw coordinates but on a **time-frequency representation of gaze error** — motivated by the clinical finding that MCI disrupts oculomotor dynamics in ways that are simultaneously non-stationary (latency changes) and frequency-specific (micro-saccadic tremor).

**Why oculomotor signals?** The superior colliculus and frontal eye fields, which govern saccadic control, are among the earliest structures affected in MCI-related neurodegeneration. Saccade latency, accuracy, and post-saccadic stability have been proposed as sensitive biomarkers in multiple clinical studies, offering a fast (~2 min), non-invasive, and objective screening modality.

---

## 2. Dataset & Experimental Design

### 2.1 Cohort

| Group | Subjects (local) | Subjects (total w/ fine-tuning) | CSV files | Label |
|-------|:---:|:---:|:---:|:---:|
| Healthy Control (HC) | 14 | ~28 | 116 | 0 |
| MCI | 12 | ~24 | 96 | 1 |
| MCI+ (severe) | 11 | ~22 | 88 | 1 |
| **Total** | **37** | **~74** | **300** | binary |

MCI and MCI+ are merged into a single positive class (label 1), yielding a **binary classification** task with a class ratio of approximately **1 : 1.6 (HC : MCI)**.

### 2.2 Recording Protocol

Each subject undergoes 8 structured saccade tasks per session:

| Task type | Axis | Description |
|-----------|------|-------------|
| Saccade A | H / V | Predictable step stimulus |
| Saccade B | H / V | Randomised step stimulus |
| Saccade B (anti) | H / V | Anti-saccade: move opposite to target |
| Saccade R | H / V | Remembered target location |

Each CSV file records columns `Time(sec)`, `LH`, `RH`, `LV`, `RV`, `TargetH`, `TargetV` at a nominal sampling rate of 120 Hz.

### 2.3 Anti-Saccade Correction

For anti-saccade trials, the target signal is inverted prior to error computation:

$$\theta_{\text{target}}^{(\text{anti})}(t) = -\theta_{\text{target}}(t)$$

This re-frames the anti-saccade as a pro-saccade in the error domain, making gaze error directionally consistent across all task types.

---

## 3. Signal Processing: Event-Locked CWT Pipeline

### 3.1 Gaze Error Signal

The **signed gaze error** is defined as:

$$e(t) = \theta_{\text{target}}(t) - \theta_{\text{actual}}(t)$$

The absolute value is deliberately not applied. Preserving sign differentiates:
- **Hypometria** ($e > 0$): undershoot — the eye falls short of the target
- **Hypermetria** ($e < 0$): overshoot — the eye moves past the target

Both are clinically distinct patterns in MCI and are encoded in the phase of the complex CWT coefficients.

### 3.2 Event Detection and Epoching

A saccade event is defined by a discrete change in target position:

$$\mathcal{T} = \bigl\{t_k : \theta_{\text{target}}(t_k) \neq \theta_{\text{target}}(t_k - \Delta t)\bigr\}$$

All transitions (including return-to-zero) are included as valid events. For each $t_k \in \mathcal{T}$, a fixed epoch is extracted:

$$\mathcal{E}_k = e(t) \quad \text{for } t \in [t_k - 0.2\,\text{s},\; t_k + 0.8\,\text{s}]$$

The epoch window was chosen to capture:
- **Pre-stimulus baseline** ($[-0.2, 0)$ s): fixation stability
- **Latency window** ($[0, 0.3)$ s): onset and initiation of the saccade
- **Execution and fixation** ($[0.3, 0.8]$ s): saccade dynamics and re-fixation stability

### 3.3 Baseline Correction

Pre-stimulus mean subtraction removes the DC component of fixation bias:

$$\tilde{e}_k(t) = e_k(t) - \frac{1}{N_{\text{pre}}} \sum_{\tau < t_k} e_k(\tau)$$

where $N_{\text{pre}} = \lfloor 0.2 \cdot f_s \rfloor$ is the number of pre-stimulus samples. This is the standard baseline correction used in ERP (Event-Related Potential) analysis and ensures the CWT captures *deviation from steady-state* rather than absolute gaze position, which varies across subjects and trials.

### 3.4 Continuous Wavelet Transform

The CWT of $\tilde{e}_k(t)$ with mother wavelet $\psi$ is:

$$W_\psi[\tilde{e}_k](a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} \tilde{e}_k(t)\, \overline{\psi\!\left(\frac{t - b}{a}\right)} \, dt$$

where $a > 0$ is the **scale** parameter and $b$ is the **translation** (time shift). The frequency-scale correspondence is:

$$f = \frac{f_c}{a \cdot \Delta t}$$

with $f_c$ the center frequency of $\psi$ and $\Delta t = 1/f_s$.

### 3.5 Choice of Wavelet: Complex Morlet

The **Complex Morlet wavelet** (`cmor5.0-1.0` in PyWavelets) is:

$$\psi_{\text{cmor}}(t) = \frac{1}{\sqrt{\pi B}}\, e^{2\pi i f_c t}\, e^{-t^2 / B}$$

with bandwidth $B = 1.0$ and center frequency $f_c = 5.0$ (the bandwidth-frequency product $w = f_c / B = 5.0$).

**Why Morlet over STFT?**

The Short-Time Fourier Transform (STFT) applies a *fixed* window $h(t)$ to the signal, yielding uniform time-frequency resolution across all frequencies. This is suboptimal for saccade signals, which require:
- **High time resolution at high frequencies** (to localize micro-saccadic tremors)
- **High frequency resolution at low frequencies** (to characterize slow latency profiles)

The CWT achieves this adaptively through the Heisenberg-Gabor uncertainty principle:

$$\sigma_t \cdot \sigma_f \geq \frac{1}{4\pi}$$

At high scales (low frequencies), the Morlet window is stretched, improving frequency resolution. At low scales (high frequencies), the window is compressed, improving time resolution — exactly the trade-off needed here.

### 3.6 Scale Grid

Scales are derived from the desired frequency grid $\{f_j\}_{j=1}^{F}$ (linearly spaced from 1 to 40 Hz, $F = 40$ bins):

$$a_j = \frac{f_c}{f_j \cdot \Delta t}$$

Each resulting scalogram $W_\psi[\tilde{e}_k] \in \mathbb{C}^{F \times T}$ is resampled to a fixed time dimension $T = 100$ bins via linear interpolation, enabling batch processing.

---

## 4. Feature Representation: 2-Channel Complex Scalogram

Rather than collapsing the complex CWT to a real-valued power spectrum $|W_\psi|^2$, the pipeline retains the full complex output as two separate channels:

$$\mathbf{Z}_k = \begin{bmatrix} \operatorname{Re}(W_\psi[\tilde{e}_k]) \\ \operatorname{Im}(W_\psi[\tilde{e}_k]) \end{bmatrix} \in \mathbb{R}^{2 \times F \times T}$$

**Motivation.** The real and imaginary parts of the Morlet CWT encode complementary projections of the signal onto cosine and sine basis functions, respectively. Their ratio determines the **instantaneous phase** $\phi(f, t) = \arctan(\operatorname{Im}/\operatorname{Re})$, which distinguishes:
- Hypometria: positive real-channel dominance at saccade onset
- Hypermetria: negative real-channel dominance at saccade onset
- Oscillatory instability: phase cycling in the imaginary channel at high frequencies

Discarding phase via $|W_\psi|^2$ would render these directional patterns indistinguishable.

### Normalization

Each channel is normalized independently per epoch via Z-score:

$$\hat{Z}_k^{(c)} = \frac{Z_k^{(c)} - \mu_k^{(c)}}{\sigma_k^{(c)} + \varepsilon}, \quad c \in \{\operatorname{Re},\, \operatorname{Im}\}, \quad \varepsilon = 10^{-8}$$

Per-channel normalization preserves the relative scale relationship between real and imaginary components while preventing the typically larger-magnitude real channel from dominating the imaginary channel during gradient updates.

---

## 5. Explainability: SPM-Style Group Contrast Maps

Before any machine learning, the pipeline generates a **group contrast map** that provides model-agnostic evidence for the discriminability of the CWT feature space — serving as a clinical validity check.

### 5.1 Group-Mean Power Scalogram

For each group $g \in \{\text{HC}, \text{MCI}\}$, the mean power scalogram is computed over all $N_g$ epochs:

$$\bar{P}_g(f, t) = \frac{1}{N_g} \sum_{i=1}^{N_g} \left|W_\psi[\tilde{e}_i](f, t)\right|^2 = \frac{1}{N_g} \sum_{i=1}^{N_g} \left(Z_i^{\operatorname{Re}}(f,t)^2 + Z_i^{\operatorname{Im}}(f,t)^2\right)$$

Converting to decibels compresses the dynamic range:

$$\overline{\text{CWT}}_g(f, t)_{\text{dB}} = 10 \log_{10}\!\bigl(\bar{P}_g(f, t) + \varepsilon\bigr)$$

### 5.2 Difference Map

The **MCI − HC contrast** is:

$$\Delta(f, t) = \overline{\text{CWT}}_{\text{MCI}}(f, t)_{\text{dB}} - \overline{\text{CWT}}_{\text{HC}}(f, t)_{\text{dB}}$$

Displayed with a diverging colormap (`RdBu_r`) centered at zero:

| Region | Interpretation |
|--------|---------------|
| $\Delta > 0$ (red) | MCI patients show higher gaze error power — oscillatory instability, prolonged undershoot |
| $\Delta < 0$ (blue) | HC subjects show higher power — faster, more vigorous corrective saccades |

**Expected Regions of Interest (RoIs)** based on clinical literature:
- $t \in [0.2, 0.4]$ s, $f \in [1, 10]$ Hz: MCI reaction latency elevation
- $t \in [0.4, 0.8]$ s, $f \in [15, 30]$ Hz: MCI post-saccadic tremor

### 5.3 Note on Statistical Thresholding

The current implementation plots the raw mean difference $\Delta(f, t)$ without formal thresholding. A complete SPM analysis would additionally apply:
- **Permutation testing** (label-shuffling) to derive a null distribution of $\Delta(f, t)$ under $H_0$: no group difference
- **Cluster-based correction** for the family-wise error rate (FWER) across the $F \times T = 4{,}000$ time-frequency bins — analogous to the random field theory correction used in fMRI SPM

---

## 6. Model Architecture: CNN–CBAM Hybrid

### 6.1 Input and Motivation

The input tensor $\mathbf{Z}_k \in \mathbb{R}^{2 \times 40 \times 100}$ is treated as a 2-channel image in the frequency-time plane. A convolutional architecture is appropriate because:
1. MCI biomarkers are **locally structured** in $(f, t)$ space — not scattered randomly
2. Convolution provides **translation equivariance**, detecting patterns regardless of their exact position in the scalogram
3. The dataset size (~74 subjects) necessitates a **low-parameter** model to avoid overfitting

### 6.2 Depthwise-Separable Convolutions

Standard convolution with kernel $k \times k$, $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels requires $k^2 C_{\text{in}} C_{\text{out}}$ parameters. Depthwise-separable convolution factorizes this into:

$$\underbrace{k^2 C_{\text{in}}}_{\text{depthwise}} + \underbrace{C_{\text{in}} C_{\text{out}}}_{\text{pointwise}} \quad \text{vs.} \quad k^2 C_{\text{in}} C_{\text{out}}$$

For $k = 3$, this is a reduction factor of $\approx 8\times$, enabling a deeper network within the parameter budget imposed by the small dataset.

### 6.3 CBAM Attention

The Convolutional Block Attention Module applies two sequential gates to each feature map $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$:

**Channel attention** — *which frequency bands are informative?*

$$\mathbf{M}_c(\mathbf{F}) = \sigma\!\left(\mathrm{MLP}\bigl(\mathbf{F}^c_{\mathrm{avg}}\bigr) + \mathrm{MLP}\bigl(\mathbf{F}^c_{\mathrm{max}}\bigr)\right) \in [0, 1]^{C}$$

where $\mathbf{F}^c_{\mathrm{avg}} = \frac{1}{HW}\sum_{h,w}\mathbf{F}_{:,h,w}$ and the shared MLP has a bottleneck of $\lfloor C/4 \rfloor$ neurons.

**Spatial attention** — *at which $(f, t)$ position does the discriminative pattern occur?*

$$\mathbf{M}_s(\mathbf{F}') = \sigma\!\left(f^{7\times7}\bigl([\mathbf{F}'^s_{\mathrm{avg}};\, \mathbf{F}'^s_{\mathrm{max}}]\bigr)\right) \in [0, 1]^{H \times W}$$

where $\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$. The $7 \times 7$ kernel spans a large receptive field in the $(f, t)$ plane, capturing broad latency and frequency patterns simultaneously.

**XAI interpretation:** $\mathbf{M}_s$ in the final block (before Global Average Pooling) constitutes a learned, class-discriminative saliency map over the time-frequency plane — directly analogous to the manually computed difference map $\Delta(f, t)$, but derived end-to-end from the classification objective.

### 6.4 Architecture Summary

| Stage | Output shape | Param. count |
|-------|-------------|:---:|
| Input | `[B, 2, 40, 100]` | — |
| Block 1: Conv(2→16) + BN + ReLU + MaxPool | `[B, 16, 20, 50]` | 304 |
| CBAM 1: Channel(16) + Spatial(7×7) | `[B, 16, 20, 50]` | 386 |
| Block 2: DW(16) + PW(16→32) + BN + ReLU + MaxPool | `[B, 32, 10, 25]` | 688 |
| CBAM 2: Channel(32) + Spatial(7×7) | `[B, 32, 10, 25]` | 1,346 |
| Block 3: DW(32) + PW(32→64) + BN + ReLU | `[B, 64, 10, 25]` | 2,368 |
| CBAM 3: Channel(64) + Spatial(7×7) | `[B, 64, 10, 25]` | 4,994 |
| GAP → Flatten | `[B, 64]` | — |
| FC(64→32) + ReLU | `[B, 32]` | 2,080 |
| FC(32→2) | `[B, 2]` | 66 |
| **Total** | | **~12,200** |

The model is intentionally parameter-sparse relative to the dataset size, placing the effective degrees of freedom well below the number of independent training subjects.

---

## 7. Statistical Learning Framework

### 7.1 Subject-Level Stratified Split

**The critical issue:** Each subject contributes multiple saccade epochs (one per CSV file per eye per task). A naive epoch-level random split allows epochs from the same subject to appear in both the training and validation sets. The model can then exploit **subject-specific idiosyncrasies** (e.g., a distinctive baseline fixation tremor) to achieve high validation accuracy without learning generalizable MCI biomarkers — a form of **data leakage** that directly violates the i.i.d. assumption of the train/test split.

**Solution:** The split is performed at the **subject level**:

1. Partition subjects by class: $\mathcal{S}_{\text{HC}}$, $\mathcal{S}_{\text{MCI}}$
2. Independently shuffle each partition
3. Assign $\lfloor 0.2 |\mathcal{S}_c| \rfloor$ subjects per class to the validation set
4. Map subject IDs to epoch indices: $\mathcal{I}_{\text{val}} = \bigcup_{s \in \mathcal{S}_{\text{val}}} \mathcal{I}_s$

This guarantees $\mathcal{I}_{\text{train}} \cap \mathcal{I}_{\text{val}} = \emptyset$ at the **subject level**, not merely the epoch level.

### 7.2 Class-Weighted Cross-Entropy Loss

The dataset exhibits a class imbalance of approximately 1:1.6 (HC:MCI). Standard cross-entropy

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i \mid \mathbf{x}_i)$$

minimizes the average log-loss uniformly, biasing the gradient toward the majority class. The weighted cross-entropy applies **inverse-frequency weights**:

$$w_c = \frac{N}{K \cdot N_c}, \qquad \mathcal{L}_{\text{WCE}} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \log p(y_i \mid \mathbf{x}_i)$$

where $K = 2$ and $N_c$ is the epoch count of class $c$. These weights normalize the effective contribution of each class to the total loss, so that $\mathbb{E}[\mathcal{L}_{\text{WCE}} \mid y = 0] = \mathbb{E}[\mathcal{L}_{\text{WCE}} \mid y = 1]$.

For the current cohort ($N_{\text{HC}} \approx 1{,}000$, $N_{\text{MCI}} \approx 1{,}600$ epochs):

$$w_{\text{HC}} \approx 1.30, \qquad w_{\text{MCI}} \approx 0.81$$

When the dataset is perfectly balanced ($N_{\text{HC}} = N_{\text{MCI}}$), both weights equal 1.0 and the loss reduces identically to standard CE — the implementation is robust to both cases.

### 7.3 Optimizer: AdamW

Standard Adam applies L2 regularization by augmenting the gradient:

$$g_t \leftarrow g_t + \lambda \theta_t$$

This couples weight decay with the adaptive learning rate scaling, causing it to be effectively reduced for parameters with large gradient variance — a documented failure mode. **AdamW** (decoupled weight decay) corrects this:

$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} - \alpha \lambda \theta_t$$

where $\hat{m}_t$, $\hat{v}_t$ are the bias-corrected first and second moment estimates. The regularization term $-\alpha \lambda \theta_t$ is applied independently of the adaptive scaling, providing more reliable L2 regularization — important given the small dataset and risk of overfitting.

Parameters: $\alpha_0 = 10^{-3}$, $\lambda = 10^{-4}$.

### 7.4 Learning Rate Schedule: Cosine Annealing

The learning rate follows a cosine decay from $\alpha_0$ to $\alpha_{\min} \approx 0$:

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_0 - \alpha_{\min})\left(1 + \cos\frac{\pi t}{T_{\max}}\right)$$

with $T_{\max} = $ `epochs`. Unlike step decay, cosine annealing provides a smooth, continuous reduction with no abrupt transitions, and has been shown empirically to converge to flatter minima with better generalization properties.

### 7.5 Mixed-Precision Training (FP16)

On the Jetson AGX Orin (Ampere architecture), FP16 operations on Tensor Cores run at up to $2\times$ the throughput of FP32. PyTorch AMP (`torch.amp.autocast`) automatically casts eligible operations (matrix multiplications, convolutions) to FP16 while preserving FP32 precision for numerically sensitive operations (batch normalization statistics, loss accumulation).

**Gradient scaling** (`GradScaler`) multiplies the loss by a large scalar $s$ before backpropagation to prevent FP16 underflow in the gradient values, then divides by $s$ before the optimizer step:

$$\tilde{g} = s \cdot \nabla_\theta \mathcal{L}, \qquad \theta \leftarrow \theta - \frac{\alpha}{s} \tilde{g}$$

The scaler $s$ is adapted dynamically: reduced if overflow (NaN/Inf gradients) is detected, increased otherwise.

---

## 8. Inference: Epoch-Level Ensemble

Given a new CSV recording with $K$ extractable saccade epochs, the inference engine computes a per-epoch posterior and aggregates via **soft-voting**:

$$\hat{p}(y = 1 \mid \mathcal{X}) = \frac{1}{K} \sum_{k=1}^{K} p(y = 1 \mid \mathbf{z}_k;\, \hat{\theta})$$

where $\mathbf{z}_k$ is the normalized 2-channel CWT tensor for epoch $k$.

**Statistical justification.** Under the assumption that individual saccade epochs are conditionally independent given the patient's clinical state,

$$\operatorname{Var}[\hat{p}] = \frac{\sigma_{\text{epoch}}^2}{K}$$

so variance decreases as $1/K$ with the number of epochs. In practice a single CSV file yields $K \in [5, 20]$ valid epochs, providing a factor of $\sqrt{5}$–$\sqrt{20} \approx 2.2$–$4.5\times$ reduction in prediction noise compared to single-epoch classification.

The final binary decision uses the MAP rule:

$$\hat{y} = \mathbf{1}\!\left[\hat{p}(y=1 \mid \mathcal{X}) > 0.5\right]$$

with $\hat{p}$ reported as the MCI confidence score.

---

## 9. Limitations & Future Directions

### 9.1 Sample Size

With $N \approx 74$ subjects total, the effective sample size for evaluating generalization is limited. The subject-level 80/20 split leaves approximately 15 subjects for validation, making accuracy estimates high-variance. **Recommended extension:** Replace the single held-out split with **Leave-One-Subject-Out Cross-Validation (LOSO-CV)**, which uses all $N$ subjects as validation subjects exactly once, yielding a lower-variance estimate of generalization accuracy.

### 9.2 Epoch Independence Assumption

The soft-voting ensemble assumes conditional independence of epochs from the same subject. In reality, successive saccade epochs within a trial are likely positively correlated (shared fatigue, attention, and baseline drift effects). Accounting for this within-subject correlation — e.g., via a mixed-effects model or a subject-level aggregation before classification — is a natural extension.

### 9.3 Evaluation Metrics

Binary accuracy is not an appropriate primary metric for a clinical screening tool. Recommended metrics for reporting:
- **Sensitivity (Recall):** $\text{TP} / (\text{TP} + \text{FN})$ — fraction of MCI patients correctly identified
- **Specificity:** $\text{TN} / (\text{TN} + \text{FP})$ — fraction of HC correctly identified
- **AUROC:** area under the ROC curve, threshold-independent
- **Balanced Accuracy:** $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$

### 9.4 Formal XAI Thresholding

The SPM-style difference map $\Delta(f, t)$ currently visualizes raw group mean differences without statistical thresholding. A rigorous analysis would apply:
1. **Permutation testing:** shuffle group labels $B = 1{,}000$ times to estimate the null distribution $\Delta_0(f, t)$
2. **Cluster-based FWER correction:** identify connected clusters where $|\Delta(f, t)| > \delta_0$ (e.g., $p < 0.001$ uncorrected) and threshold by cluster size under the null — the standard approach in SPM12 / FieldTrip for electrophysiological data

### 9.5 Toward a Foundation Model

The current architecture trains from scratch on the available cohort. With the additional fine-tuning dataset ($\sim 37$ subjects), a two-stage approach is recommended:
1. **Pre-training** on all available data to learn a generalizable CWT-feature extractor
2. **Fine-tuning** on site-specific or protocol-specific cohorts by freezing the convolutional backbone and retraining the classifier head

This mirrors the standard practice in transfer learning and is expected to substantially improve generalization given the small per-cohort sample sizes.
