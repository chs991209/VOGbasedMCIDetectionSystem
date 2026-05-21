import matplotlib
matplotlib.use('Agg')

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import pywt
from scipy.ndimage import zoom
from scipy.stats import ttest_ind

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# =========================================================================
# CBAM: Convolutional Block Attention Module
# Channel attention (WHAT to focus on) + Spatial attention (WHERE to focus)
# Spatial attention aligns with XAI goal: highlight Time-Freq RoIs
# =========================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg + mx).view(b, c, 1, 1) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg    = torch.mean(x, dim=1, keepdim=True)
        mx, _  = torch.max(x,  dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))

# =========================================================================
# Focal Loss: down-weights easy samples so the model focuses on hard
# HC/MCI boundary cases. Replaces CrossEntropyLoss with label smoothing.
# gamma=2.0 is standard; alpha mirrors class-weight balancing.
# =========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class-weight tensor (same role as CE weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# =========================================================================
# [Layer 1] Data Engineering: Event-Locked CWT Pipeline
# Enhancements:
#   - Logarithmic frequency spacing (finer resolution at low freqs)
#   - Binocular 4-channel output [Re_L, Im_L, Re_R, Im_R] per epoch
#   - Artifact rejection (peak gaze error > threshold → skip epoch)
#   - 3-level data_store: group → subject_id → task → [tensors]
# =========================================================================
class EventLockedCWTPipeline:
    def __init__(self, pre_stimulus_sec=0.2, post_stimulus_sec=0.8,
                 min_freq=1.0, max_freq=60.0, freq_bins=40,
                 target_time_bins=100, w_morlet=5.0,
                 artifact_threshold=30.0, dual_band=True):
        self.pre_sec            = pre_stimulus_sec
        self.post_sec           = post_stimulus_sec
        self.min_freq           = min_freq
        self.max_freq           = max_freq
        self.freq_bins          = freq_bins
        self.target_time_bins   = target_time_bins
        self.w                  = w_morlet
        self.artifact_threshold = artifact_threshold
        # Dual-band: 0-10Hz (cognitive delay) + 30-60Hz (micro-tremors)
        # Middle 10-30Hz band excluded — not clinically informative for MCI
        if dual_band:
            low_band  = np.logspace(np.log10(0.5), np.log10(10.0), 25)
            high_band = np.logspace(np.log10(30.0), np.log10(60.0), 15)
            self.frequencies = np.concatenate([low_band, high_band])
        else:
            self.frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), freq_bins)

        self.target_tasks = {
            "Horizontal": ["Horizontal Saccade A", "Horizontal Saccade B",
                           "Horizontal Saccade B (anti)", "Horizontal Saccade R"],
            "Vertical":   ["Vertical Saccade A",   "Vertical Saccade B",
                           "Vertical Saccade B (anti)",   "Vertical Saccade R"],
        }
        # 3-level store: group -> subject_id -> task -> [binocular tensors]
        self.data_store = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )

    # ------------------------------------------------------------------
    def _load_csv_safely(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if any('lh' in c for c in df.columns):
                return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
        except Exception:
            pass
        for enc in ['utf-16', 'utf-16le', 'utf-8-sig', 'cp949']:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    line_clean = line.replace('\x00', '').lower()
                    if 'lh' in line_clean and 'rh' in line_clean:
                        header_cols = [col.replace('\x00', '').strip().lower() for col in line.split(',')]
                        parsed = [
                            [v.strip() for v in l.replace('\x00', '').strip().split(',')]
                            for l in lines[i + 1:] if l.strip()
                        ]
                        df = pd.DataFrame(parsed, columns=header_cols)
                        return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
            except UnicodeError:
                continue
        raise ValueError(f"Headers missing or unreadable in {file_path.name}")

    # ------------------------------------------------------------------
    def _cwt_one_signal(self, signal, scales, wavelet_name, dt):
        """CWT for a single 1-D signal. Returns (real_resized, imag_resized)."""
        cwtm, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=dt)
        tz = self.target_time_bins / cwtm.shape[1]
        return (zoom(np.real(cwtm), (1.0, tz), mode='nearest', order=1),
                zoom(np.imag(cwtm), (1.0, tz), mode='nearest', order=1))

    # ------------------------------------------------------------------
    def _extract_binocular_epochs(self, df, target_col, left_col, right_col, fs):
        """
        Returns list of [4, freq, time] tensors: [Re_L, Im_L, Re_R, Im_R].
        Combines both eyes into one binocular epoch; skips artifacts.
        """
        target_val = df[target_col].fillna(0).values
        left_val   = df[left_col].fillna(0).values
        right_val  = df[right_col].fillna(0).values
        event_indices = np.where(np.diff(target_val, prepend=0) != 0)[0]

        samples_pre  = int(self.pre_sec  * fs)
        samples_post = int(self.post_sec * fs)
        dt           = 1.0 / fs
        wavelet_name = f'cmor{self.w}-1.0'
        scales       = pywt.central_frequency(wavelet_name) / (self.frequencies * dt)

        valid_cwts = []
        for idx in event_indices:
            s, e = idx - samples_pre, idx + samples_post
            if s < 0 or e > len(df):
                continue

            err_L = target_val[s:e] - left_val[s:e]
            err_L = err_L - np.mean(err_L[:samples_pre])
            err_R = target_val[s:e] - right_val[s:e]
            err_R = err_R - np.mean(err_R[:samples_pre])

            # Artifact rejection: skip if either eye exceeds threshold (degrees)
            if (np.max(np.abs(err_L)) > self.artifact_threshold or
                    np.max(np.abs(err_R)) > self.artifact_threshold):
                continue

            re_L, im_L = self._cwt_one_signal(err_L, scales, wavelet_name, dt)
            re_R, im_R = self._cwt_one_signal(err_R, scales, wavelet_name, dt)

            # Stack: [Re_L, Im_L, Re_R, Im_R] → [4, freq, time]
            valid_cwts.append(np.stack([re_L, im_L, re_R, im_R], axis=0))

        return valid_cwts

    # ------------------------------------------------------------------
    def process_directory(self, base_dir: Path):
        csv_files = [f for f in base_dir.rglob('*.csv') if 'PD VOG' in f.name]
        processed = 0
        for filepath in csv_files:
            clean_task = filepath.stem.replace("PD VOG -_", "").replace("PD VOG -", "").strip()
            axis_type = ("Horizontal" if "Horizontal" in clean_task
                         else "Vertical" if "Vertical" in clean_task else None)
            if not axis_type or clean_task not in self.target_tasks[axis_type]:
                continue

            group = None
            cur = filepath.parent
            while cur != base_dir and cur != cur.parent:
                if cur.name.upper().startswith("HC"):  group = "HC"; break
                if cur.name.upper().startswith("MCI"): group = "MCI"; break
                cur = cur.parent
            if not group:
                for part in filepath.parts:
                    if part.upper().startswith("HC"):  group = "HC"; break
                    if part.upper().startswith("MCI"): group = "MCI"; break
            if not group:
                continue

            subject_id = filepath.parent.name
            try:
                df        = self._load_csv_safely(filepath)
                is_anti   = "anti" in clean_task.lower()
                axis_char = 'h' if axis_type == "Horizontal" else 'v'

                time_col   = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
                time_val   = df[time_col].dropna().values
                current_fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else 120.0

                target_col = next(
                    (c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c), None
                )
                if not target_col: continue
                if is_anti: df[target_col] = df[target_col] * -1

                left_col  = next((c for c in df.columns if c == f'l{axis_char}'), None)
                right_col = next((c for c in df.columns if c == f'r{axis_char}'), None)
                if not left_col or not right_col:
                    continue

                cwt_epochs = self._extract_binocular_epochs(
                    df, target_col, left_col, right_col, current_fs
                )
                if cwt_epochs:
                    self.data_store[group][subject_id][clean_task].extend(cwt_epochs)
                    processed += 1
            except Exception as e:
                print(f"[!] Skipped {filepath.name}: {e}")

        print(f"[*] Processed {processed} CSV files")

# =========================================================================
# [Layer 2] PyTorch Dataset Bridge
# Enhancements:
#   - Updated for 3-level data_store (group→subject→task→tensors)
#   - 4-channel normalization [Re_L, Im_L, Re_R, Im_R]
# =========================================================================
class VOG_CWT_Dataset(Dataset):
    def __init__(self, data_store):
        self.X           = []
        self.y           = []
        self.subject_ids = []

        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for subject_id, tasks in subjects.items():
                for task, tensors in tasks.items():       # 3-level: no eye dimension
                    for tensor in tensors:                # [4, freq, time]
                        norm_chs = []
                        for ch in range(tensor.shape[0]):
                            d = tensor[ch]
                            norm_chs.append((d - np.mean(d)) / (np.std(d) + 1e-8))
                        self.X.append(np.stack(norm_chs, axis=0))
                        self.y.append(label)
                        self.subject_ids.append(subject_id)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# =========================================================================
# [Layer 2b] SpecAugment Wrapper
# Applied only to training subsets; validation/inference use raw tensors.
# Frequency masking + time masking applied on-the-fly per batch.
# =========================================================================
class AugmentedSubset(Dataset):
    """
    Wraps any Subset and applies SpecAugment-style masking on-the-fly.
    Each mask is applied independently with probability p.
    Max mask widths are kept small to preserve latency spikes in short saccade epochs.
    """
    def __init__(self, subset, f_mask_param=5, t_mask_param=8, p=0.5):
        self.subset       = subset
        self.f_mask_param = f_mask_param   # max freq bins to zero
        self.t_mask_param = t_mask_param   # max time bins to zero
        self.p            = p              # probability of applying each mask

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = x.clone()
        freq_bins, time_bins = x.shape[1], x.shape[2]
        if random.random() < self.p:
            f_size = random.randint(1, min(self.f_mask_param, freq_bins))
            f0 = random.randint(0, freq_bins - f_size)
            x[:, f0:f0 + f_size, :] = 0.0
        if random.random() < self.p:
            t_size = random.randint(1, min(self.t_mask_param, time_bins))
            t0 = random.randint(0, time_bins - t_size)
            x[:, :, t0:t0 + t_size] = 0.0
        return x, y

# =========================================================================
# [Layer 3] Edge AI Model: CNN-CBAM Hybrid (enhanced)
#
# Enhancements:
#   - in_channels=4 (binocular: Re_L, Im_L, Re_R, Im_R)
#   - Residual connections on Blocks 2 & 3 (skip connection via 1×1 conv)
#
# Input:  [B, 4, 40, 100]
# Block 1: Conv(4→16)  + BN + ReLU + MaxPool       → [B, 16, 20, 50]  + CBAM
# Block 2: DW+PW(16→32) + residual + MaxPool        → [B, 32, 10, 25]  + CBAM
# Block 3: DW+PW(32→64) + residual                  → [B, 64, 10, 25]  + CBAM → GAP
# Head:   Dropout(0.3) + FC(64→32) + ReLU + FC(32→2)
# =========================================================================
class EdgeCWTClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=4):
        super().__init__()

        # Block 1: standard conv — no residual (input channels vary)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.cbam1 = CBAM(16)

        # Block 2: DW-Sep + residual skip
        self.block2_main = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),  # depthwise
            nn.Conv2d(16, 32, kernel_size=1),                         # pointwise
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.block2_skip = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.cbam2 = CBAM(32)

        # Block 3: DW-Sep + residual skip (no pool — preserve spatial for CBAM)
        self.block3_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # depthwise
            nn.Conv2d(32, 64, kernel_size=1),                         # pointwise
            nn.BatchNorm2d(64),
        )
        self.block3_skip = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.cbam3 = CBAM(64)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # Block 1 (no residual)
        x = self.cbam1(self.block1(x))
        # Block 2 with residual
        x = self.cbam2(F.relu(self.block2_main(x) + self.block2_skip(x), inplace=True))
        # Block 3 with residual
        x = self.cbam3(F.relu(self.block3_main(x) + self.block3_skip(x), inplace=True))
        # GAP → flatten → classify
        x = self.gap(x)
        return self.classifier(torch.flatten(x, 1))

# =========================================================================
# [Layer 4a] Model Trainer
# Enhancements:
#   - AugmentedSubset applied to training split only
#   - Label smoothing (0.1) in CrossEntropyLoss
#   - Early stopping with patience=10
#   - Default epochs raised to 50
#   - Device priority: CUDA → MPS (Apple Silicon) → CPU
# =========================================================================
class ModelTrainer:
    def __init__(self, model, device="auto"):
        if device in ("auto", "cuda"):
            if torch.cuda.is_available():            self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():  self.device = torch.device("mps")
            else:                                    self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model     = model.to(self.device)
        self.use_amp   = self.device.type == "cuda"
        self.scaler    = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def _subject_stratified_split(self, dataset, val_ratio=0.2, seed=42):
        random.seed(seed)
        subject_to_idx = defaultdict(list)
        for i, sid in enumerate(dataset.subject_ids):
            subject_to_idx[sid].append(i)

        hc_subjects  = [s for s in subject_to_idx if dataset.y[subject_to_idx[s][0]].item() == 0]
        mci_subjects = [s for s in subject_to_idx if dataset.y[subject_to_idx[s][0]].item() == 1]
        random.shuffle(hc_subjects); random.shuffle(mci_subjects)

        hc_val_n  = max(1, int(len(hc_subjects)  * val_ratio))
        mci_val_n = max(1, int(len(mci_subjects) * val_ratio))

        val_subjects   = set(hc_subjects[:hc_val_n]   + mci_subjects[:mci_val_n])
        train_subjects = set(hc_subjects[hc_val_n:]   + mci_subjects[mci_val_n:])

        train_idx = [i for i, s in enumerate(dataset.subject_ids) if s in train_subjects]
        val_idx   = [i for i, s in enumerate(dataset.subject_ids) if s in val_subjects]

        n_hc  = sum(1 for s in train_subjects if dataset.y[subject_to_idx[s][0]].item() == 0)
        n_mci = sum(1 for s in train_subjects if dataset.y[subject_to_idx[s][0]].item() == 1)
        print(f"[*] Train subjects: HC={n_hc}, MCI={n_mci} | "
              f"Val subjects: HC={hc_val_n}, MCI={mci_val_n}")

        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def train_model(self, dataset, epochs=50, batch_size=32, patience=10):
        print(f"[*] 학습 시작 (디바이스: {self.device})")

        train_raw, val_subset = self._subject_stratified_split(dataset)
        # SpecAugment applied to training data only
        train_subset = AugmentedSubset(train_raw)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

        label_counts  = Counter(dataset.y.tolist())
        total         = len(dataset)
        alpha = torch.tensor(
            [total / (2 * label_counts[i]) for i in range(2)], dtype=torch.float32
        )
        alpha = (alpha / alpha.sum()).to(self.device)   # normalize → sums to 1
        criterion = FocalLoss(alpha=alpha, gamma=1.5)
        print(f"[*] Focal alpha — HC: {alpha[0]:.3f}, MCI: {alpha[1]:.3f}  gamma=1.5")

        scheduler      = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_val_loss  = float('inf')
        no_improve     = 0

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss = 0.0; train_correct = 0; train_total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss    = criterion(outputs, labels)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward(); self.optimizer.step()
                train_loss    += loss.item() * inputs.size(0)
                _, predicted   = outputs.max(1)
                train_total   += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # --- Validation ---
            self.model.eval()
            val_loss = 0.0; val_correct = 0; val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = self.model(inputs)
                        loss    = criterion(outputs, labels)
                    val_loss    += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total   += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc    = 100. * train_correct / train_total
            val_acc      = 100. * val_correct   / val_total
            val_loss_avg = val_loss / val_total
            scheduler.step()

            print(f"Epoch [{epoch+1:02d}/{epochs}] "
                  f"| Train Acc: {train_acc:.2f}% "
                  f"| Val Acc: {val_acc:.2f}% "
                  f"| Val Loss: {val_loss_avg:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                no_improve    = 0
                torch.save(self.model.state_dict(), 'best_edge_cwt_model.pth')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[Early stop] 개선 없음 {patience} epochs — 학습 종료.")
                    break

        print(f"[+] 학습 완료. Best val loss: {best_val_loss:.4f} → 'best_edge_cwt_model.pth' 저장됨.")

# =========================================================================
# [Layer 4b] Jetson Inference Engine (binocular-aware)
# Pure TensorRT Python bindings with cuda-python; falls back to PyTorch
# when no .engine file is present.
# =========================================================================
class JetsonInferenceEngine:
    """
    Loads a serialised TensorRT engine (INT8+DLA or FP16) and runs batch-1
    inference without PyTorch.  Falls back to PyTorch FP32 if the engine
    file is missing so the pipeline still works on machines without TRT.
    """

    ENGINE_PATH = "edge_cwt_model_int8_dla.engine"   # preferred
    _FP16_PATH  = "edge_cwt_model_fp16.engine"        # secondary

    def __init__(self, model_path: str, pipeline_config: dict, device="auto"):
        self.pipeline = EventLockedCWTPipeline(**pipeline_config)
        self.classes  = ["Healthy Control (HC)", "Mild Cognitive Impairment (MCI)"]

        # ── Try TRT path ──────────────────────────────────────────────────
        engine_path = (self.ENGINE_PATH if os.path.exists(self.ENGINE_PATH)
                       else self._FP16_PATH  if os.path.exists(self._FP16_PATH)
                       else None)
        self._trt_ready = False
        if engine_path:
            try:
                import tensorrt as trt
                from cuda import cudart

                self._trt    = trt
                self._cudart = cudart

                logger  = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                with open(engine_path, 'rb') as f:
                    self._engine  = runtime.deserialize_cuda_engine(f.read())
                self._context = self._engine.create_execution_context()

                # Allocate pinned host + device buffers for input and output
                in_shape  = (1, 4, 40, 100)
                out_shape = (1, 2)
                self._in_host  = np.zeros(in_shape,  dtype=np.float32)
                self._out_host = np.zeros(out_shape, dtype=np.float32)
                _, self._in_dev  = cudart.cudaMalloc(self._in_host.nbytes)
                _, self._out_dev = cudart.cudaMalloc(self._out_host.nbytes)

                # TRT 10 uses named I/O tensors
                self._in_name  = self._engine.get_tensor_name(0)
                self._out_name = self._engine.get_tensor_name(1)
                self._context.set_tensor_address(self._in_name,  int(self._in_dev))
                self._context.set_tensor_address(self._out_name, int(self._out_dev))

                self._trt_ready = True
                print(f"[+] TRT engine loaded: {engine_path}")
            except Exception as e:
                print(f"[!] TRT init failed ({e}) — falling back to PyTorch")

        # ── PyTorch fallback ──────────────────────────────────────────────
        if not self._trt_ready:
            if torch.cuda.is_available():            self._pt_device = torch.device("cuda")
            elif torch.backends.mps.is_available():  self._pt_device = torch.device("mps")
            else:                                    self._pt_device = torch.device("cpu")
            self._pt_model = EdgeCWTClassifier(num_classes=2, in_channels=4)
            if os.path.exists(model_path):
                self._pt_model.load_state_dict(
                    torch.load(model_path, map_location=self._pt_device, weights_only=True)
                )
            self._pt_model.to(self._pt_device).eval()

    # ── internal helpers ──────────────────────────────────────────────────
    @staticmethod
    def _normalize_tensor(tensor: np.ndarray) -> np.ndarray:
        out = np.empty_like(tensor)
        for ch in range(tensor.shape[0]):
            d = tensor[ch]
            out[ch] = (d - np.mean(d)) / (np.std(d) + 1e-8)
        return out

    def _infer_trt(self, input_arr: np.ndarray) -> np.ndarray:
        """Run one sample through TRT and return softmax probabilities [2]."""
        from scipy.special import softmax as scipy_softmax
        np.copyto(self._in_host, input_arr.reshape(1, 4, 40, 100))
        self._cudart.cudaMemcpy(
            self._in_dev, self._in_host.ctypes.data,
            self._in_host.nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        self._context.execute_async_v3(0)
        self._cudart.cudaMemcpy(
            self._out_host.ctypes.data, self._out_dev,
            self._out_host.nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        return scipy_softmax(self._out_host[0].astype(np.float32))

    def _infer_pytorch(self, input_arr: np.ndarray) -> np.ndarray:
        t = torch.tensor(input_arr[None], dtype=torch.float32).to(self._pt_device)
        with torch.no_grad():
            logits = self._pt_model(t)
        return torch.softmax(logits[0], dim=0).cpu().numpy()

    # ── public API ────────────────────────────────────────────────────────
    def infer_csv(self, csv_filepath: Path):
        print(f"\n[*] 추론 시작: {csv_filepath.name}")
        df = self.pipeline._load_csv_safely(csv_filepath)

        axis_char  = 'h' if 'Horizontal' in csv_filepath.name else 'v'
        target_col = next(
            (c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c), None
        )
        left_col  = next((c for c in df.columns if c == f'l{axis_char}'), None)
        right_col = next((c for c in df.columns if c == f'r{axis_char}'), None)

        if not target_col or not left_col or not right_col:
            return "추론 불가: Target 또는 Eye 컬럼을 찾을 수 없습니다."

        time_col   = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
        time_val   = df[time_col].dropna().values
        current_fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else 120.0

        cwt_epochs = self.pipeline._extract_binocular_epochs(
            df, target_col, left_col, right_col, current_fs
        )
        if not cwt_epochs:
            return "추론 불가: 유효한 Saccade 이벤트가 없습니다."

        probs_list = []
        for tensor in cwt_epochs:
            norm = self._normalize_tensor(tensor)
            if self._trt_ready:
                probs_list.append(self._infer_trt(norm))
            else:
                probs_list.append(self._infer_pytorch(norm))

        mean_prob       = np.mean(probs_list, axis=0)
        predicted_class = int(np.argmax(mean_prob))
        mci_confidence  = float(mean_prob[1]) * 100

        print(f"[>] Saccade 이벤트 수: {len(cwt_epochs)}")
        print(f"[>] 앙상블 진단: {self.classes[predicted_class]}")
        print(f"[>] MCI 확률: {mci_confidence:.2f}%")
        return predicted_class, mci_confidence

    def __del__(self):
        if self._trt_ready and hasattr(self, '_cudart'):
            if hasattr(self, '_in_dev'):  self._cudart.cudaFree(self._in_dev)
            if hasattr(self, '_out_dev'): self._cudart.cudaFree(self._out_dev)

# =========================================================================
# [Layer 5] XAI Visualizer: SPM-style Difference Maps
# Updated for 3-level data_store + 4-channel binocular tensors
# Power = mean of left-eye power + right-eye power
# =========================================================================
class XAIVisualizer:
    def __init__(self, pipeline: EventLockedCWTPipeline):
        self.frequencies = pipeline.frequencies
        self.time_bins   = pipeline.target_time_bins
        self.pre_sec     = pipeline.pre_sec
        self.post_sec    = pipeline.post_sec

    def _compute_group_mean_db(self, data_store):
        # Aggregate per-subject mean powers (independent observations for t-test)
        group_subject_powers = defaultdict(list)
        for group, subjects in data_store.items():
            for subject_id, tasks in subjects.items():
                subject_epochs = []
                for task, tensors in tasks.items():
                    for tensor in tensors:            # [4, freq, time]
                        power = (tensor[0]**2 + tensor[1]**2 +
                                 tensor[2]**2 + tensor[3]**2) / 2.0
                        subject_epochs.append(power)
                if subject_epochs:
                    group_subject_powers[group].append(np.mean(subject_epochs, axis=0))

        group_mean_db = {
            group: 10 * np.log10(np.mean(np.array(powers), axis=0) + 1e-10)
            for group, powers in group_subject_powers.items()
        }
        return group_mean_db, {g: np.array(p) for g, p in group_subject_powers.items()}

    def plot_difference_map(self, data_store, save_path=None):
        group_mean_db, group_subject_powers = self._compute_group_mean_db(data_store)
        if 'HC' not in group_mean_db or 'MCI' not in group_mean_db:
            print("[!] HC와 MCI 데이터가 모두 필요합니다.")
            return

        # Pixel-wise Welch t-test across per-subject means
        _, p_values    = ttest_ind(group_subject_powers['MCI'],
                                   group_subject_powers['HC'], axis=0, equal_var=False)
        sig_mask       = p_values < 0.05
        diff_map       = group_mean_db['MCI'] - group_mean_db['HC']
        masked_diff    = np.where(sig_mask, diff_map, np.nan)

        time_axis = np.linspace(-self.pre_sec, self.post_sec, self.time_bins)
        extent    = [time_axis[0], time_axis[-1], self.frequencies[0], self.frequencies[-1]]
        vmax      = np.abs(diff_map).max()

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        configs = [
            (group_mean_db['HC'], 'HC Mean CWT (dB)',               'viridis', None,  None),
            (group_mean_db['MCI'],'MCI Mean CWT (dB)',              'viridis', None,  None),
            (diff_map,             'Difference: MCI − HC (dB)', 'RdBu_r', -vmax, vmax),
            (masked_diff,          'Significant Regions (p<0.05)',   'RdBu_r', -vmax, vmax),
        ]
        for ax, (data, title, cmap, vmin, vmax_) in zip(axes, configs):
            im = ax.imshow(data, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.axvline(x=0, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('dB' if 'Difference' not in title and 'Significant' not in title else 'ΔdB')

        sig_pct = sig_mask.mean() * 100
        n_hc  = len(group_subject_powers['HC'])
        n_mci = len(group_subject_powers['MCI'])
        print(f"[XAI] Significant pixels: {sig_pct:.1f}% (p<0.05, Welch t-test, n_HC={n_hc}, n_MCI={n_mci})")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Difference Map 저장됨: {save_path}")
        plt.show()

# =========================================================================
# [Layer 6] LOSO-CV Evaluator (enhanced)
# Enhancements:
#   - AugmentedSubset applied to each training fold
#   - Label smoothing (0.1) in CrossEntropyLoss
#   - Default epochs raised to 50
#   - Threshold optimisation: finds threshold maximising balanced accuracy
# =========================================================================
class LOSOCrossValidator:
    def __init__(self, dataset, device="auto", epochs=50, batch_size=32):
        if device in ("auto", "cuda"):
            if torch.cuda.is_available():            self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():  self.device = torch.device("mps")
            else:                                    self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dataset    = dataset
        self.epochs     = epochs
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    @staticmethod
    def _auroc(true_labels, scores):
        order    = np.argsort(scores)[::-1]
        y_sorted = true_labels[order]
        n_pos    = np.sum(true_labels == 1)
        n_neg    = np.sum(true_labels == 0)
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        tp = fp = 0
        tprs, fprs = [0.0], [0.0]
        for label in y_sorted:
            if label == 1: tp += 1
            else:          fp += 1
            tprs.append(tp / n_pos)
            fprs.append(fp / n_neg)
        return float(np.trapezoid(tprs, fprs))

    # ------------------------------------------------------------------
    def _train_one_fold(self, train_idx):
        train_raw    = Subset(self.dataset, train_idx)
        # SpecAugment on training fold only
        train_subset = AugmentedSubset(train_raw)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True)

        model     = EdgeCWTClassifier(num_classes=2, in_channels=4).to(self.device)
        use_amp   = self.device.type == "cuda"
        scaler    = torch.amp.GradScaler('cuda') if use_amp else None
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        label_counts  = Counter(self.dataset.y[train_idx].tolist())
        total         = len(train_idx)
        alpha = torch.tensor(
            [total / (2 * label_counts[i]) for i in range(2)], dtype=torch.float32
        )
        alpha = (alpha / alpha.sum()).to(self.device)   # normalize → sums to 1
        criterion = FocalLoss(alpha=alpha, gamma=1.5)

        for _ in range(self.epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    loss = criterion(model(inputs), labels)
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            scheduler.step()
        return model

    # ------------------------------------------------------------------
    def _infer_subject(self, model, test_idx):
        test_loader = DataLoader(Subset(self.dataset, test_idx),
                                 batch_size=self.batch_size, shuffle=False)
        all_probs = []
        model.eval()
        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    probs = F.softmax(model(inputs), dim=1)
                all_probs.append(probs.cpu())
        mean_prob = torch.cat(all_probs, dim=0).mean(dim=0)
        return torch.argmax(mean_prob).item(), mean_prob[1].item()

    # ------------------------------------------------------------------
    def run(self):
        subject_to_idx = defaultdict(list)
        for i, sid in enumerate(self.dataset.subject_ids):
            subject_to_idx[sid].append(i)

        all_subjects = list(subject_to_idx.keys())
        N            = len(all_subjects)
        all_indices  = set(range(len(self.dataset)))

        records = []
        print(f"[*] LOSO-CV 시작 — {N} subjects, device={self.device}\n")

        for fold, subject in enumerate(all_subjects):
            test_idx   = subject_to_idx[subject]
            train_idx  = list(all_indices - set(test_idx))
            true_label = self.dataset.y[test_idx[0]].item()

            model = self._train_one_fold(train_idx)
            pred_label, mci_prob = self._infer_subject(model, test_idx)

            mark     = "✓" if true_label == pred_label else "✗"
            true_str = "MCI" if true_label else "HC "
            pred_str = "MCI" if pred_label else "HC "
            print(f"  [{fold+1:02d}/{N}] {mark}  True={true_str}  Pred={pred_str}  "
                  f"MCI_prob={mci_prob:.3f}  epochs={len(test_idx):4d}  ({subject[:24]})")
            records.append((subject, true_label, pred_label, mci_prob, len(test_idx)))

        return self._report(records)

    # ------------------------------------------------------------------
    def _report(self, records):
        _, true_arr, pred_arr, prob_arr, _ = zip(*records)
        true_arr = np.array(true_arr)
        pred_arr = np.array(pred_arr)
        prob_arr = np.array(prob_arr)

        def _metrics(ta, pa):
            tp = int(np.sum((ta==1)&(pa==1))); tn = int(np.sum((ta==0)&(pa==0)))
            fp = int(np.sum((ta==0)&(pa==1))); fn = int(np.sum((ta==1)&(pa==0)))
            acc  = (tp+tn)/len(ta)
            sens = tp/(tp+fn) if tp+fn>0 else 0.0
            spec = tn/(tn+fp) if tn+fp>0 else 0.0
            return tp, tn, fp, fn, acc, sens, spec

        tp, tn, fp, fn, accuracy, sensitivity, specificity = _metrics(true_arr, pred_arr)
        balanced_acc = (sensitivity + specificity) / 2
        auroc        = self._auroc(true_arr, prob_arr)

        # ── Threshold optimisation ────────────────────────────────────
        best_t, best_bacc = 0.5, balanced_acc
        for t in np.linspace(0.1, 0.9, 81):
            pa_t = (prob_arr > t).astype(int)
            _, _, _, _, _, s_, sp_ = _metrics(true_arr, pa_t)
            b_ = (s_ + sp_) / 2
            if b_ > best_bacc:
                best_bacc = b_; best_t = t
        pa_opt = (prob_arr > best_t).astype(int)
        tp_o, tn_o, fp_o, fn_o, acc_o, sens_o, spec_o = _metrics(true_arr, pa_opt)

        print("\n" + "=" * 56)
        print("  LOSO-CV  Summary")
        print("=" * 56)
        n_hc  = int(np.sum(true_arr == 0))
        n_mci = int(np.sum(true_arr == 1))
        print(f"  Subjects     : {len(records)}  (HC={n_hc}, MCI={n_mci})")
        print(f"  ── Threshold = 0.50 (default) ──")
        print(f"  Accuracy     : {accuracy:.3f}   ({tp+tn}/{len(records)})")
        print(f"  Sensitivity  : {sensitivity:.3f}   TP={tp}  FN={fn}")
        print(f"  Specificity  : {specificity:.3f}   TN={tn}  FP={fp}")
        print(f"  Balanced Acc : {balanced_acc:.3f}")
        print(f"  AUROC        : {auroc:.3f}")
        print(f"  ── Threshold = {best_t:.2f} (optimised for balanced acc) ──")
        print(f"  Accuracy     : {acc_o:.3f}   ({tp_o+tn_o}/{len(records)})")
        print(f"  Sensitivity  : {sens_o:.3f}   TP={tp_o}  FN={fn_o}")
        print(f"  Specificity  : {spec_o:.3f}   TN={tn_o}  FP={fp_o}")
        print(f"  Balanced Acc : {best_bacc:.3f}")
        print("=" * 56)

        # ── ROC curve + probability strip ────────────────────────────
        order    = np.argsort(prob_arr)[::-1]
        y_sorted = true_arr[order]
        n_pos    = int(np.sum(true_arr == 1))
        n_neg    = int(np.sum(true_arr == 0))
        tp_r = fp_r = 0
        tprs, fprs = [0.0], [0.0]
        for lbl in y_sorted:
            if lbl == 1: tp_r += 1
            else:        fp_r += 1
            tprs.append(tp_r / n_pos); fprs.append(fp_r / n_neg)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(fprs, tprs, 'b-o', markersize=5, label=f'AUROC = {auroc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
        axes[0].fill_between(fprs, tprs, alpha=0.1)
        axes[0].set_xlabel('1 − Specificity (FPR)', fontsize=12)
        axes[0].set_ylabel('Sensitivity (TPR)', fontsize=12)
        axes[0].set_title('LOSO-CV ROC Curve', fontsize=13)
        axes[0].legend(fontsize=11); axes[0].grid(alpha=0.3)

        hc_probs  = prob_arr[true_arr == 0]
        mci_probs = prob_arr[true_arr == 1]
        axes[1].scatter(hc_probs,  np.zeros_like(hc_probs)  + 0.15,
                        color='steelblue', s=90, label=f'HC  (n={n_neg})', zorder=3, alpha=0.85)
        axes[1].scatter(mci_probs, np.zeros_like(mci_probs) + 0.85,
                        color='tomato',    s=90, label=f'MCI (n={n_pos})', zorder=3, alpha=0.85)
        axes[1].axvline(0.5,    color='gray',   linestyle='--', linewidth=1.2, label='default=0.5')
        axes[1].axvline(best_t, color='orange', linestyle=':',  linewidth=1.5,
                        label=f'optimal={best_t:.2f}')
        axes[1].set_xlim(-0.05, 1.05); axes[1].set_ylim(0, 1)
        axes[1].set_yticks([0.15, 0.85]); axes[1].set_yticklabels(['HC', 'MCI'], fontsize=12)
        axes[1].set_xlabel('Soft-vote MCI probability', fontsize=12)
        axes[1].set_title('Per-subject MCI probability', fontsize=13)
        axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('../loso_results.png', dpi=150, bbox_inches='tight')
        print("[+] Saved: ../loso_results.png")
        plt.show()

        return dict(accuracy=accuracy, sensitivity=sensitivity, specificity=specificity,
                    balanced_accuracy=balanced_acc, auroc=auroc,
                    opt_threshold=best_t, opt_balanced_acc=best_bacc,
                    records=records)

def save_calibration_data(dataset, out_path: str = "int8_calibration_data.npy", n: int = 200):
    """Save up to n random CWT tensors from dataset for INT8 TRT calibration."""
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    tensors = np.stack([dataset[int(i)][0].numpy() for i in indices], axis=0)
    np.save(out_path, tensors)
    print(f"[+] Calibration data saved: {out_path}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    DATA_DIR = Path("../data")

    pipeline_config = {
        "pre_stimulus_sec":  0.2,
        "post_stimulus_sec": 0.8,
        "freq_bins":         40,      # 25 low-band + 15 high-band bins
        "dual_band":         True,    # 0.5-10Hz (cognitive delay) + 30-60Hz (micro-tremors)
        "target_time_bins":  100,
        "w_morlet":          5.0,
        "artifact_threshold": 30.0,
    }
    pipeline = EventLockedCWTPipeline(**pipeline_config)

    if not DATA_DIR.exists():
        print(f"[!] 데이터 경로가 존재하지 않습니다: {DATA_DIR}")
    else:
        print(f"[*] 데이터 로드 중: {DATA_DIR.resolve()}")
        pipeline.process_directory(DATA_DIR)

        if not pipeline.data_store:
            print("[!] 처리된 데이터가 없습니다. 경로 및 파일명을 확인하세요.")
        else:
            # --- XAI: Difference Maps (before training) ---
            visualizer = XAIVisualizer(pipeline)
            visualizer.plot_difference_map(pipeline.data_store, save_path="../xai_difference_map.png")

            # --- Dataset ---
            dataset = VOG_CWT_Dataset(pipeline.data_store)
            n_hc  = (dataset.y == 0).sum().item()
            n_mci = (dataset.y == 1).sum().item()
            print(f"[*] 총 {len(dataset)}개 CWT 샘플 (HC: {n_hc}, MCI: {n_mci})")

            # --- Save calibration data for INT8 TRT export ---
            save_calibration_data(dataset, "int8_calibration_data.npy", n=200)

            # --- Single-split Training (quick baseline + saves best checkpoint) ---
            model   = EdgeCWTClassifier(num_classes=2, in_channels=4)
            trainer = ModelTrainer(model)
            trainer.train_model(dataset, epochs=50, batch_size=32, patience=10)

            # --- LOSO-CV (rigorous evaluation) ---
            loso = LOSOCrossValidator(dataset, epochs=50, batch_size=32)
            loso_results = loso.run()

            # --- Inference example ---
            engine = JetsonInferenceEngine('best_edge_cwt_model.pth', pipeline_config)
            sample_files = list(DATA_DIR.rglob('*.csv'))
            if sample_files:
                engine.infer_csv(sample_files[0])
