"""
TensorRT export and inference benchmark for EdgeCWTClassifier.

Steps:
  1. Load best_edge_cwt_model.pth
  2. Export to ONNX (opset 17)
  3. Build TensorRT FP16 engine via trtexec
  4. Benchmark:  plain PyTorch FP32 vs FP16 vs TensorRT FP16
                 metric: epochs/sec (batch=1, batch=32)
"""
import os, time, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── model definition (must match training code) ────────────────────────────
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
        avg   = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x,  dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        return self.sa(self.ca(x))

class EdgeCWTClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.cbam1 = CBAM(16)
        self.block2_main = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
        )
        self.block2_skip = nn.Sequential(
            nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
        )
        self.cbam2 = CBAM(32)
        self.block3_main = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64),
        )
        self.block3_skip = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64),
        )
        self.cbam3 = CBAM(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, num_classes),
        )
    def forward(self, x):
        x = self.cbam1(self.block1(x))
        x = self.cbam2(F.relu(self.block2_main(x) + self.block2_skip(x), inplace=True))
        x = self.cbam3(F.relu(self.block3_main(x) + self.block3_skip(x), inplace=True))
        return self.classifier(torch.flatten(self.gap(x), 1))


# ── helpers ─────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "best_edge_cwt_model.pth"
ONNX_PATH   = "edge_cwt_model.onnx"
TRT_PATH    = "edge_cwt_model_fp16.engine"
INPUT_SHAPE = (1, 4, 40, 100)   # single-epoch inference shape
WARMUP      = 50
RUNS        = 500


def load_model():
    m = EdgeCWTClassifier().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def benchmark_torch(model, batch_size, dtype=torch.float32, label=""):
    x = torch.randn(batch_size, 4, 40, 100, dtype=dtype, device=DEVICE)
    use_amp = (dtype == torch.float16)
    with torch.no_grad():
        for _ in range(WARMUP):
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    eps = (RUNS * batch_size) / elapsed
    print(f"  {label:35s}  {eps:9.1f} epochs/sec  ({elapsed*1000/RUNS:.3f} ms/batch)")
    return eps


def export_onnx(model):
    model.eval()
    dummy = torch.randn(*INPUT_SHAPE, device=DEVICE)
    # Export with fixed batch=1 (no dynamic axes) so TRT can resolve
    # the Shape/Gather/Reshape pattern from ChannelAttention.view(b,c)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
    )
    print(f"[+] ONNX exported → {ONNX_PATH}")


def build_trt_engine():
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        trtexec = "trtexec"
    cmd = [
        trtexec,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={TRT_PATH}",
        "--fp16",
        "--memPoolSize=workspace:4096MiB",
        "--verbose=false",
    ]
    print(f"[*] Building TRT engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] trtexec failed:")
        print(result.stderr[-2000:])
        return False
    print(f"[+] TRT engine saved → {TRT_PATH}")
    return True


def benchmark_trt_via_trtexec(batch_size):
    """Use trtexec --loadEngine for timing (no pycuda needed)."""
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        trtexec = "trtexec"
    cmd = [
        trtexec,
        f"--loadEngine={TRT_PATH}",
        "--shapes=input:1x4x40x100",
        f"--iterations={RUNS}",
        "--warmUp=200",
        "--fp16",
        "--noDataTransfers",
        "--useCudaGraph",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    # Parse "mean = X.XX ms" from trtexec output
    import re
    m = re.search(r"mean\s*=\s*([\d.]+)\s*ms", output)
    if m:
        mean_ms = float(m.group(1))
        eps = (batch_size * 1000) / mean_ms
        label = f"TRT FP16  batch={batch_size}"
        print(f"  {label:35s}  {eps:9.1f} epochs/sec  ({mean_ms:.3f} ms/batch)")
        return eps
    else:
        print(f"  TRT FP16  batch={batch_size}: could not parse trtexec output")
        if result.returncode != 0:
            print(result.stderr[-500:])
        return None


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[*] Device: {DEVICE}  |  CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"[*] Model:  {MODEL_PATH}")
    print()

    # ── Step 1: PyTorch benchmarks ──────────────────────────────────────────
    model = load_model()
    print("── PyTorch benchmarks ─────────────────────────────────────────────")
    for bs in [1, 32]:
        benchmark_torch(model, bs, torch.float32, f"PyTorch FP32  batch={bs}")
    for bs in [1, 32]:
        benchmark_torch(model, bs, torch.float16, f"PyTorch FP16  batch={bs}")
    print()

    # ── Step 2: ONNX export ─────────────────────────────────────────────────
    print("── ONNX export ────────────────────────────────────────────────────")
    export_onnx(model)
    print()

    # ── Step 3: TensorRT engine build ───────────────────────────────────────
    print("── TensorRT build ─────────────────────────────────────────────────")
    trt_ok = build_trt_engine()
    print()

    # ── Step 4: TRT benchmarks (if build succeeded) ─────────────────────────
    if trt_ok and os.path.exists(TRT_PATH):
        print("── TensorRT benchmarks ────────────────────────────────────────────")
        for bs in [1, 32]:
            benchmark_trt_via_trtexec(bs)
        print()

    print("[+] Done.")
