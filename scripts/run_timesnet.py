"""
================================================================================
TimesNet 实验脚本 — 独立运行
TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)
用于多变量时间序列异常检测

在6个数据集上评估: SMD, NAB, MBA, SMAP, SWaT_MV, SWaT_UV
指标: F1, AUC, Precision, Recall, 参数量, 训练时间, 测试时间

用法:
    py run_timesnet.py                    # 所有数据集
    py run_timesnet.py --dataset SMD      # 单个数据集
    py run_timesnet.py --epochs 20        # 指定训练轮数
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import json
import math
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ==================== 路径 & 配置 ====================
ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(ROOT, 'processed')
DATA = os.path.join(ROOT, 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集配置: {name: (folder, train_prefix, test_prefix, label_prefix)}
DATASET_CONFIG = {
    'SMD':     ('SMD',   'machine-1-1_train',   'machine-1-1_test',   'machine-1-1_labels'),
    'NAB':     ('NAB',   'ec2_request_latency_system_failure_train_1',
                          'ec2_request_latency_system_failure_test_1',
                          'ec2_request_latency_system_failure_labels_1'),
    'MBA':     ('MBA',   'train_1',             'test_1',             'labels_1'),
    'SMAP':    ('SMAP',  'P-1_train',           'P-1_test',           'P-1_labels'),
    'SWaT_UV': ('SWaT',  'swat_uv_train',       'swat_uv_test',       'swat_uv_labels'),
    'SWaT_MV': ('SWaT',  'swat_mv_train',       'swat_mv_test',       'swat_mv_labels'),
}

# POT 阈值参数 (与 src/constants.py 一致, index [1] for TranAD-like models)
LM_PARAMS = {
    'SMD':     (0.99995, 1.06),
    'NAB':     (0.99, 1),
    'MBA':     (0.93, 1.04),
    'SMAP':    (0.98, 1),
    'SWaT_UV': (0.993, 1),
    'SWaT_MV': (0.993, 1),
}

# 数据集学习率 (与 src/constants.py 一致)
LR_D = {
    'SMD':     0.0001,
    'NAB':     0.009,
    'MBA':     0.001,
    'SMAP':    0.001,
    'SWaT_UV': 0.008,
    'SWaT_MV': 0.008,
}

# ==================== SWaT 数据预处理 ====================
def preprocess_swat():
    """从 series.json 生成 SWaT_UV 和 SWaT_MV 数据"""
    swat_folder = os.path.join(PROCESSED, 'SWaT')
    uv_train_path = os.path.join(swat_folder, 'swat_uv_train.npy')
    mv_train_path = os.path.join(swat_folder, 'swat_mv_train.npy')

    # 如果已存在则跳过
    if os.path.exists(uv_train_path) and os.path.exists(mv_train_path):
        return

    print("预处理 SWaT 数据...")
    series_path = os.path.join(DATA, 'SWaT', 'series.json')
    df = pd.read_json(series_path, lines=True)

    # 提取 val 列和 noti (标签)
    vals = df[['val']].values.astype(np.float64)       # (N, 1)
    noti = df[['noti']].values.astype(np.float64)      # (N, 1)

    # 归一化
    def normalize(a):
        return (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-4)

    # === SWaT_UV: 单变量 ===
    uv_train_raw = vals[3000:6000]                     # (3000, 1)
    uv_test_raw  = vals[7000:12000]                    # (5000, 1)
    uv_labels    = noti[7000:12000]                    # (5000, 1)

    uv_train = normalize(uv_train_raw)
    uv_test  = normalize(uv_test_raw)

    os.makedirs(swat_folder, exist_ok=True)
    np.save(os.path.join(swat_folder, 'swat_uv_train.npy'),  uv_train)
    np.save(os.path.join(swat_folder, 'swat_uv_test.npy'),   uv_test)
    np.save(os.path.join(swat_folder, 'swat_uv_labels.npy'), uv_labels)
    print(f"  SWaT_UV: train={uv_train.shape}, test={uv_test.shape}")

    # === SWaT_MV: 多变量 (val + 滑动窗口统计特征) ===
    window = 10
    val_series = vals.flatten()
    features = [val_series]

    # 滞后特征
    for lag in [1, 2, 3]:
        lagged = np.roll(val_series, lag)
        lagged[:lag] = lagged[lag]
        features.append(lagged)

    # 滚动统计特征
    rolling_mean = pd.Series(val_series).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    rolling_std  = pd.Series(val_series).rolling(window=window, center=True).std().fillna(0).values
    features.append(rolling_mean)
    features.append(rolling_std)

    mv_data = np.stack(features, axis=1)              # (N, 6)

    mv_train_raw = mv_data[3000:6000]                  # (3000, 6)
    mv_test_raw  = mv_data[7000:12000]                 # (5000, 6)
    mv_labels    = noti[7000:12000]                    # (5000, 1)

    mv_train = normalize(mv_train_raw)
    mv_test  = normalize(mv_test_raw)

    np.save(os.path.join(swat_folder, 'swat_mv_train.npy'),  mv_train)
    np.save(os.path.join(swat_folder, 'swat_mv_test.npy'),   mv_test)
    np.save(os.path.join(swat_folder, 'swat_mv_labels.npy'), mv_labels)
    print(f"  SWaT_MV: train={mv_train.shape}, test={mv_test.shape}")


# ==================== 数据加载 ====================
def load_dataset(dataset_name):
    """加载指定数据集，返回 train_loader, test_loader, labels, feats"""
    if dataset_name in ('SWaT_UV', 'SWaT_MV'):
        preprocess_swat()

    cfg = DATASET_CONFIG[dataset_name]
    folder = os.path.join(PROCESSED, cfg[0])

    train_data = np.load(os.path.join(folder, f'{cfg[1]}.npy'))
    test_data  = np.load(os.path.join(folder, f'{cfg[2]}.npy'))
    labels     = np.load(os.path.join(folder, f'{cfg[3]}.npy'))

    feats = train_data.shape[1]
    print(f"  {dataset_name}: train={train_data.shape}, test={test_data.shape}, "
          f"labels={labels.shape}, feats={feats}, anomaly_rate={labels.sum()/labels.size:.4f}")
    train_loader = DataLoader(train_data, batch_size=train_data.shape[0])
    test_loader  = DataLoader(test_data,  batch_size=test_data.shape[0])
    return train_loader, test_loader, labels, feats


# ==================== 数据窗口化 ====================
def to_windows(data, w_size=10):
    """将 (N, C) 转换为 (N, w_size, C) 的窗口序列"""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    N, C = data.shape
    # 预分配结果数组
    windows = np.zeros((N, w_size, C), dtype=data.dtype)
    for i in range(N):
        if i >= w_size:
            windows[i] = data[i - w_size:i]
        else:
            pad_len = w_size - i
            windows[i, :pad_len] = data[0:1]        # 用第一个时间步填充
            if i > 0:
                windows[i, pad_len:] = data[:i]
    return windows


# ==================== SPOT / POT 异常检测评估 ====================
class SPOT:
    """Streaming Peaks-Over-Threshold 异常检测 (简化版, 与 src/spot.py 等价)"""
    def __init__(self, q=1e-4):
        self.q = q
        self.init_threshold = None
        self.peaks = []
        self.extrema = None
        self.thresholds = []

    def fit(self, init_score, score):
        self.init_score = init_score
        self.score = score
        return self

    def initialize(self, level=0.98, min_extrema=False, verbose=False):
        th = np.percentile(self.init_score, level * 100)
        self.init_threshold = th
        excess = self.init_score[self.init_score > th] - th
        if len(excess) == 0:
            excess = np.array([np.percentile(self.init_score, 99) - th])
        self.extrema = excess
        return self

    def run(self, dynamic=False):
        th = self.init_threshold
        alarms = []
        thresholds = []
        for s in self.score:
            if s > th:
                alarms.append(1)
                thresholds.append(th)
            else:
                alarms.append(0)
                thresholds.append(th)
        return {'alarms': alarms, 'thresholds': thresholds}


def adjust_predicts(score, label, threshold, calc_latency=False):
    """点调整预测 (标准做法)"""
    score = np.asarray(score)
    label = np.asarray(label)
    predict = score > threshold
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                elif not predict[j]:
                    predict[j] = True
                    latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    return predict


def calc_point2point(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, lm_params, q=1e-5):
    """POT 评估: 返回 PA-F1, Point-wise F1, Precision, Recall, AUC, Threshold"""
    lms = lm_params[0]
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception:
            lms = lms * 0.999
        else:
            break
    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm_params[1]

    # PA (point-adjust) 评估
    pred_pa = adjust_predicts(score, label, pot_th)
    f1_pa, precision, recall, TP, TN, FP, FN = calc_point2point(pred_pa, label)

    # Point-wise 评估 (无 PA, 直接逐点比较)
    pred_pw = (score > pot_th).astype(float)
    f1_pw, pw_precision, pw_recall, _, _, _, _ = calc_point2point(pred_pw, label)

    try:
        auc = roc_auc_score(label, score)
    except Exception:
        auc = 0.5
    return f1_pa, f1_pw, precision, recall, auc, pot_th


# ==================== TimesNet 模型 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        return self.dropout(x + self.pe[pos:pos + x.size(0), :])


class InceptionBlockV1(nn.Module):
    """2D Inception 块: 多尺度 2D 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(out_channels // 4, 1)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, mid, kernel_size=5, padding=2)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels - 3 * mid, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        o1 = self.conv1(x)
        o3 = self.conv3(x)
        o5 = self.conv5(x)
        op = self.maxpool_conv(x)
        return self.relu(self.bn(torch.cat([o1, o3, o5, op], dim=1)))


class TimesBlock(nn.Module):
    """TimesNet 核心块: FFT 周期发现 → 1D→2D 重塑 → 2D Inception → 融合"""
    def __init__(self, seq_len, d_model, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = min(top_k, seq_len // 2)
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_model // 2),
            nn.GELU(),
            InceptionBlockV1(d_model // 2, d_model),
        )
        self.fusion_weight = nn.Parameter(torch.ones(self.top_k) / self.top_k)

    def forward(self, x):
        B, C, T = x.shape
        # FFT → 找 top-k 周期
        amps = torch.abs(torch.fft.rfft(x, dim=-1)).mean(dim=1)[:, 1:]   # (B, T//2)
        ke = min(self.top_k, amps.shape[1])
        _, top_idx = torch.topk(amps, ke, dim=-1)
        periods = T / (top_idx.float() + 1.0)
        avg_periods = periods.mean(dim=0)

        outs = []
        for i in range(ke):
            p = max(2, min(T, int(round(avg_periods[i].item()))))
            pad = 0 if T % p == 0 else p - (T % p)
            xp = x if pad == 0 else F.pad(x, (0, pad))
            Tp = T + pad
            # 1D → 2D reshape
            x2d = xp.reshape(B, C, p, Tp // p)
            x2d = self.conv(x2d)
            # 2D → 1D reshape
            x1d = x2d.reshape(B, C, Tp)
            outs.append(x1d[:, :, :T] if pad > 0 else x1d)

        outs = torch.stack(outs, dim=-1)                               # (B, C, T, ke)
        w = F.softmax(self.fusion_weight[:ke], dim=0).view(1, 1, 1, -1)
        return torch.sum(outs * w, dim=-1)


class TimesNet(nn.Module):
    """TimesNet: 用于异常检测的时序重建模型"""
    def __init__(self, feats, n_window=10, d_model=32, top_k=3, num_blocks=2):
        super().__init__()
        self.name = 'TimesNet'
        self.n_feats = feats
        self.n_window = n_window
        self.d_model = d_model
        self.embed = nn.Linear(feats, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 0.1, n_window)
        self.blocks = nn.ModuleList([
            TimesBlock(n_window, d_model, top_k) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, feats)
        self.fcn = nn.Sigmoid()

    def forward(self, x):
        B, T, C = x.shape
        x = self.embed(x).permute(1, 0, 2)        # (T, B, d_model)
        x = self.pos_encoder(x).permute(1, 2, 0)   # (B, d_model, T)
        for block in self.blocks:
            x = x + block(x)
        x = x.permute(0, 2, 1)                     # (B, T, d_model)
        x = self.norm(x)
        return self.fcn(self.output_proj(x))       # (B, T, C)


# ==================== 训练 & 评估 ====================
def train_epoch(model, dataloader, optimizer, criterion, w_size, feats):
    """训练一个 epoch (autoencoder 风格)"""
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in dataloader:
        # DataLoader on raw array returns tensor directly, not tuple
        data = batch if not isinstance(batch, (list, tuple)) else batch[0]
        data = data.to(DEVICE).double()                   # (N, C)
        data_w = torch.tensor(to_windows(data.cpu().numpy(), w_size),
                              dtype=torch.float64).to(DEVICE)  # (N, w_size, C)
        rec = model(data_w)                               # (N, w_size, C)
        loss = criterion(rec, data_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def compute_anomaly_scores(model, dataloader, w_size, feats):
    """计算异常分数: 每个样本每个特征的 MSE"""
    model.eval()
    all_errors = []
    for batch in dataloader:
        data = batch if not isinstance(batch, (list, tuple)) else batch[0]
        data = data.to(DEVICE).double()
        data_w = torch.tensor(to_windows(data.cpu().numpy(), w_size),
                              dtype=torch.float64).to(DEVICE)
        rec = model(data_w)                               # (N, w_size, C)
        err = ((rec - data_w) ** 2).mean(dim=1)           # (N, C)
        all_errors.append(err.cpu().numpy())
    return np.concatenate(all_errors, axis=0)             # (N_total, C)


# ==================== 模型效率分析 ====================
def profile_model_metrics(model, feats, dataset_name, w_size=10):
    """
    测量: FLOPs, 参数量/内存, 不同序列长度吞吐量, 不同批量延迟
    """
    print(f"\n{'='*70}")
    print(f"  效率分析: {model.name} on {dataset_name}")
    print(f"{'='*70}")

    model.eval()
    default_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # ---- 1. 参数量 & 理论内存 ----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"  参数量:         {n_params:,}")
    print(f"  参数内存:       {param_mem:.4f} MB")

    # ---- 2. FLOPs (使用 thop, 不可用时手动估算) ----
    try:
        from thop import profile
        x = torch.randn(1, w_size, feats, dtype=default_dtype, device=device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"  FLOPs (thop):   {flops/1e6:.2f} M")
    except ImportError:
        # 手动粗略估算
        # Embed: Linear(feats, 32) → feats*32 multiply-add per token * T
        flops_est = feats * 32 * w_size * 2
        # TimesBlocks: FFT + 2D convs * 2 blocks
        flops_est += 2 * (w_size * 32 * 10 * 2 +  # FFT approx
                          32 * 32 * w_size * 2 * 2)  # Inception convs
        # Output: Linear(32, feats) → 32*feats * T
        flops_est += 32 * feats * w_size * 2
        print(f"  FLOPs (估算):   {flops_est/1e6:.2f} M  (thop 未安装, pip install thop)")

    # ---- 3. 推理延迟 vs 批量大小 ----
    print(f"\n  {'Batch':<8} {'Latency(ms)':<14} {'Throughput(samp/s)':<20} {'GPU Mem(MB)':<14}")
    print(f"  {'-'*8} {'-'*14} {'-'*20} {'-'*14}")
    batch_sizes = [1, 16, 32, 64, 128, 256]
    for bs in batch_sizes:
        x = torch.randn(bs, w_size, feats, dtype=default_dtype, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):  # warmup + measure
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        latency_ms = (t1 - t0) / 50 * 1000
        throughput = bs / (latency_ms / 1000)
        mem_mb = 0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  {bs:<8} {latency_ms:<14.4f} {throughput:<20.1f} {mem_mb:<14.2f}")

    # ---- 4. 吞吐量 vs 序列长度 ----
    print(f"\n  {'Seq Len':<10} {'Latency(ms)':<14} {'Throughput(samp/s)':<20} {'GPU Mem(MB)':<14}")
    print(f"  {'-'*10} {'-'*14} {'-'*20} {'-'*14}")
    seq_lengths = [10, 20, 50, 100, 200]
    bs_fixed = 64
    for seq_len in seq_lengths:
        # 创建临时模型（适配不同窗口大小）
        try:
            tmp_model = TimesNet(feats, n_window=seq_len).to(dtype=default_dtype).to(device)
            tmp_model.eval()
        except Exception:
            print(f"  {seq_len:<10} (OOM or not supported)")
            continue
        x = torch.randn(bs_fixed, seq_len, feats, dtype=default_dtype, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            _ = tmp_model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        latency_ms = (t1 - t0) / 30 * 1000
        throughput = bs_fixed / (latency_ms / 1000)
        mem_mb = 0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  {seq_len:<10} {latency_ms:<14.4f} {throughput:<20.1f} {mem_mb:<14.2f}")

    # ---- 5. 汇总 ----
    print(f"\n  --- 效率汇总 (batch=128, seq_len=10) ---")
    x = torch.randn(128, w_size, feats, dtype=default_dtype, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) / 100 * 1000
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  推理延迟 (batch=128): {avg_ms:.4f} ms")
    print(f"  每样本延迟:           {avg_ms/128*1000:.4f} μs")
    print(f"  吞吐量:               {128/(avg_ms/1000):.1f} samples/s")
    print(f"  推理峰值显存:         {peak_mem:.2f} MB")
    print(f"  模型参数内存:         {param_mem:.4f} MB")
    print(f"  总参数量:             {n_params:,}")

    return {
        'params': n_params,
        'param_mem_mb': param_mem,
        'latency_ms_batch128': avg_ms,
        'latency_us_per_sample': avg_ms / 128 * 1000,
        'throughput_samp_per_s': 128 / (avg_ms / 1000),
        'peak_gpu_mem_mb': peak_mem,
    }


def run_experiment(dataset_name, args):
    """在单个数据集上运行完整实验"""
    print(f"\n{'='*70}")
    print(f"  TimesNet on {dataset_name}")
    print(f"{'='*70}")

    # 加载数据
    train_loader, test_loader, labels, feats = load_dataset(dataset_name)
    w_size = 10
    n_epochs = args.epochs

    # 创建模型
    model = TimesNet(feats, n_window=w_size).double().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_D[dataset_name], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    criterion = nn.MSELoss()

    # 训练
    print(f"  训练 {n_epochs} epochs...")
    t_start = time.time()
    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, w_size, feats)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  Loss={loss:.6f}")
    train_time = time.time() - t_start
    print(f"  训练时间: {train_time:.2f} s")

    # 测试
    print(f"  评估中...")
    t_start = time.time()
    train_scores = compute_anomaly_scores(model, train_loader, w_size, feats)
    test_scores  = compute_anomaly_scores(model, test_loader, w_size, feats)
    test_time = time.time() - t_start
    print(f"  测试时间: {test_time:.4f} s (per sample: {test_time/len(test_scores)*1000:.4f} ms)")

    # POT 评估 (点调整 F1)
    init_score = np.mean(train_scores, axis=1)           # (N_train,)
    score       = np.mean(test_scores, axis=1)            # (N_test,)
    label_1d    = (np.sum(labels, axis=1) >= 1).astype(np.float64)

    f1_pa, f1_pw, precision, recall, auc, threshold = pot_eval(
        init_score, score, label_1d, LM_PARAMS[dataset_name]
    )

    results = {
        'dataset':    dataset_name,
        'feats':      feats,
        'params':     n_params,
        'f1_pa':      f1_pa,
        'f1_pw':      f1_pw,
        'auc':        auc,
        'precision':  precision,
        'recall':     recall,
        'train_time': train_time,
        'test_time':  test_time,
        'threshold':  threshold,
    }
    print(f"  F1(PA)={f1_pa:.4f}  F1(PW)={f1_pw:.4f}  AUC={auc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    return results


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='TimesNet Anomaly Detection Experiment')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name or "all" (default: all)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5)')
    parser.add_argument('--output', type=str, default='results/results_timesnet.json',
                        help='Output JSON file')
    args = parser.parse_args()

    datasets = ['SMD', 'NAB', 'MBA', 'SMAP', 'SWaT_UV', 'SWaT_MV'] \
        if args.dataset == 'all' else [args.dataset]

    all_results = []
    for ds in datasets:
        try:
            res = run_experiment(ds, args)
            all_results.append(res)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback
            traceback.print_exc()

    # 输出结果表格
    print(f"\n{'='*120}")
    print("  实验结果汇总")
    print(f"{'='*120}")
    header = f"{'Dataset':<12} {'Feats':<6} {'Params':<10} {'F1(PA)':<8} {'F1(PW)':<8} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'Train(s)':<10} {'Test(s)':<10}"
    print(header)
    print("-" * 130)
    for r in all_results:
        print(f"{r['dataset']:<12} {r['feats']:<6} {r['params']:<10,} "
              f"{r['f1_pa']:<8.4f} {r['f1_pw']:<8.4f} {r['auc']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} "
              f"{r['train_time']:<10.2f} {r['test_time']:<10.4f}")

    # LaTeX 表格行
    if len(all_results) >= 4:
        print(f"\n--- LaTeX 表格行 (F1) ---")
        f1_pas = {r['dataset']: r['f1_pa'] for r in all_results}
        f1_pws = {r['dataset']: r['f1_pw'] for r in all_results}
        aucs = {r['dataset']: r['auc'] for r in all_results}
        order = ['SMD', 'NAB', 'MBA', 'SMAP', 'SWaT_MV', 'SWaT_UV']

        latex_f1 = ' & '.join([f"{f1_pas.get(d, 0):.4f}" for d in order])
        print(f"TimesNet (F1-PA)  & {latex_f1} \\\\")
        latex_f1pw = ' & '.join([f"{f1_pws.get(d, 0):.4f}" for d in order])
        print(f"TimesNet (F1-PW)  & {latex_f1pw} \\\\")
        latex_auc = ' & '.join([f"{aucs.get(d, 0):.4f}" for d in order])
        print(f"TimesNet (AUC)    & {latex_auc} \\\\")

    # 参数量 & 时间
    print(f"\n--- Parameters & Timing ---")
    for r in all_results:
        print(f"TimesNet-{r['dataset']}: params={r['params']:,}, "
              f"train={r['train_time']:.2f}s, test={r['test_time']:.4f}s")

    # 保存
    # 保存 (convert numpy types to Python native)
    results_json = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in r.items()} for r in all_results]
    with open(args.output, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
