"""
================================================================================
DCdetector 实验脚本 — 独立运行
DCdetector: Dual Attention Contrastive Representation Learning
for Time Series Anomaly Detection

在6个数据集上评估: SMD, NAB, MBA, SMAP, SWaT_MV, SWaT_UV
指标: F1, AUC, Precision, Recall, 参数量, 训练时间, 测试时间

用法:
    py run_dcdetector.py                    # 所有数据集
    py run_dcdetector.py --dataset SMD      # 单个数据集
    py run_dcdetector.py --epochs 20        # 指定训练轮数
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

# 数据集配置
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
    """从 series.json 生成 SWaT_UV 和 SWaT_MV"""
    swat_folder = os.path.join(PROCESSED, 'SWaT')
    uv_train_path = os.path.join(swat_folder, 'swat_uv_train.npy')
    mv_train_path = os.path.join(swat_folder, 'swat_mv_train.npy')

    if os.path.exists(uv_train_path) and os.path.exists(mv_train_path):
        return

    print("预处理 SWaT 数据...")
    series_path = os.path.join(DATA, 'SWaT', 'series.json')
    df = pd.read_json(series_path, lines=True)

    vals = df[['val']].values.astype(np.float64)
    noti = df[['noti']].values.astype(np.float64)

    def normalize(a):
        return (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-4)

    # === SWaT_UV ===
    uv_train = normalize(vals[3000:6000])
    uv_test  = normalize(vals[7000:12000])
    uv_labels = noti[7000:12000]

    os.makedirs(swat_folder, exist_ok=True)
    np.save(os.path.join(swat_folder, 'swat_uv_train.npy'),  uv_train)
    np.save(os.path.join(swat_folder, 'swat_uv_test.npy'),   uv_test)
    np.save(os.path.join(swat_folder, 'swat_uv_labels.npy'), uv_labels)
    print(f"  SWaT_UV: train={uv_train.shape}, test={uv_test.shape}")

    # === SWaT_MV: val + lag features + rolling stats ===
    window = 10
    vs = vals.flatten()
    features = [vs]
    for lag in [1, 2, 3]:
        lagged = np.roll(vs, lag)
        lagged[:lag] = lagged[lag]
        features.append(lagged)
    rm = pd.Series(vs).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    rs = pd.Series(vs).rolling(window=window, center=True).std().fillna(0).values
    features.append(rm)
    features.append(rs)
    mv_data = np.stack(features, axis=1)

    mv_train = normalize(mv_data[3000:6000])
    mv_test  = normalize(mv_data[7000:12000])
    mv_labels = noti[7000:12000]

    np.save(os.path.join(swat_folder, 'swat_mv_train.npy'),  mv_train)
    np.save(os.path.join(swat_folder, 'swat_mv_test.npy'),   mv_test)
    np.save(os.path.join(swat_folder, 'swat_mv_labels.npy'), mv_labels)
    print(f"  SWaT_MV: train={mv_train.shape}, test={mv_test.shape}")


# ==================== 数据加载 ====================
def load_dataset(dataset_name):
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
    windows = np.zeros((N, w_size, C), dtype=data.dtype)
    for i in range(N):
        if i >= w_size:
            windows[i] = data[i - w_size:i]
        else:
            pad_len = w_size - i
            windows[i, :pad_len] = data[0:1]
            if i > 0:
                windows[i, pad_len:] = data[:i]
    return windows


# ==================== SPOT / POT 异常检测评估 ====================
class SPOT:
    def __init__(self, q=1e-4):
        self.q = q
        self.init_threshold = None

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

    # PA (point-adjust)
    pred_pa = adjust_predicts(score, label, pot_th)
    f1_pa, precision, recall, TP, TN, FP, FN = calc_point2point(pred_pa, label)

    # Point-wise (无 PA)
    pred_pw = (score > pot_th).astype(float)
    f1_pw, pw_precision, pw_recall, _, _, _, _ = calc_point2point(pred_pw, label)

    try:
        auc = roc_auc_score(label, score)
    except Exception:
        auc = 0.5
    return f1_pa, f1_pw, precision, recall, auc, pot_th


# ==================== DCdetector 模型 ====================
class PatchEmbedding(nn.Module):
    """将时间序列分割为重叠 patch 并嵌入"""
    def __init__(self, n_window, patch_len, stride, n_feats, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (n_window - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * n_feats, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        B, T, C = x.shape
        patches = []
        for i in range(0, T - self.patch_len + 1, self.stride):
            patches.append(x[:, i:i + self.patch_len, :].reshape(B, -1))
        patches = torch.stack(patches, dim=1)                     # (B, num_patches, patch_len*C)
        return self.embed(patches) + self.pos_embed               # (B, num_patches, d_model)


class DualAttentionBlock(nn.Module):
    """双重注意力: patch-wise 时序注意力 + channel-wise 特征注意力"""
    def __init__(self, d_model, n_feats, nhead=8, dropout=0.1):
        super().__init__()
        # Patch-wise attention (时序维度)
        self.patch_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Channel-wise attention (特征维度)
        self.channel_proj = nn.Linear(d_model, n_feats)
        ch_nhead = 1
        for h in range(min(nhead, n_feats), 0, -1):
            if n_feats % h == 0:
                ch_nhead = h
                break
        self.channel_attn = nn.MultiheadAttention(n_feats, ch_nhead, dropout=dropout, batch_first=True)
        self.channel_proj_back = nn.Linear(n_feats, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # Patch attention
        a, _ = self.patch_attn(x, x, x)
        x = self.norm1(x + self.dropout1(a))
        # Channel attention
        x_proj = self.channel_proj(x)                            # (B, N, n_feats)
        c, _ = self.channel_attn(x_proj, x_proj, x_proj)
        x_ch = self.channel_proj_back(c)                          # (B, N, d_model)
        x = self.norm2(x + self.dropout2(x_ch))
        # FFN
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class DCdetector(nn.Module):
    """DCdetector: 双重注意力对比学习异常检测模型"""
    def __init__(self, feats, n_window=10, patch_len=3, stride=2, d_model=64,
                 nhead=8, n_layers=2):
        super().__init__()
        self.name = 'DCdetector'
        self.n_feats = feats
        self.n_window = n_window
        self.patch_len = patch_len
        self.stride = stride

        self.patch_embed = PatchEmbedding(n_window, patch_len, stride, feats, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.blocks = nn.ModuleList([
            DualAttentionBlock(d_model, feats, nhead) for _ in range(n_layers)
        ])
        self.reconstruct = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, patch_len * feats),
        )
        self.fcn = nn.Sigmoid()

    def _fold_patches(self, rec_patches, B, T, C):
        """将重叠 patch 折叠回完整时序"""
        rec = torch.zeros(B, T, C, device=rec_patches.device)
        cnt = torch.zeros(B, T, 1, device=rec_patches.device)
        for i in range(self.num_patches):
            start = i * self.stride
            rec[:, start:start + self.patch_len, :] += \
                rec_patches[:, i, :].reshape(B, self.patch_len, C)
            cnt[:, start:start + self.patch_len, :] += 1
        return rec / cnt.clamp(min=1)

    def forward(self, x):
        B, T, C = x.shape
        h = self.patch_embed(x)                                  # (B, num_patches, d_model)
        for block in self.blocks:
            h = block(h)
        rec_patches = self.reconstruct(h)                         # (B, num_patches, patch_len*C)
        return self.fcn(self._fold_patches(rec_patches, B, T, C))


# ==================== 训练 & 评估 ====================
def train_epoch(model, dataloader, optimizer, criterion, w_size, feats):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in dataloader:
        data = batch if not isinstance(batch, (list, tuple)) else batch[0]
        data = data.float().to(DEVICE)
        data_w = torch.tensor(to_windows(data.cpu().numpy(), w_size),
                              dtype=torch.float32).to(DEVICE)
        rec = model(data_w)
        loss = criterion(rec, data_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def compute_anomaly_scores(model, dataloader, w_size, feats):
    model.eval()
    all_errors = []
    for batch in dataloader:
        data = batch if not isinstance(batch, (list, tuple)) else batch[0]
        data = data.float().to(DEVICE)
        data_w = torch.tensor(to_windows(data.cpu().numpy(), w_size),
                              dtype=torch.float32).to(DEVICE)
        rec = model(data_w)
        err = ((rec - data_w) ** 2).mean(dim=1)                  # (N, C)
        all_errors.append(err.cpu().numpy())
    return np.concatenate(all_errors, axis=0)


def run_experiment(dataset_name, args):
    print(f"\n{'='*70}")
    print(f"  DCdetector on {dataset_name}")
    print(f"{'='*70}")

    train_loader, test_loader, labels, feats = load_dataset(dataset_name)
    w_size = 10
    n_epochs = args.epochs

    # 根据特征数自适应调整 d_model 和 nhead
    d_model = min(64, feats * 4) if feats < 10 else 64
    nhead_val = 8 if d_model >= 8 else max(1, d_model // 4)
    # 确保 d_model 可被 nhead 整除
    while d_model % nhead_val != 0 and nhead_val > 1:
        nhead_val -= 1
    patch_len = min(5, w_size // 2)
    stride = max(1, patch_len // 2)

    model = DCdetector(feats, n_window=w_size, patch_len=patch_len,
                       stride=stride, d_model=d_model,
                       nhead=nhead_val, n_layers=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  d_model={d_model}, nhead={nhead_val}, patch_len={patch_len}, stride={stride}")
    print(f"  num_patches={model.num_patches}")
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

    # POT 评估
    init_score = np.mean(train_scores, axis=1)
    score       = np.mean(test_scores, axis=1)
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
    parser = argparse.ArgumentParser(description='DCdetector Anomaly Detection Experiment')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name or "all"')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5)')
    parser.add_argument('--output', type=str, default='results/results_dcdetector.json',
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
        print(f"\n--- LaTeX 表格行 ---")
        f1_pas = {r['dataset']: r['f1_pa'] for r in all_results}
        f1_pws = {r['dataset']: r['f1_pw'] for r in all_results}
        aucs = {r['dataset']: r['auc'] for r in all_results}
        order = ['SMD', 'NAB', 'MBA', 'SMAP', 'SWaT_MV', 'SWaT_UV']
        latex_f1 = ' & '.join([f"{f1_pas.get(d, 0):.4f}" for d in order])
        latex_f1pw = ' & '.join([f"{f1_pws.get(d, 0):.4f}" for d in order])
        latex_auc = ' & '.join([f"{aucs.get(d, 0):.4f}" for d in order])
        print(f"DCdetector (F1-PA)  & {latex_f1} \\\\")
        print(f"DCdetector (F1-PW)  & {latex_f1pw} \\\\")
        print(f"DCdetector (AUC)    & {latex_auc} \\\\")

    print(f"\n--- Parameters & Timing ---")
    for r in all_results:
        print(f"DCdetector-{r['dataset']}: params={r['params']:,}, "
              f"train={r['train_time']:.2f}s, test={r['test_time']:.4f}s")

    # 保存 (convert numpy types to Python native)
    results_json = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in r.items()} for r in all_results]
    with open(args.output, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
