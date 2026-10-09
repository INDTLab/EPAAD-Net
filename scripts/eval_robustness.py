"""
================================================================================
EPAADNet/EPAAD-Net 鲁棒性评估
评估: 跨数据集泛化, 分布漂移, 异常比例, 含噪数据, 数据缺失
================================================================================
用法:
    py eval_robustness.py --dataset SMD --epochs 5
    py eval_robustness.py --all --epochs 5
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, json, math, time, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 配置 ====================
DATASET_CONFIG = {
    'SMD':  ('SMD',  'machine-1-1_train',   'machine-1-1_test',   'machine-1-1_labels'),
    'NAB':  ('NAB',  'ec2_request_latency_system_failure_train_1',
                      'ec2_request_latency_system_failure_test_1',
                      'ec2_request_latency_system_failure_labels_1'),
    'MBA':  ('MBA',  'train_1',             'test_1',             'labels_1'),
    'SMAP': ('SMAP', 'P-1_train',           'P-1_test',           'P-1_labels'),
    'MSL':  ('MSL',  'C-1_train',           'C-1_test',           'C-1_labels'),
    'SWaT': ('SWaT', 'train',               'test',               'labels'),
}

LR_D = {'SMD': 1e-4, 'NAB': 9e-3, 'MBA': 1e-3, 'SMAP': 1e-3, 'MSL': 2e-3, 'SWaT': 8e-3}
LM_P = {'SMD': (0.99995, 1.06), 'NAB': (0.99, 1), 'MBA': (0.93, 1.04),
        'SMAP': (0.98, 1), 'MSL': (0.999, 1.04), 'SWaT': (0.993, 1)}


# ==================== 数据加载 ====================
def load_dataset(name):
    cfg = DATASET_CONFIG[name]
    folder = os.path.join(ROOT, 'processed', cfg[0])
    train = np.load(os.path.join(folder, f'{cfg[1]}.npy'))
    test  = np.load(os.path.join(folder, f'{cfg[2]}.npy'))
    labels = np.load(os.path.join(folder, f'{cfg[3]}.npy'))
    return train, test, labels


def to_windows(data, w_size=10):
    data = np.asarray(data)
    if data.ndim == 1: data = data.reshape(-1, 1)
    N, C = data.shape
    windows = np.zeros((N, w_size, C), dtype=data.dtype)
    for i in range(N):
        if i >= w_size:
            windows[i] = data[i - w_size:i]
        else:
            pad = w_size - i
            windows[i, :pad] = data[0:1]
            if i > 0: windows[i, pad:] = data[:i]
    return windows


# ==================== EPAADNet 模型 ====================
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous()

class TemporalCnn(nn.Module):
    def __init__(self, ni, no, ks, stride, dil, pad, dropout=0.2):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(ni, no, ks, stride=stride, padding=pad, dilation=dil))
        self.chomp = Chomp1d(pad)
        self.net = nn.Sequential(self.conv, self.chomp, nn.ReLU(True), nn.Dropout(dropout))
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)
    def forward(self, x): return self.relu(self.net(x) + x)

class Tcn_Local(nn.Module):
    def __init__(self, no, ks=3, dropout=0.2):
        super().__init__()
        layers = [TemporalCnn(no, no, ks, 1, 1, ks-1, dropout) for _ in range(3)]
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

class SCTM(nn.Module):
    def __init__(self, sd, hd):
        super().__init__()
        self.rw = nn.Parameter(torch.Tensor(sd, hd)); self.iw = nn.Parameter(torch.Tensor(sd, hd))
        nn.init.xavier_uniform_(self.rw); nn.init.xavier_uniform_(self.iw)
    def forward(self, x):
        return 0.3 * torch.cos(torch.matmul(x, self.rw)) + 0.7 * torch.sin(torch.matmul(x, self.iw))

class TransformerDecoderLayer1(nn.Module):
    def __init__(self, dm, nh, dff=16, dropout=0):
        super().__init__()
        self.ae1 = nn.Sequential(nn.Linear(dm, dm//3), nn.ReLU(), nn.Linear(dm//3, dm), nn.ReLU())
        self.ae2 = nn.Sequential(nn.Linear(dm, dm//3), nn.ReLU(), nn.Linear(dm//3, dm), nn.ReLU())
        self.mha = nn.MultiheadAttention(dm, nh, dropout=dropout)
        self.l1 = nn.Linear(dm, dff); self.drop = nn.Dropout(dropout); self.l2 = nn.Linear(dff, dm)
        self.d1 = nn.Dropout(dropout); self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout); self.d4 = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(True)
    def forward(self, tgt, mem):
        tgt = tgt + self.d1(self.ae1(tgt)); tgt = tgt + self.d2(self.ae2(tgt))
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.drop(self.act(self.l1(tgt)))))

class EPAADNet(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'EPAADNet'; self.n_feats = feats; self.n_window = nw
        self.l_tcn = Tcn_Local(feats, 4, 0.2); self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1); self.dec = TransformerDecoderLayer1(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src, tgt=None):
        if tgt is None: tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))


# ==================== POT 评估 ====================
class SPOT:
    def __init__(self, q=1e-4): self.q = q
    def fit(self, a, b): self.a = a; self.b = b; return self
    def initialize(self, level=0.98, **kw):
        th = np.percentile(self.a, level * 100); self.th = th
        ex = self.a[self.a > th] - th
        if len(ex) == 0: ex = np.array([np.percentile(self.a, 99) - th])
        self.extrema = ex; return self
    def run(self, **kw):
        alarms = []; ths = []
        for s in self.b:
            a = 1 if s > self.th else 0; alarms.append(a); ths.append(self.th)
        return {'alarms': alarms, 'thresholds': ths}

def adjust_predicts(score, label, th):
    s = np.asarray(score); l = np.asarray(label)
    pred = s > th; act = l > 0.1
    anomaly = False
    for i in range(len(s)):
        if act[i] and pred[i] and not anomaly:
            anomaly = True
            for j in range(i, 0, -1):
                if not act[j]: break
                elif not pred[j]: pred[j] = True
        elif not act[i]: anomaly = False
        if anomaly: pred[i] = True
    return pred

def pot_eval(init_s, score, label, lm, q=1e-5):
    lms = lm[0]
    while True:
        try:
            s = SPOT(q); s.fit(init_s, score); s.initialize(level=lms, min_extrema=False)
        except: lms *= 0.999
        else: break
    ret = s.run(); th = np.mean(ret['thresholds']) * lm[1]
    # PA
    pred_pa = adjust_predicts(score, label, th)
    TP = np.sum(pred_pa*label); TN = np.sum((1-pred_pa)*(1-label))
    FP = np.sum(pred_pa*(1-label)); FN = np.sum((1-pred_pa)*label)
    p = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8); f1_pa = 2*p*r/(p+r+1e-8)
    # Point-wise
    pred_pw = (score > th).astype(float)
    TP2 = np.sum(pred_pw*label); FP2 = np.sum(pred_pw*(1-label)); FN2 = np.sum((1-pred_pw)*label)
    f1_pw = 2*(TP2/(TP2+FP2+1e-8))*(TP2/(TP2+FN2+1e-8))/((TP2/(TP2+FP2+1e-8))+(TP2/(TP2+FN2+1e-8))+1e-8)
    try: auc = roc_auc_score(label, score)
    except: auc = 0.5
    return {'f1_pa': f1_pa, 'f1_pw': f1_pw, 'precision': p, 'recall': r, 'auc': auc}


# ==================== 训练 & 推理 ====================
def train_model(model, train_data, feats, lr, epochs=5):
    w = model.n_window
    loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float64)),
                        batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
    crit = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total = 0; n = 0
        for (batch,) in loader:
            data = batch.to(DEVICE).double()
            data_w = torch.tensor(to_windows(data.cpu().numpy(), w),
                                  dtype=torch.float64).to(DEVICE)
            # EPAADNet format: (T, B, C)
            inp = data_w.permute(1, 0, 2)
            tgt = inp[-1:, :, :]
            rec = model(inp, tgt)
            loss = crit(rec.squeeze(0), data_w[:, -1, :])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        sch.step()
    return model

@torch.no_grad()
def get_scores(model, data, feats):
    w = model.n_window
    loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float64)),
                        batch_size=128)
    model.eval()
    errors = []
    for (batch,) in loader:
        batch = batch.to(DEVICE).double()
        data_w = torch.tensor(to_windows(batch.cpu().numpy(), w),
                              dtype=torch.float64).to(DEVICE)
        inp = data_w.permute(1, 0, 2)
        tgt = inp[-1:, :, :]
        rec = model(inp, tgt).squeeze(0)
        err = ((rec - data_w[:, -1, :]) ** 2).mean(dim=1)
        errors.append(err.cpu().numpy())
    return np.concatenate(errors)


# ==================== 鲁棒性测试 ====================
def evaluate_standard(model, train_data, test_data, test_labels, feats, ds_name):
    """标准评估"""
    train_s = get_scores(model, train_data, feats)
    test_s = get_scores(model, test_data, feats)
    label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
    r = pot_eval(train_s, test_s, label_1d, LM_P[ds_name])
    return r


# ---- 1. 跨数据集泛化 ----
def eval_cross_dataset(model, train_data, feats, ds_name, all_datasets):
    """在数据集 A 上训练, 在 B/C/D 上测试"""
    results = {}
    for other_ds in all_datasets:
        if other_ds == ds_name: continue
        try:
            _, test_d, labels_d = load_dataset(other_ds)
            # 只测试特征维度匹配的数据集 (或取公共维度)
            feats_other = test_d.shape[1]
            if feats_other != feats:
                results[f'{ds_name}->{other_ds}'] = 'feats mismatch'
                continue
            train_s = get_scores(model, train_data, feats)
            test_s = get_scores(model, test_d, feats)
            label_1d = (np.sum(labels_d, axis=1) >= 1).astype(float)
            r = pot_eval(train_s, test_s, label_1d, LM_P.get(other_ds, LM_P[ds_name]))
            results[f'{ds_name}->{other_ds}'] = r
        except Exception as e:
            results[f'{ds_name}->{other_ds}'] = f'error: {e}'
    return results


# ---- 2. 分布漂移鲁棒性 ----
def add_distribution_shift(test_data, shift_type, severity):
    """
    shift_type: 'trend' (线性趋势), 'scale' (方差变化), 'shift' (均值漂移), 'seasonal' (周期变化)
    severity: 0.0 ~ 1.0
    """
    data = test_data.copy()
    N, C = data.shape
    t = np.arange(N)[:, None] / N

    if shift_type == 'trend':
        data += severity * 2 * t * data.std(0)
    elif shift_type == 'scale':
        data *= (1 + severity * np.sin(t * 20))
    elif shift_type == 'shift':
        data += severity * data.std(0)
    elif shift_type == 'seasonal':
        data += severity * data.std(0) * np.sin(t * np.pi * 4)
    return data


def eval_distribution_shift(model, train_data, test_data, test_labels, feats, ds_name):
    """评估不同类型和强度的分布漂移"""
    shifts = ['trend', 'scale', 'shift', 'seasonal']
    severities = [0.1, 0.3, 0.5]
    results = {}
    for st in shifts:
        for sv in severities:
            shifted = add_distribution_shift(test_data, st, sv)
            train_s = get_scores(model, train_data, feats)
            test_s = get_scores(model, shifted, feats)
            label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
            r = pot_eval(train_s, test_s, label_1d, LM_P[ds_name])
            results[f'{st}_{sv}'] = r
    return results


# ---- 3. 不同异常比例 ----
def eval_anomaly_ratio(model, train_data, test_data, test_labels, feats, ds_name):
    """调整测试集中的异常比例"""
    label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
    anom_idx = np.where(label_1d == 1)[0]
    normal_idx = np.where(label_1d == 0)[0]
    ratios = [0.01, 0.05, 0.10, 0.20, 0.50]
    results = {}

    for ratio in ratios:
        n_anom = max(1, int(len(normal_idx) * ratio / (1 - ratio)))
        if n_anom > len(anom_idx):
            sel_anom = np.random.choice(anom_idx, n_anom, replace=True)
        else:
            sel_anom = np.random.choice(anom_idx, n_anom, replace=False)

        combined_idx = np.sort(np.concatenate([normal_idx, sel_anom]))
        sub_test = test_data[combined_idx]
        sub_labels = test_labels[combined_idx]
        sub_label_1d = (np.sum(sub_labels, axis=1) >= 1).astype(float)

        train_s = get_scores(model, train_data, feats)
        test_s = get_scores(model, sub_test, feats)
        r = pot_eval(train_s, test_s, sub_label_1d, LM_P[ds_name])
        r['n_anom'] = n_anom
        results[f'ratio_{ratio:.2f}'] = r
    return results


# ---- 4. 含噪测量 ----
def add_noise(data, snr_db):
    """添加高斯噪声, SNR in dB"""
    sig_power = np.mean(data ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*data.shape) * np.sqrt(noise_power)
    return data + noise


def eval_noise(model, train_data, test_data, test_labels, feats, ds_name):
    """不同信噪比下的鲁棒性"""
    snr_levels = [5, 10, 15, 20, 30]
    results = {}
    for snr in snr_levels:
        noisy = add_noise(test_data, snr)
        train_s = get_scores(model, train_data, feats)
        test_s = get_scores(model, noisy, feats)
        label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
        r = pot_eval(train_s, test_s, label_1d, LM_P[ds_name])
        results[f'SNR_{snr}dB'] = r
    return results


# ---- 5. 数据缺失 ----
def add_missing(data, missing_rate):
    """随机 mask 数据"""
    masked = data.copy()
    mask = np.random.random(data.shape) < missing_rate
    masked[mask] = 0  # zero-fill
    return masked


def eval_missing(model, train_data, test_data, test_labels, feats, ds_name):
    """不同缺失率下的鲁棒性"""
    rates = [0.05, 0.10, 0.20, 0.30]
    results = {}
    for rate in rates:
        missing = add_missing(test_data, rate)
        train_s = get_scores(model, train_data, feats)
        test_s = get_scores(model, missing, feats)
        label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
        r = pot_eval(train_s, test_s, label_1d, LM_P[ds_name])
        results[f'missing_{rate:.0%}'] = r
    return results


# ==================== 主流程 ====================
def run_all(ds_name, datasets_list, epochs=5):
    print(f"\n{'='*70}")
    print(f"  鲁棒性评估: EPAADNet on {ds_name}")
    print(f"{'='*70}")

    train_data, test_data, test_labels = load_dataset(ds_name)
    feats = train_data.shape[1]
    print(f"  train={train_data.shape}, test={test_data.shape}, feats={feats}")

    # 训练模型
    model = EPAADNet(feats).double().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,},  训练 {epochs} epochs...")
    t0 = time.time()
    model = train_model(model, train_data, feats, LR_D[ds_name], epochs)
    print(f"  训练完成 ({time.time()-t0:.1f}s)")

    # 1. 标准评估
    std = evaluate_standard(model, train_data, test_data, test_labels, feats, ds_name)
    print(f"  标准:  F1={std['f1']:.4f}  AUC={std['auc']:.4f}")

    # 2. 跨数据集泛化
    print(f"\n  --- 跨数据集泛化 ---")
    cross = eval_cross_dataset(model, train_data, feats, ds_name, datasets_list)
    for k, v in cross.items():
        if isinstance(v, dict):
            print(f"    {k}: F1={v['f1_pa']:.4f} AUC={v['auc']:.4f}")
        else:
            print(f"    {k}: {v}")

    # 3. 分布漂移
    print(f"\n  --- 分布漂移 ---")
    shift = eval_distribution_shift(model, train_data, test_data, test_labels, feats, ds_name)
    for k, v in shift.items():
        print(f"    {k}: F1={v['f1_pa']:.4f} AUC={v['auc']:.4f}")

    # 4. 异常比例
    print(f"\n  --- 不同异常比例 ---")
    ratio_r = eval_anomaly_ratio(model, train_data, test_data, test_labels, feats, ds_name)
    for k, v in ratio_r.items():
        print(f"    {k}: F1={v['f1_pa']:.4f} AUC={v['auc']:.4f} (n_anom={v.get('n_anom', '?')})")

    # 5. 含噪数据
    print(f"\n  --- 含噪测量 ---")
    noise_r = eval_noise(model, train_data, test_data, test_labels, feats, ds_name)
    for k, v in noise_r.items():
        print(f"    {k}: F1={v['f1_pa']:.4f} AUC={v['auc']:.4f}")

    # 6. 数据缺失
    print(f"\n  --- 数据缺失 ---")
    miss_r = eval_missing(model, train_data, test_data, test_labels, feats, ds_name)
    for k, v in miss_r.items():
        print(f"    {k}: F1={v['f1_pa']:.4f} AUC={v['auc']:.4f}")

    return {
        'dataset': ds_name, 'feats': feats, 'params': n_params,
        'standard': std, 'cross_dataset': cross, 'distribution_shift': shift,
        'anomaly_ratio': ratio_r, 'noise': noise_r, 'missing_data': miss_r,
    }


def print_summary(all_results):
    """打印 LaTeX 汇总表"""
    print(f"\n\n{'='*100}")
    print(f"  鲁棒性评估汇总")
    print(f"{'='*100}")

    for res in all_results:
        ds = res['dataset']
        std = res['standard']
        print(f"\n--- {ds} (标准: F1(PA)={std['f1_pa']:.4f}, F1(PW)={std['f1_pw']:.4f}, AUC={std['auc']:.4f}) ---")

        # 跨数据集
        cross_items = {k: v for k, v in res['cross_dataset'].items() if isinstance(v, dict)}
        if cross_items:
            vals = '  '.join([f"{k}:F1={v['f1_pa']:.3f}" for k, v in cross_items.items()])
            print(f"  跨数据集: {vals}")

        # 分布漂移 (取平均)
        shift_f1s = [v['f1_pa'] for v in res['distribution_shift'].values()]
        print(f"  分布漂移: avg F1={np.mean(shift_f1s):.4f} (worst={np.min(shift_f1s):.4f})")

        # 异常比例
        print(f"  异常比例: " + '  '.join([f"{k}:{v['f1_pa']:.3f}" for k, v in res['anomaly_ratio'].items()]))

        # 噪声
        print(f"  含噪数据: " + '  '.join([f"{k}:{v['f1_pa']:.3f}" for k, v in res['noise'].items()]))

        # 缺失
        print(f"  数据缺失: " + '  '.join([f"{k}:{v['f1_pa']:.3f}" for k, v in res['missing_data'].items()]))

    # LaTeX 鲁棒性表
    print(f"\n\n--- LaTeX: 鲁棒性表 (F1) ---")
    for res in all_results:
        ds = res['dataset']
        std = res['standard']
        noise_avg = np.mean([v['f1_pa'] for v in res['noise'].values()])
        miss_avg = np.mean([v['f1_pa'] for v in res['missing_data'].values()])
        shift_avg = np.mean([v['f1_pa'] for v in res['distribution_shift'].values()])
        print(f"{ds} & {std['f1']:.4f} & {shift_avg:.4f} & {noise_avg:.4f} & {miss_avg:.4f} \\\\")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/robustness_results.json')
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.all else [args.dataset]
    all_results = []

    for ds in datasets:
        try:
            res = run_all(ds, datasets, args.epochs)
            all_results.append(res)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    print_summary(all_results)

    # 保存
    results_json = []
    for r in all_results:
        def convert(o):
            if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
            if isinstance(o, (np.floating, np.integer)): return float(o)
            if isinstance(o, list): return [convert(i) for i in o]
            return o
        results_json.append(convert(r))

    with open(args.output, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n结果已保存到 {args.output}")
