"""
================================================================================
EPAAD-Net 跨数据集/跨机器泛化评估
1. 跨数据集: NAB(1D) <-> SWaT(1D)
2. 跨机器(SMD): train machine-1-1, test machine-1-2..1-8 / machine-2-* / machine-3-*
================================================================================
用法: py eval_cross_dataset.py --epochs 5
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, glob, time, json, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== EPAAD-Net ====================
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous()
class TemporalCnn(nn.Module):
    def __init__(self, ni, no, ks, st, dil, pad, drop=0.2):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(ni, no, ks, stride=st, padding=pad, dilation=dil))
        self.net = nn.Sequential(self.conv, Chomp1d(pad), nn.ReLU(True), nn.Dropout(drop))
        self.relu = nn.ReLU(); self.conv.weight.data.normal_(0, 0.01)
    def forward(self, x): return self.relu(self.net(x) + x)
class Tcn_Local(nn.Module):
    def __init__(self, no, ks=4, drop=0.2):
        super().__init__()
        self.network = nn.Sequential(*[TemporalCnn(no, no, ks, 1, 1, ks-1, drop) for _ in range(3)])
    def forward(self, x): return self.network(x)
class SCTM(nn.Module):
    def __init__(self, sd, hd):
        super().__init__()
        self.rw = nn.Parameter(torch.Tensor(sd, hd)); self.iw = nn.Parameter(torch.Tensor(sd, hd))
        nn.init.xavier_uniform_(self.rw); nn.init.xavier_uniform_(self.iw)
    def forward(self, x): return 0.3*torch.cos(torch.matmul(x, self.rw)) + 0.7*torch.sin(torch.matmul(x, self.iw))
class DecoderLayer(nn.Module):
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.ae1 = nn.Sequential(nn.Linear(dm, dm//3), nn.ReLU(), nn.Linear(dm//3, dm), nn.ReLU())
        self.ae2 = nn.Sequential(nn.Linear(dm, dm//3), nn.ReLU(), nn.Linear(dm//3, dm), nn.ReLU())
        self.mha = nn.MultiheadAttention(dm, nh, dropout=drop)
        self.l1 = nn.Linear(dm, dff); self.dr = nn.Dropout(drop); self.l2 = nn.Linear(dff, dm)
        self.d1 = nn.Dropout(drop); self.d2 = nn.Dropout(drop)
        self.d3 = nn.Dropout(drop); self.d4 = nn.Dropout(drop); self.act = nn.LeakyReLU(True)
    def forward(self, tgt, mem):
        tgt = tgt + self.d1(self.ae1(tgt)); tgt = tgt + self.d2(self.ae2(tgt))
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.dr(self.act(self.l1(tgt)))))
class EPAADNet(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw
        self.l_tcn = Tcn_Local(feats, 4, 0.2); self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1); self.dec = DecoderLayer(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src, tgt=None):
        if tgt is None: tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        return self.fcn(self.dec(tgt, self.sp(src + self.drop(s2.permute(2, 0, 1)))))

# ==================== 工具 ====================
def to_windows(data, w=10):
    data = np.asarray(data)
    if data.ndim == 1: data = data.reshape(-1, 1)
    N, C = data.shape
    win = np.zeros((N, w, C), dtype=data.dtype)
    for i in range(N):
        if i >= w: win[i] = data[i - w:i]
        else:
            pad = w - i; win[i, :pad] = np.tile(data[0:1], (pad, 1))
            if i > 0: win[i, pad:] = data[0:i].reshape(i, C)
    return win

class SPOT:
    def __init__(self, q=1e-4): self.q = q
    def fit(self, a, b): self.a = a; self.b = b; return self
    def initialize(self, level=0.98, **kw):
        th = np.percentile(self.a, level*100); self.th = th
        ex = self.a[self.a > th] - th
        if len(ex) == 0: ex = np.array([np.percentile(self.a, 99) - th])
        self.extrema = ex; return self
    def run(self, **kw): return {'thresholds': [self.th]*len(self.b)}

def adjust_predicts(score, label, th):
    """Point-Adjust: 异常段内有一点检出即全标为检出"""
    s = np.asarray(score); l = np.asarray(label)
    pred = s > th; act = l > 0.1; anom = False
    for i in range(len(s)):
        if act[i] and pred[i] and not anom:
            anom = True
            for j in range(i, 0, -1):
                if not act[j]: break
                elif not pred[j]: pred[j] = True
        elif not act[i]: anom = False
        if anom: pred[i] = True
    return pred

def pa_f1(init_score, score, label, lms=0.98, lm2=1.0):
    """标准 PA-F1 (与 main.py 一致)"""
    while True:
        try: s = SPOT(); s.fit(init_score, score); s.initialize(level=lms)
        except: lms *= 0.999
        else: break
    th = np.mean(s.run()['thresholds']) * lm2
    pred = adjust_predicts(score, label, th)
    TP = np.sum(pred*label); FP = np.sum(pred*(1-label)); FN = np.sum((1-pred)*label)
    p = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8); f1 = 2*p*r/(p+r+1e-8)
    try: auc = roc_auc_score(label, score)
    except: auc = 0.5
    return f1, auc

# ==================== 训练 ====================
def train_and_score(model, train, test, test_labels, lr=1e-4, epochs=5):
    w = model.n_window; feats = model.n_feats
    label_1d = (np.sum(test_labels, axis=1) >= 1).astype(float)
    loader = DataLoader(TensorDataset(torch.tensor(train, dtype=torch.float64)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9); crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            d = batch.to(DEVICE).double()
            dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=torch.float64).to(DEVICE)
            inp = dw.permute(1,0,2); tgt = inp[-1:,:,:]
            loss = crit(model(inp, tgt).squeeze(0), dw[:,-1,:])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    @torch.no_grad()
    def scores(data):
        model.eval(); errs = []
        for (batch,) in DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float64)), batch_size=128):
            d = batch.to(DEVICE).double()
            dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=torch.float64).to(DEVICE)
            inp = dw.permute(1,0,2); rec = model(inp, inp[-1:,:,:]).squeeze(0)
            errs.append(((rec - dw[:,-1,:])**2).mean(dim=1).cpu().numpy())
        return np.concatenate(errs)
    ts = scores(train); ss = scores(test)
    return pa_f1(ts, ss, label_1d)

# ==================== 实验 1: 跨数据集 (NAB <-> SWaT) ====================
def cross_dataset(epochs):
    print(f"\n{'='*70}")
    print("  实验 1: 跨数据集泛化 (NAB <-> SWaT, 均为 1D)")
    print(f"{'='*70}")

    # Load
    swat_train = np.load(os.path.join(ROOT, 'processed', 'SWaT', 'train.npy'))
    swat_test  = np.load(os.path.join(ROOT, 'processed', 'SWaT', 'test.npy'))
    swat_labels = np.load(os.path.join(ROOT, 'processed', 'SWaT', 'labels.npy'))

    nab_train = np.load(os.path.join(ROOT, 'processed', 'NAB',
                       'ec2_request_latency_system_failure_train_1.npy'))
    nab_test  = np.load(os.path.join(ROOT, 'processed', 'NAB',
                       'ec2_request_latency_system_failure_test_1.npy'))
    nab_labels = np.load(os.path.join(ROOT, 'processed', 'NAB',
                        'ec2_request_latency_system_failure_labels_1.npy'))

    lr = 8e-3  # SWaT LR (higher for 1D)

    results = {}
    for src_name, src_train, tgt_name, tgt_test, tgt_labels in [
        ('NAB', nab_train, 'SWaT', swat_test, swat_labels),
        ('SWaT', swat_train, 'NAB', nab_test, nab_labels),
    ]:
        print(f"\n  Train {src_name} -> Test {tgt_name}")
        model = EPAADNet(1).double().to(DEVICE)
        f1, auc = train_and_score(model, src_train, tgt_test, tgt_labels, lr, epochs)
        print(f"    F1(PA)={f1:.4f}  AUC={auc:.4f}")
        results[f'{src_name}->{tgt_name}'] = {'f1': f1, 'auc': auc}

    # Self-test baselines
    for name, train, test, labels in [
        ('NAB', nab_train, nab_test, nab_labels),
        ('SWaT', swat_train, swat_test, swat_labels),
    ]:
        print(f"\n  Self-test {name} (baseline)")
        model = EPAADNet(1).double().to(DEVICE)
        f1, auc = train_and_score(model, train, test, labels, lr, epochs)
        print(f"    F1(PA)={f1:.4f}  AUC={auc:.4f}")
        results[f'{name}(self)'] = {'f1': f1, 'auc': auc}

    return results


# ==================== 实验 2: 跨数据集泛化 (Train SMD → Test 5 datasets) ====================
def cross_dataset_5(epochs):
    """
    Train on SMD (machine-1-1, 38D), test on 5 other datasets.
    通过公共特征子集匹配维度:
      NAB(1D) / SWaT(1D): 取 SMD 第1个特征
      MBA(2D):             取 SMD 前2个特征
      SMAP(25D):           取 SMD 前25个特征
      MSL(55D):            取 SMD 前38个特征 + 补零
    """
    print(f"\n{'='*70}")
    print("  实验 2: 跨数据集泛化 (Train SMD -> Test 5 datasets)")
    print(f"{'='*70}")

    smd_dir = os.path.join(ROOT, 'processed', 'SMD')
    src_train_full = np.load(os.path.join(smd_dir, 'machine-1-1_train.npy'))

    # 目标数据集
    targets = {
        'NAB': {
            'feats': 1,
            'train': np.load(os.path.join(ROOT, 'processed', 'NAB',
                'ec2_request_latency_system_failure_train_1.npy')),
            'test': np.load(os.path.join(ROOT, 'processed', 'NAB',
                'ec2_request_latency_system_failure_test_1.npy')),
            'labels': np.load(os.path.join(ROOT, 'processed', 'NAB',
                'ec2_request_latency_system_failure_labels_1.npy')),
        },
        'MBA': {
            'feats': 2,
            'train': np.load(os.path.join(ROOT, 'processed', 'MBA', 'train_1.npy')),
            'test': np.load(os.path.join(ROOT, 'processed', 'MBA', 'test_1.npy')),
            'labels': np.load(os.path.join(ROOT, 'processed', 'MBA', 'labels_1.npy')),
        },
        'SMAP': {
            'feats': 25,
            'train': np.load(os.path.join(ROOT, 'processed', 'SMAP', 'P-1_train.npy')),
            'test': np.load(os.path.join(ROOT, 'processed', 'SMAP', 'P-1_test.npy')),
            'labels': np.load(os.path.join(ROOT, 'processed', 'SMAP', 'P-1_labels.npy')),
        },
        'MSL': {
            'feats': 38,  # 取前38维匹配SMD
            'train': np.load(os.path.join(ROOT, 'processed', 'MSL', 'C-1_train.npy')),
            'test': np.load(os.path.join(ROOT, 'processed', 'MSL', 'C-1_test.npy')),
            'labels': np.load(os.path.join(ROOT, 'processed', 'MSL', 'C-1_labels.npy')),
        },
        'SWaT': {
            'feats': 1,
            'train': np.load(os.path.join(ROOT, 'processed', 'SWaT', 'train.npy')),
            'test': np.load(os.path.join(ROOT, 'processed', 'SWaT', 'test.npy')),
            'labels': np.load(os.path.join(ROOT, 'processed', 'SWaT', 'labels.npy')),
        },
    }

    results = {}
    for tgt_name, tgt in targets.items():
        k = tgt['feats']
        src_train = src_train_full[:, :k].copy()  # 取前k个特征
        tgt_test  = tgt['test'][:, :k] if tgt['test'].shape[1] >= k else \
                     np.pad(tgt['test'], ((0, 0), (0, k - tgt['test'].shape[1])))
        tgt_labels = tgt['labels'][:, :k] if tgt['labels'].shape[1] >= k else \
                      np.pad(tgt['labels'], ((0, 0), (0, k - tgt['labels'].shape[1])))

        print(f"\n  Train SMD(38D)[:{k}] -> Test {tgt_name}({tgt['test'].shape[1]}D)")
        lr = 1e-4
        print(f"    src_train={src_train.shape}  tgt_test={tgt_test.shape}", end=' ', flush=True)
        model = EPAADNet(k).double().to(DEVICE)
        f1, auc = train_and_score(model, src_train, tgt_test, tgt_labels, lr, epochs)
        results[f'SMD->{tgt_name}'] = {'f1': f1, 'auc': auc}
        print(f"F1(PA)={f1:.4f}  AUC={auc:.4f}")

    # Self-test baseline
    print(f"\n  Self-test SMD (baseline):")
    smd_test  = np.load(os.path.join(smd_dir, 'machine-1-1_test.npy'))
    smd_labels = np.load(os.path.join(smd_dir, 'machine-1-1_labels.npy'))
    model = EPAADNet(38).double().to(DEVICE)
    f1, auc = train_and_score(model, src_train_full, smd_test, smd_labels, 1e-4, epochs)
    results['SMD(self)'] = {'f1': f1, 'auc': auc}
    print(f"    F1(PA)={f1:.4f}  AUC={auc:.4f}")

    return results


# ==================== 汇总 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # 实验 1
    cd = cross_dataset(args.epochs)

    print(f"\n{'='*70}")
    print("  跨数据集泛化 汇总")
    print(f"{'='*70}")
    for k, v in cd.items():
        print(f"  {k:<20} F1(PW)={v['f1']:.4f}  AUC={v['auc']:.4f}")
    print(f"\nLaTeX:")
    for k, v in cd.items():
        print(f"  {k} & {v['f1']:.4f} & {v['auc']:.4f} \\\\")

    # 实验 2
    cd5 = cross_dataset_5(args.epochs)

    print(f"\n{'='*70}")
    print("  跨数据集泛化 汇总 (Train SMD -> Test 5 datasets)")
    print(f"{'='*70}")
    for k, v in cd5.items():
        print(f"  {k:<20} F1(PA)={v['f1']:.4f}  AUC={v['auc']:.4f}")
    print(f"\nLaTeX:")
    for k, v in cd5.items():
        print(f"  {k} & {v['f1']:.4f} & {v['auc']:.4f} \\\\")
