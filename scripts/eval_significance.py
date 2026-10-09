"""
EPAAD-Net 统计显著性检验 — 6 数据集 × 5 次独立运行
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, math, time, json, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'SMD':  ('SMD',  'machine-1-1_train',   'machine-1-1_test',   'machine-1-1_labels'),
    'NAB':  ('NAB',  'ec2_request_latency_system_failure_train_1',
                      'ec2_request_latency_system_failure_test_1',
                      'ec2_request_latency_system_failure_labels_1'),
    'MBA':  ('MBA',  'train_1',             'test_1',             'labels_1'),
    'SMAP': ('SMAP', 'P-1_train',           'P-1_test',           'P-1_labels'),
    'MSL':  ('MSL',  'C-1_train',           'C-1_test',           'C-1_labels'),
    'SWaT': ('SWaT', 'train',               'test',               'labels'),
}
LR = {'SMD': 1e-4, 'NAB': 9e-3, 'MBA': 1e-3, 'SMAP': 1e-3, 'MSL': 2e-3, 'SWaT': 8e-3}
LM = {'SMD': (0.99995, 1.06), 'NAB': (0.99, 1), 'MBA': (0.93, 1.04),
      'SMAP': (0.98, 1), 'MSL': (0.999, 1.04), 'SWaT': (0.993, 1)}

def load_data(name):
    c = DATASETS[name]; f = os.path.join(ROOT, 'processed', c[0])
    return (np.load(os.path.join(f, f'{c[1]}.npy')),
            np.load(os.path.join(f, f'{c[2]}.npy')),
            np.load(os.path.join(f, f'{c[3]}.npy')))

def to_windows(data, w=10):
    data = np.asarray(data)
    if data.ndim == 1: data = data.reshape(-1, 1)
    N, C = data.shape
    win = np.zeros((N, w, C), dtype=data.dtype)
    for i in range(N):
        if i >= w:
            win[i] = data[i - w:i]
        else:
            pad = w - i
            win[i, :pad] = np.tile(data[0:1], (pad, 1))
            if i > 0:
                win[i, pad:] = data[0:i].reshape(i, C)
    return win

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
class TransformerDecoderLayer1(nn.Module):
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
        self.drop = nn.Dropout(0.1); self.dec = TransformerDecoderLayer1(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src, tgt=None):
        if tgt is None: tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        return self.fcn(self.dec(tgt, self.sp(src + self.drop(s2.permute(2, 0, 1)))))

# ==================== 评估 ====================
class SPOT:
    def __init__(self, q=1e-4): self.q = q
    def fit(self, a, b): self.a = a; self.b = b; return self
    def initialize(self, level=0.98, **kw):
        th = np.percentile(self.a, level*100); self.th = th
        ex = self.a[self.a > th] - th
        if len(ex) == 0: ex = np.array([np.percentile(self.a, 99) - th])
        self.extrema = ex; return self
    def run(self, **kw):
        return {'thresholds': [self.th]*len(self.b)}

def pa_f1(init_score, score, label, lm_params):
    """标准 PA-F1 (与 main.py 一致)"""
    lms = lm_params[0]
    while True:
        try: s = SPOT(); s.fit(init_score, score); s.initialize(level=lms, min_extrema=False)
        except: lms *= 0.999
        else: break
    th = np.mean(s.run()['thresholds']) * lm_params[1]
    pred = (score > th).astype(float)
    label_a = np.asarray(label)
    # Point-Adjust
    act = label_a > 0.1; anom = False
    for i in range(len(pred)):
        if act[i] and pred[i] and not anom:
            anom = True
            for j in range(i, 0, -1):
                if not act[j]: break
                elif not pred[j]: pred[j] = True
        elif not act[i]: anom = False
        if anom: pred[i] = True
    TP = np.sum(pred*label_a); FP = np.sum(pred*(1-label_a)); FN = np.sum((1-pred)*label_a)
    p = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8); f1 = 2*p*r/(p+r+1e-8)
    try: auc = roc_auc_score(label_a, pred)
    except: auc = 0.5
    return f1, p, r, auc

def run_one(train, test, labels, feats, lr, lm, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    label_1d = (np.sum(labels, axis=1) >= 1).astype(float)
    w = 10
    model = EPAADNet(feats).double().to(DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(train, dtype=torch.float64)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
    crit = nn.MSELoss(); model.train()
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
            errs.append(((rec-dw[:,-1,:])**2).mean(dim=1).cpu().numpy())
        return np.concatenate(errs)
    ts = scores(train); ss = scores(test)
    return pa_f1(ts, ss, label_1d, lm)


def ci(arr):
    m = np.mean(arr); s = np.std(arr, ddof=1)
    t = stats.t.ppf(0.975, len(arr)-1); e = t*s/np.sqrt(len(arr))
    return m, s, m-e, m+e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"  EPAAD-Net 统计显著性: {args.n_runs} runs x {args.epochs} epochs")
    print(f"{'='*100}")

    all_stats = {}
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        train, test, labels = load_data(ds)
        feats = train.shape[1]

        f1s, aucs = [], []
        for s in range(1, args.n_runs+1):
            t0 = time.time()
            f1, pr, rc, auc = run_one(train, test, labels, feats, LR[ds], LM[ds], args.epochs, s)
            f1s.append(f1); aucs.append(auc)
            print(f"  seed={s:2d}: F1(PA)={f1:.4f}  AUC={auc:.4f}  ({time.time()-t0:.1f}s)")

        f1s = np.array(f1s); aucs = np.array(aucs)
        m1, s1, lo1, hi1 = ci(f1s)
        m2, s2, lo2, hi2 = ci(aucs)
        print(f"  => F1(PW) = {m1:.4f} +- {s1:.4f}  [95% CI: {lo1:.4f}, {hi1:.4f}]")
        print(f"     AUC    = {m2:.4f} +- {s2:.4f}  [95% CI: {lo2:.4f}, {hi2:.4f}]")
        all_stats[ds] = {'f1_mean': m1, 'f1_std': s1, 'f1_lo': lo1, 'f1_hi': hi1,
                          'auc_mean': m2, 'auc_std': s2, 'auc_lo': lo2, 'auc_hi': hi2}

    # LaTeX
    print(f"\n{'='*100}")
    print("  LaTeX 统计显著性表")
    print(f"{'='*100}")
    for ds, s in all_stats.items():
        print(f"{ds} & ${s['f1_mean']:.4f} \\pm {s['f1_std']:.4f}$ & "
              f"$[{s['f1_lo']:.4f}, {s['f1_hi']:.4f}]$ & "
              f"${s['auc_mean']:.4f} \\pm {s['auc_std']:.4f}$ \\\\")
