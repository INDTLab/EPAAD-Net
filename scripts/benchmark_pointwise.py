"""
================================================================================
Point-wise F1 对比: 基准 + EPAADNet/TimesNet/DCdetector 在 6 数据集上
无 PA (Point-Adjust), 纯逐点 F1
================================================================================
用法:
    py benchmark_pointwise.py --epochs 5
    py benchmark_pointwise.py --dataset SMD --epochs 10
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, math, time, json, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据集 ====================
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
    c = DATASETS[name]
    f = os.path.join(ROOT, 'processed', c[0])
    return (np.load(os.path.join(f, f'{c[1]}.npy')),
            np.load(os.path.join(f, f'{c[2]}.npy')),
            np.load(os.path.join(f, f'{c[3]}.npy')))

def to_windows(data, w=10):
    data = np.asarray(data)
    if data.ndim == 1: data = data.reshape(-1, 1)
    N, C = data.shape
    win = np.zeros((N, w, C), dtype=data.dtype)
    for i in range(N):
        if i >= w: win[i] = data[i - w:i]
        else:
            p = w - i; win[i, :p] = data[0:1]
            if i > 0: win[i, p:] = data[:i]
    return win


# ==================== EPAADNet/EPAAD-Net ====================
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
    """EPAADNet"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'EPAADNet'; self.n_feats = feats; self.n_window = nw
        self.l_tcn = Tcn_Local(feats, 4, 0.2); self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1); self.dec = TransformerDecoderLayer1(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src, tgt=None):
        if tgt is None: tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        return self.fcn(self.dec(tgt, self.sp(src + self.drop(s2.permute(2, 0, 1)))))


# ==================== TimesNet ====================
class PE(nn.Module):
    def __init__(self, d, dropout=0.1, ml=5000):
        super().__init__(); self.dp = nn.Dropout(p=dropout)
        pe = torch.zeros(ml, d); pos = torch.arange(0, ml, dtype=torch.float).unsqueeze(1)
        dt = torch.exp(torch.arange(0, d).float() * (-math.log(10000.0) / d))
        pe += torch.sin(pos*dt); pe += torch.cos(pos*dt); self.register_buffer('pe', pe.unsqueeze(0).transpose(0,1))
    def forward(self, x, pos=0): return self.dp(x+self.pe[pos:pos+x.size(0),:])

class InceptionV1(nn.Module):
    def __init__(self, i, o):
        super().__init__(); m = max(o//4,1)
        self.c1 = nn.Conv2d(i, m, 1); self.c3 = nn.Conv2d(i, m, 3, padding=1)
        self.c5 = nn.Conv2d(i, m, 5, padding=2)
        self.mp = nn.Sequential(nn.MaxPool2d(3,1,1), nn.Conv2d(i, o-3*m, 1))
        self.bn = nn.BatchNorm2d(o); self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.bn(torch.cat([self.c1(x),self.c3(x),self.c5(x),self.mp(x)],dim=1)))

class TimesBlock(nn.Module):
    def __init__(self, T, C, k=3):
        super().__init__(); self.T = T; self.k = min(k, T//2)
        self.conv = nn.Sequential(InceptionV1(C, C//2), nn.GELU(), InceptionV1(C//2, C))
        self.fw = nn.Parameter(torch.ones(self.k)/self.k)
    def forward(self, x):
        B, C, T = x.shape
        amps = torch.abs(torch.fft.rfft(x, dim=-1)).mean(dim=1)[:, 1:]
        ke = min(self.k, amps.shape[1]); _, ti = torch.topk(amps, ke, dim=-1)
        periods = T/(ti.float()+1.0); avg_p = periods.mean(dim=0)
        outs = []
        for i in range(ke):
            p = max(2, min(T, int(round(avg_p[i].item()))))
            pad = 0 if T%p==0 else p-(T%p); xp = x if pad==0 else F.pad(x, (0,pad))
            x2d = xp.reshape(B, C, p, (T+pad)//p); x1d = self.conv(x2d).reshape(B, C, T+pad)
            outs.append(x1d[:,:,:T] if pad>0 else x1d)
        w = F.softmax(self.fw[:ke], dim=0).view(1,1,1,-1)
        return torch.sum(torch.stack(outs, dim=-1)*w, dim=-1)

class TimesNetModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'TimesNet'; self.n_feats = feats; self.n_window = nw
        self.embed = nn.Linear(feats, 32); self.pe = PE(32, 0.1, nw)
        self.blocks = nn.ModuleList([TimesBlock(nw, 32, 3) for _ in range(2)])
        self.norm = nn.LayerNorm(32); self.op = nn.Linear(32, feats); self.fcn = nn.Sigmoid()
    def forward(self, x):
        B, T, C = x.shape
        x = self.embed(x).permute(1,0,2); x = self.pe(x).permute(1,2,0)
        for b in self.blocks: x = x + b(x)
        return self.fcn(self.op(self.norm(x.permute(0,2,1))))


# ==================== DCdetector ====================
class PatchEmb(nn.Module):
    def __init__(self, nw, pl, st, nf, dm):
        super().__init__(); self.pl = pl; self.st = st
        self.np = (nw - pl)//st + 1
        self.emb = nn.Linear(pl*nf, dm)
        self.pe = nn.Parameter(torch.randn(1, self.np, dm)*0.02)
    def forward(self, x):
        B, T, C = x.shape; pts = []
        for i in range(0, T-self.pl+1, self.st):
            pts.append(x[:, i:i+self.pl, :].reshape(B, -1))
        return self.emb(torch.stack(pts, dim=1)) + self.pe

class DAB(nn.Module):
    def __init__(self, dm, nf, nh=8, drop=0.1):
        super().__init__()
        self.pa = nn.MultiheadAttention(dm, nh, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(dm); self.d1 = nn.Dropout(drop)
        self.cp = nn.Linear(dm, nf)
        ch = 1
        for h in range(min(nh, nf), 0, -1):
            if nf % h == 0: ch = h; break
        self.ca = nn.MultiheadAttention(nf, ch, dropout=drop, batch_first=True)
        self.cb = nn.Linear(nf, dm); self.n2 = nn.LayerNorm(dm); self.d2 = nn.Dropout(drop)
        self.ffn = nn.Sequential(nn.Linear(dm, dm*4), nn.GELU(), nn.Dropout(drop), nn.Linear(dm*4, dm))
        self.n3 = nn.LayerNorm(dm); self.d3 = nn.Dropout(drop)
    def forward(self, x):
        a, _ = self.pa(x, x, x); x = self.n1(x + self.d1(a))
        c, _ = self.ca(self.cp(x), self.cp(x), self.cp(x)); x = self.n2(x + self.d2(self.cb(c)))
        return self.n3(x + self.d3(self.ffn(x)))

class DCdetectorModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'DCdetector'; self.n_feats = feats; self.n_window = nw
        self.pe = PatchEmb(nw, 3, 2, feats, 64); self.np = self.pe.np
        self.blocks = nn.ModuleList([DAB(64, feats, 8) for _ in range(2)])
        self.rec = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3*feats))
        self.fcn = nn.Sigmoid()
    def _fold(self, rp, B, T, C):
        rec = torch.zeros(B, T, C, device=rp.device); cnt = torch.zeros(B, T, 1, device=rp.device)
        for i in range(self.np):
            s = i*2; rec[:, s:s+3, :] += rp[:, i, :].reshape(B, 3, C); cnt[:, s:s+3, :] += 1
        return rec / cnt.clamp(min=1)
    def forward(self, x):
        B, T, C = x.shape; h = self.pe(x)
        for b in self.blocks: h = b(h)
        return self.fcn(self._fold(self.rec(h), B, T, C))


# 基线结果: 运行 main.py 获取 (已内置所有基线的正确训练逻辑)
# 示例: py main.py --model TranAD --dataset SMD --retrain
# 详见下方 run_all() 末尾的说明

# ==================== 评估 (纯 Point-wise) ====================
class SPOT:
    def __init__(self, q=1e-4): self.q = q
    def fit(self, a, b): self.a = a; self.b = b; return self
    def initialize(self, level=0.98, **kw):
        th = np.percentile(self.a, level*100); self.th = th
        ex = self.a[self.a > th] - th
        if len(ex) == 0: ex = np.array([np.percentile(self.a, 99) - th])
        self.extrema = ex; return self
    def run(self, **kw):
        return {'alarms': [1 if s > self.th else 0 for s in self.b],
                'thresholds': [self.th]*len(self.b)}


def pointwise_f1(init_score, score, label, lm_params):
    """纯逐点 F1, 无 PA"""
    lms = lm_params[0]
    while True:
        try:
            s = SPOT(); s.fit(init_score, score); s.initialize(level=lms, min_extrema=False)
        except: lms *= 0.999
        else: break
    ret = s.run(); th = np.mean(ret['thresholds']) * lm_params[1]

    pred = (score > th).astype(float)
    TP = np.sum(pred * label); FP = np.sum(pred * (1-label)); FN = np.sum((1-pred) * label)
    p = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8)
    f1 = 2*p*r/(p+r+1e-8)
    try: auc = roc_auc_score(label, score)
    except: auc = 0.5
    return f1, p, r, auc


# ==================== 训练 ====================
def train_epaad(model, train_data, feats, lr, epochs):
    w = model.n_window
    loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float64)), batch_size=128, shuffle=True)
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
    return model

def train_ae(model, train_data, feats, lr, epochs, flatten_input=False):
    """Autoencoder 风格训练"""
    w = model.n_window
    dtype = next(model.parameters()).dtype
    loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=dtype)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
    crit = nn.MSELoss(); model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            d = batch.to(DEVICE).to(dtype)
            dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=dtype).to(DEVICE)
            if flatten_input:
                dw = dw.reshape(dw.shape[0], -1)       # (B, T*C)
            loss = crit(model(dw), dw)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model

@torch.no_grad()
def scores_epaad(model, data, feats):
    w = model.n_window; model.eval(); errs = []
    for (batch,) in DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float64)), batch_size=128):
        d = batch.to(DEVICE).double()
        dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=torch.float64).to(DEVICE)
        inp = dw.permute(1,0,2); rec = model(inp, inp[-1:,:,:]).squeeze(0)
        errs.append(((rec - dw[:,-1,:])**2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)

@torch.no_grad()
def scores_ae(model, data, feats, flatten_input=False):
    w = model.n_window; model.eval(); errs = []
    dtype = next(model.parameters()).dtype
    for (batch,) in DataLoader(TensorDataset(torch.tensor(data, dtype=dtype)), batch_size=128):
        d = batch.to(DEVICE).to(dtype)
        dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=dtype).to(DEVICE)
        if flatten_input:
            dw_flat = dw.reshape(dw.shape[0], -1)
            rec = model(dw_flat)
            errs.append(((rec.reshape(dw.shape)-dw)**2).mean(dim=1).cpu().numpy())
        else:
            rec = model(dw)
            errs.append(((rec-dw)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


# ==================== 主流程 ====================
def run_all(epochs=5):
    results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")
        train, test, labels = load_data(ds_name)
        feats = train.shape[1]
        label_1d = (np.sum(labels, axis=1) >= 1).astype(float)
        lr = LR[ds_name]; lm = LM[ds_name]

        results[ds_name] = {}

        # --- EPAADNet (float64) ---
        print("  EPAAD-Net...", end=' ', flush=True)
        m = EPAADNet(feats).double().to(DEVICE)
        m = train_epaad(m, train, feats, lr, epochs)
        ts = scores_epaad(m, train, feats); ss = scores_epaad(m, test, feats)
        f1, pr, rc, auc = pointwise_f1(ts, ss, label_1d, lm)
        results[ds_name]['EPAADNet'] = {'f1': f1, 'auc': auc, 'prec': pr, 'rec': rc}
        print(f"F1(PW)={f1:.4f}")

        # --- TimesNet (float64) ---
        print("  TimesNet...", end=' ', flush=True)
        m2 = TimesNetModel(feats).double().to(DEVICE)
        m2 = train_ae(m2, train, feats, lr, epochs)
        ts2 = np.mean(scores_ae(m2, train, feats), axis=1)
        ss2 = np.mean(scores_ae(m2, test, feats), axis=1)
        f2, pr2, rc2, auc2 = pointwise_f1(ts2, ss2, label_1d, lm)
        results[ds_name]['TimesNet'] = {'f1': f2, 'auc': auc2, 'prec': pr2, 'rec': rc2}
        print(f"F1(PW)={f2:.4f}")

        # --- DCdetector (float32) ---
        print("  DCdetector...", end=' ', flush=True)
        m3 = DCdetectorModel(feats).to(DEVICE)
        m3 = train_ae(m3, train, feats, lr, epochs)
        ts3 = np.mean(scores_ae(m3, train, feats), axis=1)
        ss3 = np.mean(scores_ae(m3, test, feats), axis=1)
        f3, pr3, rc3, auc3 = pointwise_f1(ts3, ss3, label_1d, lm)
        results[ds_name]['DCdetector'] = {'f1': f3, 'auc': auc3, 'prec': pr3, 'rec': rc3}
        print(f"F1(PW)={f3:.4f}")


    # ==================== 汇总表 ====================
    print(f"\n\n{'='*80}")
    print("  Point-wise F1 汇总")
    print(f"{'='*80}")
    # 收集所有出现的模型名
    all_models = set()
    for ds_r in results.values():
        all_models.update(ds_r.keys())
    models = sorted(all_models, key=lambda m: (m != 'EPAADNet', m != 'TranAD', m))
    datasets = list(DATASETS.keys())

    # 控制台表
    header = f"{'Method':<14}"
    for d in datasets: header += f" {d:<10}"
    print(header)
    print("-" * (14 + 11*len(datasets)))
    for m in models:
        row = f"{m:<14}"
        for d in datasets:
            v = results[d].get(m)
            row += f" {v['f1']:<10.4f}" if v else f" {'--':<10}"
        print(row)

    # LaTeX 表
    print(f"\n--- LaTeX Point-wise F1 ---")
    for m in models:
        vals = []
        for d in datasets:
            v = results[d].get(m)
            vals.append(f"{v['f1']:.4f}" if v else "--")
        print(f"{m} & {' & '.join(vals)} \\\\")

    # LaTeX AUC
    print(f"\n--- LaTeX AUC ---")
    for m in models:
        vals = ' & '.join([f"{results[d][m]['auc']:.4f}" for d in datasets])
        print(f"{m} & {vals} \\\\")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/pointwise_f1_results.json')
    args = parser.parse_args()

    res = run_all(args.epochs)

    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    with open(args.output, 'w') as f:
        json.dump(convert(res), f, indent=2)
    print(f"\n结果已保存到 {args.output}")
