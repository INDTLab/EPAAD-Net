"""
================================================================================
12 模型 Point-wise F1 对比 — 一次性跑完所有数据集
================================================================================
用法: py run_baselines.py --epochs 5
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
        if i >= w:
            win[i] = data[i - w:i]
        else:
            pad = w - i
            win[i, :pad] = np.tile(data[0:1], (pad, 1))
            if i > 0:
                win[i, pad:] = data[0:i].reshape(i, C)
    return win

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
        return {'alarms': [1 if s > self.th else 0 for s in self.b],
                'thresholds': [self.th]*len(self.b)}

def pa_f1(init_score, score, label, lm_params):
    """标准 PA-F1 (Point-Adjust, 与 main.py 一致)"""
    lms = lm_params[0]
    while True:
        try:
            s = SPOT(); s.fit(init_score, score); s.initialize(level=lms, min_extrema=False)
        except: lms *= 0.999
        else: break
    th = np.mean(s.run()['thresholds']) * lm_params[1]
    pred = (score > th).astype(float)
    label = np.asarray(label)
    act = label > 0.1; anom = False
    for i in range(len(pred)):
        if act[i] and pred[i] and not anom:
            anom = True
            for j in range(i, 0, -1):
                if not act[j]: break
                elif not pred[j]: pred[j] = True
        elif not act[i]: anom = False
        if anom: pred[i] = True
    TP = np.sum(pred*label); FP = np.sum(pred*(1-label)); FN = np.sum((1-pred)*label)
    p = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8)
    f1 = 2*p*r/(p+r+1e-8)
    try: auc = roc_auc_score(label, score)
    except: auc = 0.5
    return f1, p, r, auc

# ==================== 共享模块 ====================
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
class PE(nn.Module):
    def __init__(self, d, dropout=0.1, ml=5000):
        super().__init__(); self.dp = nn.Dropout(p=dropout)
        pe = torch.zeros(ml, d); pos = torch.arange(0, ml, dtype=torch.float).unsqueeze(1)
        dt = torch.exp(torch.arange(0, d).float() * (-math.log(10000.0) / d))
        pe += torch.sin(pos*dt); pe += torch.cos(pos*dt); self.register_buffer('pe', pe.unsqueeze(0).transpose(0,1))
    def forward(self, x, pos=0): return self.dp(x+self.pe[pos:pos+x.size(0),:])

# ==================== 模型定义 ====================

# --- 1. EPAAD-Net (Ours) ---
class EPAADNet(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'EPAADNet'; self.n_feats = feats; self.n_window = nw
        self.l_tcn = Tcn_Local(feats, 4, 0.2); self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.ae1 = nn.Sequential(nn.Linear(feats, feats//3), nn.ReLU(), nn.Linear(feats//3, feats), nn.ReLU())
        self.ae2 = nn.Sequential(nn.Linear(feats, feats//3), nn.ReLU(), nn.Linear(feats//3, feats), nn.ReLU())
        self.mha = nn.MultiheadAttention(feats, feats, dropout=0.1)
        self.l1 = nn.Linear(feats, 16); self.dr = nn.Dropout(0.1); self.l2 = nn.Linear(16, feats)
        self.d1 = nn.Dropout(0.1); self.d2 = nn.Dropout(0.1); self.d3 = nn.Dropout(0.1); self.d4 = nn.Dropout(0.1)
        self.act = nn.LeakyReLU(True); self.fcn = nn.Sigmoid()
    def decoder(self, tgt, mem):
        tgt = tgt + self.d1(self.ae1(tgt)); tgt = tgt + self.d2(self.ae2(tgt))
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.dr(self.act(self.l1(tgt)))))
    def forward(self, src, tgt=None):
        if tgt is None: tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        return self.fcn(self.decoder(tgt, self.sp(src + self.drop(s2.permute(2, 0, 1)))))

# --- 2. TranAD (VLDB'22) ---
class TranADModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'TranAD'; self.n_feats = feats; self.n_window = nw
        self.embed = nn.Linear(feats, 32)
        self.pe = PE(32, 0.1, nw)
        el = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1)
        self.enc = nn.TransformerEncoder(el, 2)
        dl = nn.TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1)
        self.dec = nn.TransformerDecoder(dl, 1)
        self.out = nn.Sequential(nn.Linear(32, feats), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        h = self.embed(x).permute(1, 0, 2)     # (T, B, 32)
        h = self.pe(h)
        mem = self.enc(h)                        # (T, B, 32)
        out = self.dec(h, mem)                   # (T, B, 32)
        return self.out(out).permute(1, 0, 2)    # (B, T, C)

# --- 3. USAD (KDD'20) ---
class USADModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'USAD'; self.n_feats = feats; self.n_window = nw
        n = feats*max(nw, 1); nh = 16; nl = 5
        self.enc = nn.Sequential(nn.Linear(n, nh), nn.ReLU(), nn.Linear(nh, nh), nn.ReLU(), nn.Linear(nh, nl), nn.ReLU())
        self.d1 = nn.Sequential(nn.Linear(nl, nh), nn.ReLU(), nn.Linear(nh, nh), nn.ReLU(), nn.Linear(nh, n), nn.Sigmoid())
        self.d2 = nn.Sequential(nn.Linear(nl, nh), nn.ReLU(), nn.Linear(nh, nh), nn.ReLU(), nn.Linear(nh, n), nn.Sigmoid())
    def forward(self, x):
        z = self.enc(x.reshape(x.shape[0], -1))
        return self.d2(z).reshape(x.shape)

# --- 4. OmniAnomaly (KDD'19) ---
class OmniAnomalyModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'OmniAnomaly'; self.n_feats = feats; self.n_window = nw
        hid = 32; lat = 8
        self.gru = nn.GRU(feats, hid, 2, batch_first=True)
        self.enc_mu = nn.Linear(hid*nw, lat)
        self.enc_lv = nn.Linear(hid*nw, lat)
        self.dec = nn.Sequential(nn.Linear(lat, hid), nn.PReLU(), nn.Linear(hid, hid), nn.PReLU(),
                                  nn.Linear(hid, feats*nw), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape; out, _ = self.gru(x)
        h = out.reshape(B, -1)  # (B, hid*T)
        mu, lv = self.enc_mu(h), self.enc_lv(h)
        z = mu + torch.randn_like(lv) * torch.exp(0.5*lv)
        return self.dec(z).reshape(B, T, C)

# --- 5. MAD-GAN (ICANN'19) ---
class MADGANModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'MAD-GAN'; self.n_feats = feats; self.n_window = nw
        n = feats*nw; nh = 16
        self.gen = nn.Sequential(nn.Linear(n, nh), nn.LeakyReLU(), nn.Linear(nh, nh), nn.LeakyReLU(), nn.Linear(nh, n), nn.Sigmoid())
    def forward(self, x):
        return self.gen(x.reshape(x.shape[0], -1)).reshape(x.shape)

# --- 6. MSCRED (AAAI'19) ---
class MSCREDModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'MSCRED'; self.n_feats = feats; self.n_window = nw
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 1, 1), nn.ReLU(),
                                  nn.ConvTranspose2d(32, 1, 3, 1, 1), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        z = self.conv2(self.conv1(x.view(B, 1, C, T))); rec = self.dec(z)
        return rec.view(B, C*T).reshape(x.shape) if rec.shape[-1] == T else rec.view(B, T, -1)[:, :, :C]

# --- 7. CAE-M (TKDE'21) ---
class CAEMModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'CAE-M'; self.n_feats = feats; self.n_window = nw
        self.enc = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.Sigmoid(),
                                  nn.Conv2d(8, 16, 3, 1, 1), nn.Sigmoid(),
                                  nn.Conv2d(16, 32, 3, 1, 1), nn.Sigmoid())
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 4, 3, 1, 1), nn.Sigmoid(),
                                  nn.ConvTranspose2d(4, 4, 3, 1, 1), nn.Sigmoid(),
                                  nn.ConvTranspose2d(4, 1, 3, 1, 1), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        z = self.enc(x.view(B, 1, C, T))
        return self.dec(z).view(B, -1)[:, :C*T].reshape(B, T, C)

# --- 8. GDN (AAAI'21) — 无 dgl 近似 ---
class GDNModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'GDN'; self.n_feats = feats; self.n_window = nw
        self.att_w = nn.Sequential(nn.Linear(feats*nw, 16), nn.LeakyReLU(),
                                    nn.Linear(16, 16), nn.LeakyReLU(), nn.Linear(16, nw), nn.Softmax(dim=-1))
        self.fcn = nn.Sequential(nn.Linear(feats, 16), nn.LeakyReLU(), nn.Linear(16, feats*nw), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        w = self.att_w(x.reshape(B, -1)).unsqueeze(-1)  # (B, nw, 1)
        xw = (x * w).sum(dim=1)                           # (B, C)
        return self.fcn(xw).reshape(B, T, C)

# --- 9. MTAD-GAT (ICDM'20) — 无 dgl 近似 ---
class MTADGATModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'MTAD-GAT'; self.n_feats = feats; self.n_window = nw
        self.att_f = nn.Sequential(nn.Linear(feats, 1), nn.Sigmoid())
        self.att_t = nn.Sequential(nn.Linear(nw, 1), nn.Sigmoid())
        self.proj = nn.Linear(feats, feats)
        self.fcn = nn.Sigmoid()
    def forward(self, x):
        B, T, C = x.shape
        # Feature attention
        af = self.att_f(x)                     # (B, T, 1)
        hf = (x * af).sum(dim=1, keepdim=True)  # (B, 1, C)
        # Time attention
        at = self.att_t(x.permute(0, 2, 1))     # (B, C, 1)
        ht = (x.permute(0, 2, 1) * at).sum(dim=2, keepdim=True).permute(0, 2, 1)  # (B, 1, C)
        h = self.proj(hf + ht)                  # (B, 1, C)
        return self.fcn(h.expand(-1, T, -1))    # (B, T, C)

# --- 10. DTAAD ---
class DTAADModel(nn.Module):
    def __init__(self, feats, nw=10):
        super().__init__(); self.name = 'DTAAD'; self.n_feats = feats; self.n_window = nw
        self.embed = nn.Linear(feats, 64)
        el = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc1 = nn.TransformerEncoder(el, 2)
        self.enc2 = nn.TransformerEncoder(el, 2)
        self.fcn = nn.Sequential(nn.Linear(128, feats), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        h = self.embed(x)
        z1 = self.enc1(h)  # view 1
        z2 = self.enc2(h + torch.randn_like(h)*0.01)  # view 2 (noised)
        z = torch.cat([z1, z2], dim=-1)  # (B, T, 128)
        return self.fcn(z)

# --- 11. TimesNet ---
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

# --- 12. DCdetector ---
class PatchEmb(nn.Module):
    def __init__(self, nw, pl, st, nf, dm):
        super().__init__(); self.pl = pl; self.st = st
        self.np = (nw - pl)//st + 1
        self.emb = nn.Linear(pl*nf, dm); self.pos = nn.Parameter(torch.randn(1, self.np, dm)*0.02)
    def forward(self, x):
        B, T, C = x.shape; pts = []
        for i in range(0, T-self.pl+1, self.st): pts.append(x[:, i:i+self.pl, :].reshape(B, -1))
        return self.emb(torch.stack(pts, dim=1)) + self.pos
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

# ==================== 全部模型注册 ====================
ALL_MODELS = [
    EPAADNet, TranADModel, USADModel, OmniAnomalyModel, MADGANModel,
    MSCREDModel, CAEMModel, GDNModel, MTADGATModel, DTAADModel,
    TimesNetModel, DCdetectorModel,
]

# ==================== 训练 ====================
def train_model(model, train_data, feats, lr, epochs, epaad_style=False):
    """统一训练: epaad_style 用于 EPAAD-Net (T,B,C) 格式, 其余 AE 风格"""
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
            if epaad_style:
                inp = dw.permute(1,0,2); tgt = inp[-1:,:,:]
                loss = crit(model(inp, tgt).squeeze(0), dw[:,-1,:])
            else:
                loss = crit(model(dw), dw)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model

@torch.no_grad()
def scores_model(model, data, feats, epaad_style=False):
    w = model.n_window; model.eval(); errs = []
    dtype = next(model.parameters()).dtype
    for (batch,) in DataLoader(TensorDataset(torch.tensor(data, dtype=dtype)), batch_size=128):
        d = batch.to(DEVICE).to(dtype)
        dw = torch.tensor(to_windows(d.cpu().numpy(), w), dtype=dtype).to(DEVICE)
        if epaad_style:
            inp = dw.permute(1,0,2); rec = model(inp, inp[-1:,:,:]).squeeze(0)
            errs.append(((rec - dw[:,-1,:])**2).mean(dim=1).cpu().numpy())
        else:
            rec = model(dw)
            errs.append(((rec-dw)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


# ==================== 主流程 ====================
def run_all(epochs=5, model_filter=None, dataset_filter=None):
    results = {}

    for ds_name in DATASETS:
        if dataset_filter and ds_name != dataset_filter:
            continue
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")
        train, test, labels = load_data(ds_name)
        feats = train.shape[1]
        label_1d = (np.sum(labels, axis=1) >= 1).astype(float)
        lr = LR[ds_name]; lm = LM[ds_name]
        results[ds_name] = {}

        for Cls in ALL_MODELS:
            try:
                m = Cls(feats)
            except Exception as e:
                print(f"  SKIP {Cls.__name__}: init failed ({e})"); continue

            if model_filter and m.name != model_filter:
                continue

            is_epaad = (m.name == 'EPAADNet')
            print(f"  {m.name}...", end=' ', flush=True)
            try:
                m = m.to(DEVICE).double() if is_epaad else m.to(DEVICE)
                m = train_model(m, train, feats, lr, epochs, epaad_style=is_epaad)
                ts = scores_model(m, train, feats, epaad_style=is_epaad)
                ss = scores_model(m, test, feats, epaad_style=is_epaad)
                # Average over feature dim for per-feature score models
                if ts.ndim > 1: ts = np.mean(ts, axis=1)
                if ss.ndim > 1: ss = np.mean(ss, axis=1)
                f1, pr, rc, auc = pa_f1(ts, ss, label_1d, lm)
                results[ds_name][m.name] = {'f1': f1, 'auc': auc, 'prec': pr, 'rec': rc}
                print(f"F1(PA)={f1:.4f}")
            except Exception as e:
                print(f"ERR: {e}")

    # ==================== 汇总 ====================
    datasets = list(DATASETS.keys())
    models_seen = set()
    for ds_r in results.values(): models_seen.update(ds_r.keys())
    model_order = [c(1).name for c in ALL_MODELS if c(1).name in models_seen]

    print(f"\n\n{'='*100}")
    print("  Point-wise F1 汇总")
    print(f"{'='*100}")
    hdr = f"{'Method':<16}"
    for d in datasets: hdr += f" {d:<10}"
    print(hdr); print("-" * (16 + 11*len(datasets)))
    for mn in model_order:
        row = f"{mn:<16}"
        for d in datasets:
            v = results[d].get(mn, {})
            f = v.get('f1', float('nan'))
            row += f" {f:<10.4f}" if not np.isnan(f) else f" {'--':<10}"
        print(row)

    # LaTeX
    print(f"\n--- LaTeX F1 (PA) ---")
    for mn in model_order:
        vals = []
        for d in datasets:
            v = results[d].get(mn, {}); f = v.get('f1', float('nan'))
            vals.append(f"{f:.4f}" if not np.isnan(f) else "--")
        print(f"{mn} & {' & '.join(vals)} \\\\")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model', type=str, default=None, help='Run only this model')
    parser.add_argument('--dataset', type=str, default=None, help='Run only this dataset')
    parser.add_argument('--output', type=str, default='results/all_models_f1.json')
    args = parser.parse_args()
    res = run_all(args.epochs, args.model, args.dataset)

    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    with open(args.output, 'w') as f:
        json.dump(convert(res), f, indent=2)
    print(f"\n结果已保存到 {args.output}")
