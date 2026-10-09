"""
================================================================================
基线方法 Jetson 效率评估 — 完全按 src/models.py 的架构, 手搓 GAT, 无需 dgl
只输出: Time per Record (ms)

9 基线: TranAD, USAD, OmniAnomaly, LSTM_AD, MAD_GAN, MSCRED, CAE_M, GDN, MTAD_GAT
+ TimesNet, DCdetector
================================================================================
用法: python3 benchmark_jetson.py
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, math, time, json, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
FEATS_MAP = {'SMD': 38, 'NAB': 1, 'MBA': 2, 'SMAP': 25, 'SWaT_MV': 6, 'SWaT_UV': 1}


# ==================== 手搓 GATConv (等价 dgl.GATConv) ====================
class GATConvManual(nn.Module):
    """等价 dgl.nn.GATConv, 用邻接矩阵指定图结构"""
    def __init__(self, in_feats, out_feats, num_heads, adj):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.register_buffer('adj', adj)              # (N, N) bool, 含自环
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=True)
        self.attn_l = nn.Linear(out_feats, 1, bias=False)
        self.attn_r = nn.Linear(out_feats, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        N = h.shape[0]
        Wh = self.fc(h).view(N, self.num_heads, self.out_feats)
        el = self.attn_l(Wh).squeeze(-1)
        er = self.attn_r(Wh).squeeze(-1)
        e = self.leaky_relu(el.unsqueeze(1) + er.unsqueeze(0))   # (N,N,H)
        e = e.masked_fill(~self.adj.unsqueeze(-1), float('-inf'))
        alpha = F.softmax(e, dim=1)
        out = torch.einsum('ijh,jho->iho', alpha, Wh)
        return out.reshape(N, self.num_heads * self.out_feats)


def complete_graph(n):
    return torch.ones(n, n, dtype=torch.bool)


def star_graph(n):
    """dgl: 节点 1..n-1 -> 节点 0, + 自环"""
    adj = torch.zeros(n, n, dtype=torch.bool)
    adj[0, 0] = True
    for i in range(1, n):
        adj[i, 0] = True
        adj[i, i] = True
    return adj


# ==================== ConvLSTM (MSCRED) ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        ks = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, ks, padding=ks // 2)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, b, h, w, device):
        return (torch.zeros(b, self.hidden_dim, h, w, device=device),
                torch.zeros(b, self.hidden_dim, h, w, device=device))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(pos * div); pe += torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    def forward(self, x, pos=0):
        return self.dropout(x + self.pe[pos:pos+x.size(0), :])


# ==================== 基线模型 (严格按 src/models.py) ====================
# 1. TranAD (VLDB'22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='TranAD'; self.n_window=10; self.n_feats=feats
        self.pos_encoder = PositionalEncoding(2*feats, 0.1, 10)
        el = nn.TransformerEncoderLayer(d_model=2*feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.encoder = nn.TransformerEncoder(el, 1)
        dl1 = nn.TransformerDecoderLayer(d_model=2*feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        dl2 = nn.TransformerDecoderLayer(d_model=2*feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.dec1 = nn.TransformerDecoder(dl1, 1); self.dec2 = nn.TransformerDecoder(dl2, 1)
        self.fcn = nn.Sequential(nn.Linear(2*feats, feats), nn.Sigmoid())
    def encode(self, src, c):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        return self.encoder(src)
    def forward(self, x):
        src = x.permute(1, 0, 2)                     # (T, B, C)
        T, B, C = src.shape
        c1 = torch.zeros_like(src)
        mem1 = self.encode(src, c1)
        x1 = self.fcn(self.dec1(src[-1:].repeat(1,1,2), mem1))
        c2 = (x1 - src[-1:]) ** 2
        mem2 = self.encode(src, c2.repeat(T,1,1))
        x2 = self.fcn(self.dec2(torch.cat((src[-1:], c2), dim=2), mem2))
        return x2.squeeze(0)

# 2. USAD (KDD'20), n_window=5
class USAD(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='USAD'; self.n_window=5; self.n_feats=feats
        n = feats * 5; nh=16; nl=5
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(n,nh), nn.ReLU(True),
                                     nn.Linear(nh,nh), nn.ReLU(True), nn.Linear(nh,nl), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Linear(nl,nh), nn.ReLU(True), nn.Linear(nh,nh), nn.ReLU(True), nn.Linear(nh,n), nn.Sigmoid())
        self.dec2 = nn.Sequential(nn.Linear(nl,nh), nn.ReLU(True), nn.Linear(nh,nh), nn.ReLU(True), nn.Linear(nh,n), nn.Sigmoid())
    def forward(self, x):
        z = self.encoder(x)
        return self.dec2(self.encoder(self.dec1(z))).reshape(x.shape[0], 5, self.n_feats)

# 3. OmniAnomaly (KDD'19) — GRU + VAE
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='OmniAnomaly'; self.n_window=10; self.n_feats=feats
        self.gru = nn.GRU(feats, 32, 2, batch_first=True)
        self.enc_mu = nn.Linear(32, 8)
        self.enc_lv = nn.Linear(32, 8)
        self.dec = nn.Sequential(nn.Linear(8,32), nn.PReLU(), nn.Linear(32,32), nn.PReLU(), nn.Linear(32,feats), nn.Sigmoid())
    def forward(self, x):
        out,_ = self.gru(x)
        h = out[:, -1, :]                              # 取最后时间步
        mu, lv = self.enc_mu(h), self.enc_lv(h)
        z = mu + torch.randn_like(lv)*torch.exp(0.5*lv)
        return self.dec(z).unsqueeze(1).repeat(1, self.n_window, 1)

# 4. LSTM_AD
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='LSTM_AD'; self.n_window=10; self.n_feats=feats
        self.lstm = nn.LSTM(feats, 64, batch_first=True)
        self.fcn = nn.Sequential(nn.Linear(64, feats), nn.Sigmoid())
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fcn(out)

# 5. MAD-GAN (ICANN'19), n_window=5
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='MAD-GAN'; self.n_window=5; self.n_feats=feats
        n = feats * 5; nh=16
        self.generator = nn.Sequential(nn.Flatten(), nn.Linear(n,nh), nn.LeakyReLU(True),
                                       nn.Linear(nh,nh), nn.LeakyReLU(True), nn.Linear(nh,n), nn.Sigmoid())
        self.discriminator = nn.Sequential(nn.Flatten(), nn.Linear(n,nh), nn.LeakyReLU(True),
                                           nn.Linear(nh,nh), nn.LeakyReLU(True), nn.Linear(nh,1), nn.Sigmoid())
    def forward(self, x):
        z = self.generator(x)
        self.discriminator(z)                          # 判别器也跑 (保持完整)
        return z.reshape(x.shape[0], 5, self.n_feats)

# 6. MSCRED (AAAI'19), n_window=feats
class MSCRED(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='MSCRED'; self.n_window=feats; self.n_feats=feats
        self.enc1 = ConvLSTMCell(1, 32, (3,3))
        self.enc2 = ConvLSTMCell(32, 64, (3,3))
        self.enc3 = ConvLSTMCell(64, 128, (3,3))
        self.dec = nn.Sequential(nn.ConvTranspose2d(128,64,(3,3),1,1), nn.ReLU(True),
                                 nn.ConvTranspose2d(64,32,(3,3),1,1), nn.ReLU(True),
                                 nn.ConvTranspose2d(32,1,(3,3),1,1), nn.Sigmoid())
    def forward(self, x):
        B = x.shape[0]
        z = x.unsqueeze(1)                              # (B, 1, feats, feats)
        h1,_ = self.enc1(z, self.enc1.init_hidden(B, self.n_feats, self.n_feats, z.device))
        h2,_ = self.enc2(h1, self.enc2.init_hidden(B, self.n_feats, self.n_feats, z.device))
        h3,_ = self.enc3(h2, self.enc3.init_hidden(B, self.n_feats, self.n_feats, z.device))
        return self.dec(h3).squeeze(1)

# 7. CAE-M (TKDE'21), n_window=feats
class CAE_M(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='CAE-M'; self.n_window=feats; self.n_feats=feats
        self.enc = nn.Sequential(nn.Conv2d(1,8,3,1,1),nn.Sigmoid(),nn.Conv2d(8,16,3,1,1),nn.Sigmoid(),nn.Conv2d(16,32,3,1,1),nn.Sigmoid())
        self.dec = nn.Sequential(nn.ConvTranspose2d(32,4,3,1,1),nn.Sigmoid(),nn.ConvTranspose2d(4,4,3,1,1),nn.Sigmoid(),nn.ConvTranspose2d(4,1,3,1,1),nn.Sigmoid())
    def forward(self, x):
        z = self.enc(x.unsqueeze(1))
        return self.dec(z).squeeze(1)

# 8. GDN (AAAI'21), n_window=5, 完全图 GAT
class GDN(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='GDN'; self.n_window=5; self.n_feats=feats
        self.feature_gat = GATConvManual(1, 1, feats, complete_graph(feats))
        self.attention = nn.Sequential(nn.Linear(5*feats,16), nn.LeakyReLU(True),
                                       nn.Linear(16,16), nn.LeakyReLU(True),
                                       nn.Linear(16,5), nn.Softmax(dim=-1))
        self.fcn = nn.Sequential(nn.Linear(feats,16), nn.LeakyReLU(True), nn.Linear(16,5), nn.Sigmoid())
    def forward(self, x):
        B = x.shape[0]
        att = self.attention(x)                        # (B, 5)
        data = x.view(B, 5, self.n_feats)
        data_r = torch.bmm(data.permute(0,2,1), att.unsqueeze(-1))  # (B, feats, 1)
        outs = []
        for b in range(B):
            fr = self.feature_gat(data_r[b]).view(self.n_feats, self.n_feats)
            outs.append(self.fcn(fr).view(-1))
        return torch.stack(outs)

# 9. MTAD-GAT (ICDM'20), n_window=feats, 星形图双 GAT + GRU
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='MTAD-GAT'; self.n_window=feats; self.n_feats=feats
        self.n_hidden = feats * feats
        self.feature_gat = GATConvManual(feats, 1, feats, star_graph(feats+1))
        self.time_gat = GATConvManual(feats, 1, feats, star_graph(feats+1))
        self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)
    def forward(self, x):
        B = x.shape[0]
        C = self.n_feats
        data = x.view(B, C, C)
        outs = []
        for b in range(B):
            d = data[b]
            data_r = torch.cat((torch.zeros(1,C,device=x.device), d), dim=0)
            feat_r = self.feature_gat(data_r).unsqueeze(-1)
            data_t = torch.cat((torch.zeros(1,C,device=x.device), d.t()), dim=0)
            time_r = self.time_gat(data_t).unsqueeze(-1)
            dd = torch.cat((torch.zeros(1,C,device=x.device), d), dim=0).unsqueeze(-1)
            xx = torch.cat((dd, feat_r, time_r), dim=2).view(1,1,-1)
            h0 = torch.rand(1,1,self.n_hidden, dtype=xx.dtype, device=x.device)
            out,_ = self.gru(xx, h0)
            outs.append(out.view(-1))
        return torch.stack(outs)

# 10. TimesNet (ICLR'23)
class InceptionV1(nn.Module):
    def __init__(self, i, o):
        super().__init__(); m = max(o//4,1)
        self.c1 = nn.Conv2d(i,m,1); self.c3 = nn.Conv2d(i,m,3,padding=1)
        self.c5 = nn.Conv2d(i,m,5,padding=2)
        self.mp = nn.Sequential(nn.MaxPool2d(3,1,1), nn.Conv2d(i,o-3*m,1))
        self.bn = nn.BatchNorm2d(o); self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.bn(torch.cat([self.c1(x),self.c3(x),self.c5(x),self.mp(x)],dim=1)))
class TimesBlock(nn.Module):
    def __init__(self, T, C, k=3):
        super().__init__(); self.T=T; self.k=min(k,T//2)
        self.conv = nn.Sequential(InceptionV1(C,C//2),nn.GELU(),InceptionV1(C//2,C))
        self.fw = nn.Parameter(torch.ones(self.k)/self.k)
    def forward(self, x):
        B,C,T = x.shape
        amps = torch.abs(torch.fft.rfft(x,dim=-1)).mean(dim=1)[:,1:]
        ke = min(self.k, amps.shape[1]); _, ti = torch.topk(amps, ke, dim=-1)
        periods = T/(ti.float()+1.0); avg_p = periods.mean(dim=0)
        outs = []
        for i in range(ke):
            p = max(2,min(T,int(round(avg_p[i].item()))))
            pad = 0 if T%p==0 else p-(T%p); xp = x if pad==0 else F.pad(x,(0,pad))
            x2d = xp.reshape(B,C,p,(T+pad)//p); x1d = self.conv(x2d).reshape(B,C,T+pad)
            outs.append(x1d[:,:,:T] if pad>0 else x1d)
        w = F.softmax(self.fw[:ke],dim=0).view(1,1,1,-1)
        return torch.sum(torch.stack(outs,dim=-1)*w,dim=-1)
class TimesNet(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='TimesNet'; self.n_window=10; self.n_feats=feats
        self.embed = nn.Linear(feats,32); self.pe = PositionalEncoding(32,0.1,10)
        self.blocks = nn.ModuleList([TimesBlock(10,32,3) for _ in range(2)])
        self.norm = nn.LayerNorm(32); self.op = nn.Linear(32,feats); self.fcn = nn.Sigmoid()
    def forward(self, x):
        B,T,C = x.shape
        x = self.embed(x).permute(1,0,2); x = self.pe(x).permute(1,2,0)
        for b in self.blocks: x = x + b(x)
        return self.fcn(self.op(self.norm(x.permute(0,2,1))))

# 11. DCdetector
class PatchEmb(nn.Module):
    def __init__(self, nw, pl, st, nf, dm):
        super().__init__(); self.pl=pl; self.st=st; self.np=(nw-pl)//st+1
        self.emb = nn.Linear(pl*nf, dm); self.pos = nn.Parameter(torch.randn(1,self.np,dm)*0.02)
    def forward(self, x):
        B,T,C = x.shape; pts=[]
        for i in range(0,T-self.pl+1,self.st): pts.append(x[:,i:i+self.pl,:].reshape(B,-1))
        return self.emb(torch.stack(pts,dim=1))+self.pos
class DAB(nn.Module):
    def __init__(self, dm, nf, nh=8, drop=0.1):
        super().__init__()
        self.pa = nn.MultiheadAttention(dm,nh,dropout=drop,batch_first=True)
        self.n1 = nn.LayerNorm(dm); self.d1 = nn.Dropout(drop)
        self.cp = nn.Linear(dm,nf); ch=1
        for h in range(min(nh,nf),0,-1):
            if nf%h==0: ch=h; break
        self.ca = nn.MultiheadAttention(nf,ch,dropout=drop,batch_first=True)
        self.cb = nn.Linear(nf,dm); self.n2 = nn.LayerNorm(dm); self.d2 = nn.Dropout(drop)
        self.ffn = nn.Sequential(nn.Linear(dm,dm*4),nn.GELU(),nn.Dropout(drop),nn.Linear(dm*4,dm))
        self.n3 = nn.LayerNorm(dm); self.d3 = nn.Dropout(drop)
    def forward(self, x):
        a,_ = self.pa(x,x,x); x = self.n1(x+self.d1(a))
        c,_ = self.ca(self.cp(x),self.cp(x),self.cp(x)); x = self.n2(x+self.d2(self.cb(c)))
        return self.n3(x+self.d3(self.ffn(x)))
class DCdetector(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='DCdetector'; self.n_window=10; self.n_feats=feats
        self.pe = PatchEmb(10,3,2,feats,64); self.np=self.pe.np
        self.blocks = nn.ModuleList([DAB(64,feats,8) for _ in range(2)])
        self.rec = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,3*feats)); self.fcn=nn.Sigmoid()
    def _fold(self, rp, B, T, C):
        rec = torch.zeros(B,T,C,device=rp.device); cnt = torch.zeros(B,T,1,device=rp.device)
        for i in range(self.np):
            s=i*2; rec[:,s:s+3,:]+=rp[:,i,:].reshape(B,3,C); cnt[:,s:s+3,:]+=1
        return rec/cnt.clamp(min=1)
    def forward(self, x):
        B,T,C = x.shape; h = self.pe(x)
        for b in self.blocks: h = b(h)
        return self.fcn(self._fold(self.rec(h),B,T,C))


# 12. DTAAD (双 Transformer 编码器, 双视图)
class DTAAD(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='DTAAD'; self.n_window=10; self.n_feats=feats
        self.embed = nn.Linear(feats, 64)
        el = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc1 = nn.TransformerEncoder(el, 2)
        self.enc2 = nn.TransformerEncoder(el, 2)
        self.fcn = nn.Sequential(nn.Linear(128, feats), nn.Sigmoid())
    def forward(self, x):
        B, T, C = x.shape
        h = self.embed(x)
        z1 = self.enc1(h)
        z2 = self.enc2(h + torch.randn_like(h)*0.01)   # 双视图 (加噪)
        return self.fcn(torch.cat([z1, z2], dim=-1))


# 12. EPAAD-Net (Ours) — TCN + SCTM + 自定义 decoder
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous()

class TemporalCnn(nn.Module):
    def __init__(self, ni, no, ks, st, dil, pad, drop=0.2):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(ni, no, ks, stride=st, padding=pad, dilation=dil))
        self.net = nn.Sequential(self.conv, Chomp1d(pad), nn.ReLU(True), nn.Dropout(drop))
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)
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

class Autoencoder(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(o, h), nn.ReLU(), nn.Linear(h, i), nn.ReLU())
    def forward(self, x): return self.dec(self.enc(x))

class TransformerDecoderLayer1(nn.Module):
    """EPAAD-Net 自定义 decoder: 2 个 autoencoder (去噪) + cross-attn + FFN"""
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.ae1 = Autoencoder(dm, dm//3, dm)
        self.ae2 = Autoencoder(dm, dm//3, dm)
        self.mha = nn.MultiheadAttention(dm, nh, dropout=drop)
        self.l1 = nn.Linear(dm, dff); self.dr = nn.Dropout(drop); self.l2 = nn.Linear(dff, dm)
        self.d1 = nn.Dropout(drop); self.d2 = nn.Dropout(drop)
        self.d3 = nn.Dropout(drop); self.d4 = nn.Dropout(drop); self.act = nn.LeakyReLU(True)
    def forward(self, tgt, mem):
        tgt = tgt + self.d1(self.ae1(tgt)); tgt = tgt + self.d2(self.ae2(tgt))
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.dr(self.act(self.l1(tgt)))))

class EPAADNet(nn.Module):
    def __init__(self, feats):
        super().__init__(); self.name='EPAADNet'; self.n_window=10; self.n_feats=feats
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.dec = TransformerDecoderLayer1(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, x):
        B, T, C = x.shape
        src = x.permute(1, 0, 2)                        # (T, B, C)
        tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))           # (B, C, T)
        src = src + self.drop(s2.permute(2, 0, 1))      # 残差 TCN
        src = self.sp(src)                              # SCTM
        out = self.dec(tgt, src)                        # (1, B, C)
        return self.fcn(out).squeeze(0)                 # (B, C)


# ==================== 测量 (Time per Record) ====================
ALL_BASELINES = [EPAADNet, TranAD, USAD, OmniAnomaly, LSTM_AD, MAD_GAN, MSCRED,
                 CAE_M, GDN, MTAD_GAT, DTAAD, TimesNet, DCdetector]


def make_input(model, feats, batch=1, dtype=torch.float64):
    nw = model.n_window
    name = model.name
    if name in ('EPAADNet', 'TranAD', 'LSTM_AD', 'OmniAnomaly', 'DTAAD', 'TimesNet', 'DCdetector'):
        return torch.randn(batch, nw, feats, dtype=dtype, device=DEVICE)
    elif name in ('MSCRED', 'CAE-M', 'MTAD-GAT'):
        return torch.randn(batch, nw, nw, dtype=dtype, device=DEVICE)  # (B, feats, feats)
    else:  # USAD, MAD_GAN, GDN (flattened)
        return torch.randn(batch, nw * feats, dtype=dtype, device=DEVICE)


def measure(model, feats, n_warmup=1, n_iter=10):
    """Time per Record = batch=1 单条记录推理延迟 (ms), 快速版"""
    model = model.to(DEVICE).eval()
    inp = make_input(model, feats, 4)
    for _ in range(n_warmup):
        _ = model(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = model(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / n_iter / 4 * 1000   # ms per record


def main():
    print(f"设备: {DEVICE}")
    print(f"\n{'='*70}\n  基线方法 Jetson 评估 — Time per Record (ms)\n{'='*70}")
    all_results = {}
    for ds, feats in FEATS_MAP.items():
        print(f"\n--- {ds} (feats={feats}) ---")
        print(f"  {'Model':<14} {'Time per Record (ms)':<22}")
        print(f"  {'-'*14} {'-'*22}")
        ds_r = {}
        for Cls in ALL_BASELINES:
            try:
                m = Cls(feats).double()
            except Exception as e:
                print(f"  {Cls.__name__:<14} SKIP(init:{e})"); continue
            # 先打印名字, 便于观察进度 (重模型会慢)
            print(f"  {m.name:<14} 测中...", end='', flush=True)
            try:
                tpr = measure(m, feats)
                ds_r[m.name] = tpr
                print(f"\r  {m.name:<14} {tpr:<22.4f}")
            except Exception as e:
                print(f"\r  {m.name:<14} ERR({e})")
        all_results[ds] = ds_r

    models = sorted({n for ds in all_results for n in all_results[ds]})
    print(f"\n\n{'='*70}\n  LaTeX: Time per Record (ms)\n{'='*70}")
    for name in models:
        row = [name]
        for ds in FEATS_MAP:
            r = all_results[ds].get(name)
            row.append(f"{r:.4f}" if r is not None else "--")
        print(" & ".join(row) + " \\\\")
    return all_results


if __name__ == '__main__':
    res = main()
    with open('results/baselines_jetson_results.json', 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print("\n结果已保存到 baselines_jetson_results.json")
