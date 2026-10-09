"""
================================================================================
EPAAD-Net (EPAADNet) 消融变体: 参数量 & 推理时间对比
8 个变体对应论文 Table V
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, math, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATS_MAP = {'SMD': 38, 'SMAP': 25, 'MSL': 55, 'SWaT_MV': 6, 'MBA': 2, 'NAB': 1}

# ==================== 基础模块 ====================
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
    def forward(self, x):
        return 0.3*torch.cos(torch.matmul(x, self.rw)) + 0.7*torch.sin(torch.matmul(x, self.iw))

class SPL_SinOnly(nn.Module):
    """SCTM: 仅 Sine 变换"""
    def __init__(self, sd, hd):
        super().__init__()
        self.iw = nn.Parameter(torch.Tensor(sd, hd)); nn.init.xavier_uniform_(self.iw)
    def forward(self, x): return torch.sin(torch.matmul(x, self.iw))

class SPL_CosOnly(nn.Module):
    """SCTM: 仅 Cosine 变换"""
    def __init__(self, sd, hd):
        super().__init__()
        self.rw = nn.Parameter(torch.Tensor(sd, hd)); nn.init.xavier_uniform_(self.rw)
    def forward(self, x): return torch.cos(torch.matmul(x, self.rw))

class DecoderLayer_Full(nn.Module):
    """完整 decoder: DM(autoencoders) + cross-attn + FFN"""
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

class DecoderLayer_NoDM(nn.Module):
    """w/o DM: 无去噪自编码器, 仅 cross-attn + FFN"""
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dm, nh, dropout=drop)
        self.l1 = nn.Linear(dm, dff); self.dr = nn.Dropout(drop); self.l2 = nn.Linear(dff, dm)
        self.d3 = nn.Dropout(drop); self.d4 = nn.Dropout(drop); self.act = nn.LeakyReLU(True)
    def forward(self, tgt, mem):
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.dr(self.act(self.l1(tgt)))))

class DecoderLayer_SED2Att(nn.Module):
    """SED → Att: 用标准 TransformerDecoderLayer 替换"""
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model=dm, nhead=nh, dim_feedforward=dff*4,
                                                  dropout=drop, batch_first=False)
    def forward(self, tgt, mem):
        return self.layer(tgt, mem)


# ==================== 8 个变体 EPAADNet ====================
class EPAADNet_Full(nn.Module):
    """完整 EPAAD-Net"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'Full'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        # src is always (T, B, C)
        tgt = src[-1:, :, :]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_NoConv1d(nn.Module):
    """V1: w/o Conv1d — 移除 TCN 块"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o Conv1d'
        self.sp = SCTM(feats, feats)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_NoSCTM(nn.Module):
    """V2: w/o SCTM — 移除 Sine-Cosine 变换模块"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o SCTM'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.drop = nn.Dropout(0.1)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, src))

class EPAADNet_NoDM(nn.Module):
    """V3: w/o DM — 移除去噪模块 (autoencoders)"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o DM'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.dec = DecoderLayer_NoDM(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_NoSin(nn.Module):
    """V4: w/o Sin — SCTM 仅保留 Cosine"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o Sin'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SPL_CosOnly(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_NoCos(nn.Module):
    """V5: w/o Cos — SCTM 仅保留 Sine"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o Cos'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SPL_SinOnly(feats, feats)
        self.drop = nn.Dropout(0.1)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_SED2Att(nn.Module):
    """V6: SED → Att — 用标准 TransformerDecoderLayer 替换自编码器 decoder"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'SED->Att'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        # 标准 decoder layer (需要 nhead 整除 d_model)
        nhead = 1
        for h in range(min(8, feats), 0, -1):
            if feats % h == 0: nhead = h; break
        self.dec = DecoderLayer_SED2Att(feats, nhead, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        return self.fcn(self.dec(tgt, self.sp(src)))

class EPAADNet_NoTSADB(nn.Module):
    """V7: w/o TSADB — 移除时序自注意力解码块, 直接用 Linear"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o TSADB'
        self.l_tcn = Tcn_Local(feats, 4, 0.2)
        self.sp = SCTM(feats, feats)
        self.drop = nn.Dropout(0.1)
        # 用简单投影替换整个 decoder
        self.proj = nn.Sequential(nn.Linear(feats, 16), nn.ReLU(), nn.Linear(16, feats))
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        s2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.drop(s2.permute(2, 0, 1))
        mem = self.sp(src)
        # 用 memory 的最后一个时间步 + 投影
        return self.fcn(self.proj(mem[-1:]))

class EPAADNet_NoLTDMB(nn.Module):
    """V8: w/o LTDMB — 移除长期依赖建模块 (TCN), 只用 SCTM + Decoder"""
    def __init__(self, feats, nw=10):
        super().__init__(); self.n_feats = feats; self.n_window = nw; self.name = 'w/o LTDMB'
        self.sp = SCTM(feats, feats)
        self.dec = DecoderLayer_Full(feats, feats, 16, 0.1)
        self.fcn = nn.Sigmoid()
    def forward(self, src):
        if src.dim() == 3 and src.shape[1] != self.n_feats: src = src.permute(1, 0, 2)
        tgt = src[-1:, :, :] if src.dim() == 3 else src[-1:]
        return self.fcn(self.dec(tgt, self.sp(src)))


# ==================== 测量工具 ====================
VARIANTS = [EPAADNet_Full, EPAADNet_NoConv1d, EPAADNet_NoSCTM, EPAADNet_NoDM,
            EPAADNet_NoSin, EPAADNet_NoCos, EPAADNet_SED2Att, EPAADNet_NoTSADB, EPAADNet_NoLTDMB]


def measure(model, feats, n_window=10, batch_size=128, n_warmup=20, n_iter=100):
    """测量参数量 + 推理时间"""
    model.eval()
    model = model.to(DEVICE)
    dtype = next(model.parameters()).dtype

    # 参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

    # FLOPs 估算
    def estimate_flops(model, feats, nw):
        total = 0
        # TCN: 3层 * Conv1d(C→C, k=4) * T
        if hasattr(model, 'l_tcn'):
            total += 3 * (feats * feats * 4 * nw) * 2
        # SCTM: 2 * T*C*C
        if hasattr(model, 'sp') and isinstance(model.sp, SCTM):
            total += 2 * (nw * feats * feats) * 2 + 2 * nw * feats
        elif hasattr(model, 'sp') and (isinstance(model.sp, SPL_SinOnly) or isinstance(model.sp, SPL_CosOnly)):
            total += (nw * feats * feats) * 2 + nw * feats
        # Decoder: MHA + FFN
        if hasattr(model, 'dec'):
            total += (feats * feats * nw * 4) * 2  # cross-attn
            if hasattr(model.dec, 'ae1'):
                total += 2 * (feats * (feats//3) * 2 * nw) * 2
            total += (feats * 16 * nw * 2) * 2  # FFN
        elif hasattr(model, 'proj'):
            total += (feats * 16 * nw * 2) + (16 * feats * nw * 2)
        return total / 1e6

    flops = estimate_flops(model, feats, n_window)

    # 推理时间
    src = torch.randn(n_window, batch_size, feats, dtype=dtype, device=DEVICE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = model(src)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        _ = model(src)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    lat_ms = (t1 - t0) / n_iter * 1000
    lat_us_per_sample = lat_ms / batch_size * 1000

    return {
        'params': n_params,
        'param_mem_kb': round(param_mem, 2),
        'flops_m': round(flops, 2),
        'latency_ms': round(lat_ms, 4),
        'latency_us_per_sample': round(lat_us_per_sample, 2),
    }


def run_profile(feats_list, n_window=10, batch_size=128):
    """对所有变体进行测量"""
    print(f"\n{'='*120}")
    print(f"  EPAAD-Net 消融变体分析 (batch={batch_size}, seq_len={n_window})")
    print(f"{'='*120}")

    all_results = {}

    for ds_name, feats in feats_list:
        print(f"\n--- {ds_name} (feats={feats}) ---")
        print(f"  {'Variant':<16} {'Params':<10} {'Param(KB)':<10} {'FLOPs(M)':<10} "
              f"{'Lat(ms)':<10} {'Lat/samp(us)':<14}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

        ds_results = {}
        for variant_cls in VARIANTS:
            try:
                model = variant_cls(feats, nw=n_window)
                m = measure(model, feats, n_window, batch_size)
                ds_results[model.name] = m
                print(f"  {model.name:<16} {m['params']:<10,} {m['param_mem_kb']:<10.2f} "
                      f"{m['flops_m']:<10.2f} {m['latency_ms']:<10.4f} {m['latency_us_per_sample']:<14.2f}")
            except Exception as e:
                print(f"  {variant_cls.__name__:<16} FAILED: {e}")

        all_results[ds_name] = ds_results

    # ==================== LaTeX 表格 ====================
    print(f"\n\n{'='*120}")
    print(f"  LaTeX: 参数量对比表")
    print(f"{'='*120}")

    for ds_name, ds_results in all_results.items():
        full_params = ds_results['Full']['params']
        print(f"\n% --- {ds_name} ---")
        for name in ['Full', 'w/o Conv1d', 'w/o SCTM', 'w/o DM', 'w/o Sin', 'w/o Cos',
                      'SED->Att', 'w/o TSADB', 'w/o LTDMB']:
            if name in ds_results:
                r = ds_results[name]
                delta = (r['params'] - full_params) / full_params * 100
                print(f"{name} & {r['params']:,} & {r['flops_m']:.2f} & "
                      f"{r['latency_ms']:.4f} & {r['latency_us_per_sample']:.1f} & "
                      f"{delta:+.1f}\\% \\\\")

    # 汇总表 (所有数据集平均)
    print(f"\n\n{'='*120}")
    print(f"  LaTeX: 推理时间对比 (所有数据集平均)")
    print(f"{'='*120}")
    variant_names = ['Full', 'w/o Conv1d', 'w/o SCTM', 'w/o DM', 'w/o Sin', 'w/o Cos',
                     'SED->Att', 'w/o TSADB', 'w/o LTDMB']

    avg_params, avg_flops, avg_lat = {}, {}, {}
    for name in variant_names:
        vals = [all_results[ds][name] for ds in all_results if name in all_results[ds]]
        if vals:
            avg_params[name] = np.mean([v['params'] for v in vals])
            avg_flops[name] = np.mean([v['flops_m'] for v in vals])
            avg_lat[name] = np.mean([v['latency_us_per_sample'] for v in vals])

    base_p = avg_params.get('Full', 1)
    base_l = avg_lat.get('Full', 1)
    print(f"  {'Variant':<16} {'Avg Params':<12} {'ΔParams%':<10} {'Avg Lat(us)':<12} {'ΔLat%':<10}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for name in variant_names:
        if name in avg_params:
            dp = (avg_params[name] - base_p) / base_p * 100
            dl = (avg_lat[name] - base_l) / base_l * 100
            print(f"  {name:<16} {avg_params[name]:<12,.0f} {dp:<+10.1f}% "
                  f"{avg_lat[name]:<12.1f} {dl:<+10.1f}%")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str, default='results/variants_results.json')
    args = parser.parse_args()

    if args.dataset == 'all':
        feats_list = list(FEATS_MAP.items())
    else:
        feats_list = [(args.dataset, FEATS_MAP.get(args.dataset, 38))]

    results = run_profile(feats_list, args.window, args.batch_size)

    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    with open(args.output, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n结果已保存到 {args.output}")
