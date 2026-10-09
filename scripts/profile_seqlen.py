"""
================================================================================
EPAAD-Net: 不同序列长度下推理延迟实验
对比 Full 模型及各消融变体在 seq_len = [10,20,50,100,200,500] 下的延迟
================================================================================
用法:
    py profile_seqlen.py              # 所有数据集
    py profile_seqlen.py --dataset SMD
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, json, argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATS_MAP = {'SMD': 38, 'SMAP': 25, 'MSL': 55, 'SWaT_MV': 6, 'MBA': 2, 'NAB': 1}

# ==================== 基础模块 (同 EPAADNet) ====================
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

# Decoder variants
class DecFull(nn.Module):
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

class DecNoDM(nn.Module):
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dm, nh, dropout=drop)
        self.l1 = nn.Linear(dm, dff); self.dr = nn.Dropout(drop); self.l2 = nn.Linear(dff, dm)
        self.d3 = nn.Dropout(drop); self.d4 = nn.Dropout(drop); self.act = nn.LeakyReLU(True)
    def forward(self, tgt, mem):
        tgt = tgt + self.d3(self.mha(tgt, mem, mem)[0])
        return tgt + self.d4(self.l2(self.dr(self.act(self.l1(tgt)))))

class DecSED2Att(nn.Module):
    def __init__(self, dm, nh, dff=16, drop=0):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model=dm, nhead=nh, dim_feedforward=dff*4,
                                                  dropout=drop, batch_first=False)
    def forward(self, tgt, mem): return self.layer(tgt, mem)

# ==================== 变体模型 ====================
def make_epaad(feats, nw, variant='Full'):
    """构建指定变体"""

    class _EPAADNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_feats = feats; self.n_window = nw
            has_tcn = variant not in ('w/o Conv1d', 'w/o LTDMB')
            has_sctm = variant not in ('w/o SCTM',)
            has_dm = variant not in ('w/o DM',)
            use_std_attn = variant == 'SED->Att'
            has_dec = variant not in ('w/o TSADB',)

            if has_tcn:
                self.l_tcn = Tcn_Local(feats, 4, 0.2)
            self.drop = nn.Dropout(0.1)

            if has_sctm:
                if variant == 'w/o Sin':
                    self.sp = SCTM(feats, feats)
                    # overwrite to only use cos
                    self.sp.forward = lambda x: torch.cos(torch.matmul(x, self.sp.rw))
                elif variant == 'w/o Cos':
                    self.sp = SCTM(feats, feats)
                    self.sp.forward = lambda x: torch.sin(torch.matmul(x, self.sp.iw))
                else:
                    self.sp = SCTM(feats, feats)

            if has_dec:
                nhead = 1
                for h in range(min(8, feats), 0, -1):
                    if feats % h == 0: nhead = h; break
                if use_std_attn:
                    self.dec = DecSED2Att(feats, nhead, 16, 0.1)
                elif has_dm:
                    self.dec = DecFull(feats, feats, 16, 0.1)
                else:
                    self.dec = DecNoDM(feats, feats, 16, 0.1)
            else:
                self.proj = nn.Sequential(nn.Linear(feats, 16), nn.ReLU(), nn.Linear(16, feats))

            self.fcn = nn.Sigmoid()
            self._variant = variant
            self._has_tcn = has_tcn; self._has_sctm = has_sctm
            self._has_dec = has_dec; self._has_dm = has_dm

        def forward(self, src):
            if src.shape[0] != self.n_window:
                # reshape: assume (B, T, C)
                src = src.permute(1, 0, 2)
            tgt = src[-1:, :, :]
            if self._has_tcn:
                s2 = self.l_tcn(src.permute(1, 2, 0))
                src = src + self.drop(s2.permute(2, 0, 1))
            mem = self.sp(src) if self._has_sctm else src
            if self._has_dec:
                return self.fcn(self.dec(tgt, mem))
            else:
                return self.fcn(self.proj(mem[-1:]))

    return _EPAADNet()


# ==================== 延迟测量 ====================
VARIANTS = ['Full', 'w/o Conv1d', 'w/o SCTM', 'w/o DM', 'w/o Sin', 'w/o Cos',
            'SED->Att', 'w/o TSADB', 'w/o LTDMB']

SEQ_LENGTHS = [10, 20, 50, 100, 200, 500]


def measure_latency(model, nw, feats, batch_size=128, n_warmup=15, n_iter=50):
    """测量给定序列长度下的推理延迟"""
    model.eval()
    model = model.to(DEVICE)
    dtype = next(model.parameters()).dtype

    src = torch.randn(nw, batch_size, feats, dtype=dtype, device=DEVICE)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    return lat_ms, lat_us_per_sample, mem_mb


def run_seqlen_benchmark(feats_list, batch_size=128):
    """不同序列长度延迟测试"""
    print(f"\n{'='*120}")
    print(f"  EPAAD-Net 序列长度-延迟分析 (batch={batch_size})")
    print(f"{'='*120}")

    all_results = {}

    for ds_name, feats in feats_list:
        print(f"\n{'='*80}")
        print(f"  {ds_name} (feats={feats})")
        print(f"{'='*80}")

        # 表头
        header = f"  {'Variant':<16}"
        for sl in SEQ_LENGTHS:
            header += f" {'T='+str(sl):<14}"
        print(header)
        print(f"  {'-'*16}" + f"{'':->{14*len(SEQ_LENGTHS)}}")

        ds_results = {}
        for variant in VARIANTS:
            row = f"  {variant:<16}"
            latencies = {}
            for nw in SEQ_LENGTHS:
                try:
                    model = make_epaad(feats, nw, variant)
                    lat_ms, lat_us, mem = measure_latency(model, nw, feats, batch_size)
                    row += f" {lat_ms:<14.4f}"
                    latencies[str(nw)] = {'latency_ms': round(lat_ms, 4),
                                          'latency_us_per_sample': round(lat_us, 2),
                                          'gpu_mem_mb': round(mem, 2)}
                except Exception as e:
                    row += f" {'ERR':<14}"
                    latencies[str(nw)] = None
            print(row)
            ds_results[variant] = latencies
        all_results[ds_name] = ds_results

    # ==================== LaTeX 表 ====================
    print(f"\n\n{'='*120}")
    print(f"  LaTeX: 序列长度 vs 延迟 (每样本 us)")
    print(f"{'='*120}")

    for ds_name, ds_results in all_results.items():
        print(f"\n% --- {ds_name} ---")
        full_lats = {sl: ds_results['Full'][str(sl)]['latency_us_per_sample']
                     if ds_results['Full'][str(sl)] else 0
                     for sl in SEQ_LENGTHS}
        for variant in VARIANTS:
            vals = []
            for sl in SEQ_LENGTHS:
                r = ds_results[variant].get(str(sl))
                if r:
                    vals.append(f"{r['latency_us_per_sample']:.1f}")
                else:
                    vals.append("--")
            print(f"{variant} & " + " & ".join(vals) + " \\\\")

    # 增长率表 (T=10 → T=500)
    print(f"\n\n--- LaTeX: 延迟增长率 (T=10→T=500) ---")
    for ds_name, ds_results in all_results.items():
        print(f"\n% --- {ds_name} ---")
        for variant in VARIANTS:
            r10 = ds_results[variant].get('10')
            r500 = ds_results[variant].get('500')
            if r10 and r500:
                growth = (r500['latency_us_per_sample'] - r10['latency_us_per_sample']) / r10['latency_us_per_sample'] * 100
                print(f"{variant} & {r10['latency_us_per_sample']:.1f} & "
                      f"{r500['latency_us_per_sample']:.1f} & {growth:+.1f}\\% \\\\")

    # 汇总: 所有数据集平均
    print(f"\n\n--- LaTeX: 所有数据集平均延迟 (us/sample) ---")
    header = "Method & " + " & ".join([f"T={sl}" for sl in SEQ_LENGTHS]) + " \\\\"
    print(header)
    for variant in VARIANTS:
        row = f"{variant}"
        for sl in SEQ_LENGTHS:
            vals = []
            for ds_name in all_results:
                r = all_results[ds_name][variant].get(str(sl))
                if r: vals.append(r['latency_us_per_sample'])
            if vals:
                avg = np.mean(vals)
                row += f" & {avg:.1f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str, default='results/seqlen_latency_results.json')
    args = parser.parse_args()

    if args.dataset == 'all':
        feats_list = list(FEATS_MAP.items())
    else:
        feats_list = [(args.dataset, FEATS_MAP.get(args.dataset, 38))]

    results = run_seqlen_benchmark(feats_list, args.batch_size)

    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    with open(args.output, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n结果已保存到 {args.output}")
