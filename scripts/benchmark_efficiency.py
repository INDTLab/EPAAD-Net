"""
================================================================================
基线方法 Jetson 效率评估 (从 src/models.py 导入真实架构)
测量: 参数量 + 推理延迟

9 个基线: TranAD, USAD, OmniAnomaly, LSTM_AD, MAD_GAN, MSCRED, CAE_M,
          GDN, MTAD_GAT (另加 TimesNet, DCdetector)
================================================================================
用法: python3 benchmark_efficiency.py
依赖: torch + dgl (GDN/MTAD_GAT 需要 dgl)
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, json, warnings
import torch, torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 保护 src.parser 的 parse_args (它会在 import 时解析命令行)
_orig_argv = sys.argv
sys.argv = ['benchmark_efficiency.py']

try:
    from src.models import (TranAD, USAD, OmniAnomaly, LSTM_AD, MAD_GAN,
                            MSCRED, CAE_M, GDN, MTAD_GAT,
                            EPAADNet, TimesNet, DCdetector)
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print("请先安装 dgl: pip3 install dgl")
    sys.exit(1)
finally:
    sys.argv = _orig_argv

FEATS_MAP = {'SMD': 38, 'NAB': 1, 'MBA': 2, 'SMAP': 25, 'SWaT_MV': 6, 'SWaT_UV': 1}

# 每个模型的前向输入格式
# 'flatten': (B, n_window*n_feats)  — USAD/MAD_GAN/GDN/MSCRED/CAE_M
# 'tranad': (src, tgt) 各 (T, B, feats)
# 'lstm_ad': (n_window, feats) 串行窗口
# 'omni': (B, feats) 单时间步
# 'mtad_gat': (data, hidden)
# '3d': (B, T, C) — TimesNet/DCdetector
def get_input_kind(model):
    name = model.name
    if 'TranAD' in name:
        return 'tranad'
    if 'EPAADNet' in name:
        return 'epaad'
    if 'LSTM_AD' in name:
        return 'lstm_ad'
    if 'OmniAnomaly' in name:
        return 'omni'
    if 'MTAD_GAT' in name:
        return 'mtad_gat'
    if name in ('TimesNet', 'DCdetector'):
        return '3d'
    return 'flatten'


def make_input(model, feats, batch=1, dtype=torch.float64):
    nw = getattr(model, 'n_window', 10)
    kind = get_input_kind(model)
    if kind == 'tranad':
        src = torch.randn(nw, batch, feats, dtype=dtype, device=DEVICE)
        tgt = torch.randn(nw, batch, feats, dtype=dtype, device=DEVICE)
        return (src, tgt)
    elif kind == 'epaad':
        src = torch.randn(nw, batch, feats, dtype=dtype, device=DEVICE)   # (T, B, C)
        tgt = torch.randn(1, batch, feats, dtype=dtype, device=DEVICE)    # (1, B, C)
        return (src, tgt)
    elif kind == 'lstm_ad':
        return torch.randn(nw, feats, dtype=dtype, device=DEVICE)      # 串行窗口
    elif kind == 'omni':
        return torch.randn(batch, feats, dtype=dtype, device=DEVICE)   # 单时间步
    elif kind == 'mtad_gat':
        data = torch.randn(batch, nw * feats, dtype=dtype, device=DEVICE)
        hidden = torch.rand(1, 1, feats * feats, dtype=dtype, device=DEVICE)
        return (data, hidden)
    elif kind == '3d':
        return torch.randn(batch, nw, feats, dtype=dtype, device=DEVICE)  # (B,T,C)
    else:  # flatten
        return torch.randn(batch, nw * feats, dtype=dtype, device=DEVICE)


def measure(model, feats, n_warmup=10, n_iter=50):
    """Time per Record = batch=1 单条记录推理延迟 (ms)"""
    model = model.to(DEVICE).eval()
    inp = make_input(model, feats, 1)
    for _ in range(n_warmup):
        if isinstance(inp, tuple):
            _ = model(*inp)
        else:
            _ = model(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        if isinstance(inp, tuple):
            _ = model(*inp)
        else:
            _ = model(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / n_iter * 1000   # ms per record


def main():
    print(f"设备: {DEVICE}")
    print(f"\n{'='*70}")
    print(f"  基线方法效率评估 — Time per Record (ms)")
    print(f"{'='*70}")

    models_cls = [EPAADNet, TranAD, USAD, OmniAnomaly, LSTM_AD, MAD_GAN, MSCRED,
                  CAE_M, GDN, MTAD_GAT, TimesNet, DCdetector]

    all_results = {}
    for ds, feats in FEATS_MAP.items():
        print(f"\n--- {ds} (feats={feats}) ---")
        print(f"  {'Model':<14} {'Time per Record (ms)':<22}")
        print(f"  {'-'*14} {'-'*22}")
        ds_r = {}
        for Cls in models_cls:
            try:
                m = Cls(feats)
            except Exception as e:
                print(f"  {Cls.__name__:<14} SKIP(init:{e})"); continue
            name = m.name
            print(f"  {name:<14} 测中...", end='', flush=True)
            try:
                m = m.double()  # 与原 load_model 一致
                tpr = measure(m, feats)
                ds_r[name] = tpr
                print(f"\r  {name:<14} {tpr:<22.4f}")
            except Exception as e:
                print(f"\r  {name:<14} ERR({e})")
        all_results[ds] = ds_r

    # LaTeX
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
    with open('results/baselines_import_results.json', 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print("\n结果已保存到 baselines_import_results.json")
