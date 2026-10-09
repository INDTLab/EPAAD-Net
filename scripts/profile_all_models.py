"""
================================================================================
所有模型效率对比: 参数量 + 推理延迟 (同硬件公平对比)
针对评审意见: 边缘部署需要所有轻量方法在同一平台测延迟

模型: EPAAD-Net, TranAD, USAD, OmniAnomaly, MAD-GAN, MSCRED,
      CAE-M, GDN, MTAD-GAT, DTAAD, TimesNet, DCdetector
================================================================================
用法: py profile_all_models.py --dataset SMD
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, json, argparse, warnings
import numpy as np
import torch, torch.nn as nn

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从 run_all_models 导入模型定义
# from run_all_models import ALL_MODELS, DATASETS
from benchmark_jetson import ALL_BASELINES

FEATS_MAP = {'SMD': 38, 'NAB': 1, 'MBA': 2, 'SMAP': 25, 'SWaT_MV': 6, 'SWaT_UV': 1}


def measure_model(model, feats, n_window=10, batch_sizes=(1, 16, 128),
                  n_warmup=20, n_iter=100):
    """测量模型的参数量 + 推理延迟"""
    model = model.to(DEVICE)
    model.eval()
    dtype = next(model.parameters()).dtype

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024  # KB

    # 延迟测量
    latencies = {}
    for bs in batch_sizes:
        src = torch.randn(n_window, bs, feats, dtype=dtype, device=DEVICE)
        # 预热
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
        latencies[bs] = {
            'latency_ms': round(lat_ms, 4),
            'latency_us_per_sample': round(lat_ms / bs * 1000, 2),
            'throughput_samp_per_s': round(bs / (lat_ms / 1000), 1),
        }

    return {
        'params': n_params,
        'param_mem_kb': round(param_mem, 2),
        'latencies': latencies,
    }


def run(epochs_arg=0):
    print(f"设备: {DEVICE}")
    print(f"\n{'='*110}")
    print(f"  所有模型效率对比 (batch 延迟 vs 每样本延迟)")
    print(f"{'='*110}")

    all_results = {}
    for ds_name, feats in FEATS_MAP.items():
        print(f"\n--- {ds_name} (feats={feats}) ---")
        print(f"  {'Model':<16} {'Params':<10} {'Mem(KB)':<10} "
              f"{'Lat@1(ms)':<12} {'Lat@128(ms)':<14} {'us/samp@128':<14} {'Thr(samp/s)':<14}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")

        ds_results = {}
        for Cls in ALL_BASELINES:
            try:
                m = Cls(feats)
            except Exception as e:
                print(f"  {Cls.__name__:<16} SKIP (init fail: {e})")
                continue
            name = getattr(m, 'name', Cls.__name__)
            try:
                r = measure_model(m, feats)
                ds_results[name] = r
                lat1 = r['latencies'][1]['latency_ms']
                lat128 = r['latencies'][128]['latency_ms']
                us = r['latencies'][128]['latency_us_per_sample']
                thr = r['latencies'][128]['throughput_samp_per_s']
                print(f"  {name:<16} {r['params']:<10,} {r['param_mem_kb']:<10.2f} "
                      f"{lat1:<12.4f} {lat128:<14.4f} {us:<14.2f} {thr:<14.1f}")
            except Exception as e:
                print(f"  {name:<16} ERR: {e}")

        all_results[ds_name] = ds_results

    # ==================== LaTeX 汇总 ====================
    # 按数据集汇总参数 + 延迟
    print(f"\n\n{'='*110}")
    print("  LaTeX: 参数量对比")
    print(f"{'='*110}")
    models = sorted({name for ds in all_results.values() for name in ds})
    for name in models:
        row = [f"{name}"]
        for ds_name in FEATS_MAP:
            r = all_results[ds_name].get(name)
            row.append(f"{r['params']:,}" if r else "--")
        print(" & ".join(row) + " \\\\")

    print(f"\n\nLaTeX: 每样本推理延迟 (us, batch=128)")
    for name in models:
        row = [f"{name}"]
        for ds_name in FEATS_MAP:
            r = all_results[ds_name].get(name)
            row.append(f"{r['latencies'][128]['latency_us_per_sample']:.1f}" if r else "--")
        print(" & ".join(row) + " \\\\")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='results/all_models_efficiency.json')
    args = parser.parse_args()
    res = run()

    def convert(o):
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    with open(args.output, 'w') as f:
        json.dump(convert(res), f, indent=2)
    print(f"\n结果已保存到 {args.output}")
