"""
================================================================================
EPAADNet 效率分析脚本
测量: FLOPs, 参数内存, 推理延迟 vs 批量大小, 吞吐量 vs 序列长度
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ==================== EPAADNet 模型 ====================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                     stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.relu1 = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.relu1, self.dropout)
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class Tcn_Local(nn.Module):
    def __init__(self, num_outputs, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.append(TemporalCnn(num_outputs, num_outputs, kernel_size,
                                      stride=1, dilation=1,
                                      padding=(kernel_size - 1), dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SCTM(nn.Module):
    def __init__(self, seq_dim, hidden_dim):
        super().__init__()
        self.dft_real_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim))
        self.dft_imag_weight = nn.Parameter(torch.Tensor(seq_dim, hidden_dim))
        nn.init.xavier_uniform_(self.dft_real_weight)
        nn.init.xavier_uniform_(self.dft_imag_weight)

    def forward(self, x):
        real_part = torch.cos(torch.matmul(x, self.dft_real_weight))
        imag_part = torch.sin(torch.matmul(x, self.dft_imag_weight))
        return 0.3 * real_part + 0.7 * imag_part


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        return self.dropout(x + self.pe[pos:pos + x.size(0), :])


class TransformerDecoderLayer1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.autoencoder1 = nn.Sequential(
            nn.Linear(d_model, d_model // 3), nn.ReLU(),
            nn.Linear(d_model // 3, d_model), nn.ReLU(),
        )
        self.autoencoder2 = nn.Sequential(
            nn.Linear(d_model, d_model // 3), nn.ReLU(),
            nn.Linear(d_model // 3, d_model), nn.ReLU(),
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory):
        tgt2 = self.autoencoder1(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.autoencoder2(tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return tgt


class EPAADNet(nn.Module):
    """EPAADNet: 你们提出的网络"""
    def __init__(self, feats, n_window=10):
        super().__init__()
        self.name = 'EPAADNet'
        self.n_feats = feats
        self.n_window = n_window

        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)
        self.sctm = SCTM(feats, feats)
        self.dropout1 = nn.Dropout(0.1)

        self.decoder_layer = TransformerDecoderLayer1(d_model=feats, nhead=feats,
                                                       dim_feedforward=16, dropout=0.1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt=None):
        # src: (T, B, C)   EPAADNet original format
        if tgt is None:
            tgt = src[-1:, :, :]  # last timestep
        src2 = self.l_tcn(src.permute(1, 2, 0))      # (B, C, T)
        src = src + self.dropout1(src2.permute(2, 0, 1))  # (T, B, C)
        src = self.sctm(src)                      # (T, B, C)
        # 手动 decoder (避免 nn.TransformerDecoder 对 self_attn 的检查)
        x = self.decoder_layer(tgt, src)               # (1, B, C)
        return self.fcn(x)


# ==================== 效率分析 ====================
def profile_epaad(feats_list=None, n_window=10):
    """
    对 EPAADNet 在不同特征维度下进行效率分析
    feats_list: 不同数据集的 feats 数 (SMD=38, SMAP=25, MBA=2, NAB=1, SWaT=6, MSL=55)
    """
    if feats_list is None:
        feats_list = [
            ('SMD', 38),
            ('SMAP', 25),
            ('MSL', 55),
            ('SWaT_MV', 6),
            ('MBA', 2),
            ('NAB', 1),
        ]

    all_results = {}

    for ds_name, feats in feats_list:
        print(f"\n{'='*80}")
        print(f"  EPAADNet on {ds_name} (feats={feats}, window={n_window})")
        print(f"{'='*80}")

        model = EPAADNet(feats, n_window=n_window).double().to(DEVICE)
        model.eval()
        dtype = next(model.parameters()).dtype

        # ---- 参数量 ----
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        print(f"  参数量:     {n_params:,}")
        print(f"  参数内存:   {param_mem:.4f} MB")

        # ---- FLOPs (精确手动计算, 每样本) ----
        # TCN: 3层 Conv1d(C→C, k=4), 乘加各算一次 = 2*C*C*4*T per layer
        flops_tcn = 3 * (feats * feats * 4 * n_window) * 2
        # SCTM: 2个 matmul(T,C)@(C,C) + cos+sin = 2*T*C*C*2 + 2*T*C
        flops_sctm = 2 * (n_window * feats * feats) * 2 + 2 * n_window * feats
        # DecoderLayer: cross-attn QKV+out = 4*C*C*T + 2xAutoencoder + FFN
        flops_attn = (feats * feats * n_window * 4) * 2
        flops_ae = 2 * (feats * (feats//3) * 2 * n_window) * 2
        flops_ffn = (feats * 16 * n_window * 2) * 2
        flops_m = (flops_tcn + flops_sctm + flops_attn + flops_ae + flops_ffn) / 1e6
        print(f"  FLOPs:      {flops_m:.2f} M  (per sample)")

        # ---- 推理延迟 vs 批量大小 ----
        print(f"\n  {'Batch':<8} {'Lat(ms)':<12} {'Thr(samp/s)':<16} {'Lat/samp(us)':<16} {'GPU Mem(MB)':<14}")
        print(f"  {'-'*8} {'-'*12} {'-'*16} {'-'*16} {'-'*14}")
        batch_latencies = []
        for bs in [1, 16, 32, 64, 128, 256]:
            src = torch.randn(n_window, bs, feats, dtype=dtype, device=DEVICE)
            tgt = torch.randn(1, bs, feats, dtype=dtype, device=DEVICE)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # Warmup
            for _ in range(10):
                _ = model(src, tgt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Measure
            n_iter = 50
            t0 = time.time()
            for _ in range(n_iter):
                _ = model(src, tgt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            lat_ms = (t1 - t0) / n_iter * 1000
            thr = bs / (lat_ms / 1000)
            lat_us = lat_ms / bs * 1000
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

            print(f"  {bs:<8} {lat_ms:<12.4f} {thr:<16.1f} {lat_us:<16.2f} {mem_mb:<14.2f}")
            batch_latencies.append({'batch_size': bs, 'latency_ms': lat_ms,
                                     'throughput': thr, 'latency_us_per_sample': lat_us})

        # ---- 吞吐量 vs 序列长度 ----
        print(f"\n  {'SeqLen':<8} {'Lat(ms)':<12} {'Thr(samp/s)':<16} {'Lat/samp(us)':<16} {'GPU Mem(MB)':<14}")
        print(f"  {'-'*8} {'-'*12} {'-'*16} {'-'*16} {'-'*14}")
        seq_latencies = []
        bs_fixed = 64
        for seq_len in [10, 20, 50, 100, 200]:
            try:
                tmp_model = EPAADNet(feats, n_window=seq_len).double().to(DEVICE)
                tmp_model.eval()
            except Exception:
                print(f"  {seq_len:<8} (OOM / not supported)")
                continue

            src = torch.randn(seq_len, bs_fixed, feats, dtype=dtype, device=DEVICE)
            tgt = torch.randn(1, bs_fixed, feats, dtype=dtype, device=DEVICE)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            for _ in range(5):
                _ = tmp_model(src, tgt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            n_iter = 30
            t0 = time.time()
            for _ in range(n_iter):
                _ = tmp_model(src, tgt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            lat_ms = (t1 - t0) / n_iter * 1000
            thr = bs_fixed / (lat_ms / 1000)
            lat_us = lat_ms / bs_fixed * 1000
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

            print(f"  {seq_len:<8} {lat_ms:<12.4f} {thr:<16.1f} {lat_us:<16.2f} {mem_mb:<14.2f}")
            seq_latencies.append({'seq_len': seq_len, 'latency_ms': lat_ms,
                                   'throughput': thr, 'latency_us_per_sample': lat_us})

        # ---- 汇总 ----
        src = torch.randn(n_window, 128, feats, dtype=dtype, device=DEVICE)
        tgt = torch.randn(1, 128, feats, dtype=dtype, device=DEVICE)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        for _ in range(10):
            _ = model(src, tgt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        n_iter = 100
        t0 = time.time()
        for _ in range(n_iter):
            _ = model(src, tgt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        base_lat_ms = (t1 - t0) / n_iter * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

        print(f"\n  --- {ds_name} 汇总 (batch=128, seq_len={n_window}) ---")
        print(f"  推理延迟:        {base_lat_ms:.4f} ms")
        print(f"  每样本延迟:      {base_lat_ms/128*1000:.2f} us")
        print(f"  吞吐量:          {128/(base_lat_ms/1000):.1f} samples/s")
        print(f"  推理峰值显存:    {peak_mem:.2f} MB")
        print(f"  参数内存:        {param_mem:.4f} MB")
        print(f"  参数量:          {n_params:,}")

        all_results[ds_name] = {
            'feats': feats,
            'n_window': n_window,
            'params': n_params,
            'param_mem_mb': round(param_mem, 4),
            'flops_m': round(flops_m, 2),
            'latency_ms_batch128': round(base_lat_ms, 4),
            'latency_us_per_sample': round(base_lat_ms / 128 * 1000, 2),
            'throughput_samp_per_s': round(128 / (base_lat_ms / 1000), 1),
            'peak_gpu_mem_mb': round(peak_mem, 2),
            'batch_latencies': batch_latencies,
            'seq_latencies': seq_latencies,
        }

    # ==================== 汇总表格 ====================
    print(f"\n\n{'='*100}")
    print(f"  EPAADNet 效率汇总 (batch=128, seq_len={n_window})")
    print(f"{'='*100}")
    print(f"  {'Dataset':<12} {'Feats':<6} {'Params':<10} {'ParamMem':<10} {'FLOPs(M)':<10} "
          f"{'Lat(ms)':<10} {'Lat/samp(us)':<14} {'Thr(samp/s)':<14} {'GPU Mem(MB)':<12}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*14} {'-'*12}")
    for ds_name, r in all_results.items():
        print(f"  {ds_name:<12} {r['feats']:<6} {r['params']:<10,} {r['param_mem_mb']:<10.4f} "
              f"{r['flops_m']:<10.2f} {r['latency_ms_batch128']:<10.4f} "
              f"{r['latency_us_per_sample']:<14.2f} {r['throughput_samp_per_s']:<14.1f} "
              f"{r['peak_gpu_mem_mb']:<12.2f}")

    # ---- LaTeX 表格 ----
    print(f"\n--- LaTeX: 参数量 & 效率表 ---")
    for ds_name, r in all_results.items():
        print(f"EPAADNet-{ds_name} & {r['params']:,} & {r['param_mem_mb']:.2f} MB & "
              f"{r['flops_m']:.1f}M & {r['latency_ms_batch128']:.2f} ms & "
              f"{r['latency_us_per_sample']:.1f} us & {r['throughput_samp_per_s']:.0f} samp/s \\\\")

    # 保存
    out_path = os.path.join(ROOT, 'results/profile_epaad_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {out_path}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EPAADNet Efficiency Profiling')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset to profile (SMD, SMAP, MSL, etc.) or "all"')
    parser.add_argument('--window', type=int, default=10,
                        help='Sliding window size (default: 10)')
    parser.add_argument('--output', type=str, default='results/profile_epaad_results.json')
    args = parser.parse_args()

    if args.dataset == 'all':
        feats_list = [
            ('SMD', 38), ('SMAP', 25), ('MSL', 55),
            ('SWaT_MV', 6), ('MBA', 2), ('NAB', 1),
        ]
    else:
        # 从数据集映射获取 feats
        FEATS_MAP = {'SMD': 38, 'SMAP': 25, 'MSL': 55, 'SWaT_MV': 6, 'MBA': 2, 'NAB': 1, 'SWaT_UV': 1}
        feats = FEATS_MAP.get(args.dataset, 38)
        feats_list = [(args.dataset, feats)]

    profile_epaad(feats_list, n_window=args.window)
