# EPAAD-Net

**Efficient Periodicity-Aware Anomaly Detection Network** for multivariate time-series anomaly detection.

EPAAD-Net is a lightweight reconstruction-based detector built around a *learnable periodic
activation*. It combines a local temporal convolutional backbone with an explicit
sine–cosine transform module and a dual-autoencoder attention decoder, and is evaluated on
standard multivariate time-series anomaly-detection benchmarks under both Point-Adjust (PA)
and point-wise (PW) protocols.

---

## Architecture

The detector (`src/models.py::EPAADNet`) is a stack of four modules:

| Module | File | Role |
|---|---|---|
| **LTDMB** — Long-Term Dependency Modeling Block | `src/tcn.py` | Three weight-normalized causal `Conv1d` residual blocks (`Tcn_Local`) that build multi-scale temporal context from the input window. |
| **SCTM** — Sine-Cosine Transform Module | `src/sctm.py` | A learnable periodic activation: `0.3·cos(x·W_r) + 0.7·sin(x·W_i)`. The real/imaginary weights are Xavier-initialised and trained end-to-end. This is the core periodicity-aware component. |
| **TSADB** — Time-Series Attention Decoder Block | `src/dlutils.py::TransformerDecoderLayer1` | Two denoising autoencoders (**DM**) + multi-head cross-attention + feed-forward network. |
| **Head** | `src/models.py` | `Sigmoid` reconstruction of the target window. |

The default window length is `n_window = 10` with batch size `128`; the learning rate is
selected per dataset from `src/constants.py`.

---

## Installation

Requires Python 3.8+ and PyTorch.

```bash
pip install -r requirements.txt
```

> `dgl` is only needed for the GDN and MTAD-GAT baselines. All other models — including
> EPAAD-Net — run without it, and `src/models.py` degrades gracefully when `dgl` is absent.

---

## Datasets

Raw datasets are **not** distributed with this repository (most are governed by their own
licenses). Download each dataset into `data/<NAME>/` and then preprocess it:

| Dataset | Place under |
|---|---|
| `synthetic` | `data/synthetic/` |
| `SMD` | `data/SMD/` |
| `SWaT`, `WADI` | `data/SWaT/`, `data/WADI/` |
| `SMAP`, `MSL` | `data/SMAP_MSL/` |
| `MSDS`, `UCR`, `MBA`, `NAB` | `data/<NAME>/` |

Preprocessing writes normalized `train/test/labels` `.npy` splits into `processed/<NAME>/`:

```bash
python preprocess.py SMD
python preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```

Datasets you have not downloaded can simply be omitted from the command.

---

## Usage

Train and evaluate EPAAD-Net on a dataset (run from the repository root):

```bash
python main.py --model EPAADNet --dataset SMD --retrain
```

Evaluate a trained checkpoint without retraining:

```bash
python main.py --model EPAADNet --dataset SMD --test
```

Train on 20% of the data:

```bash
python main.py --model EPAADNet --dataset SMD --retrain --less
```

Multi-scale window variant (`n_window` = 10/20/30):

```bash
python main_multiscale.py --model CNNLSTM_RH --dataset SMD --retrain
```

Outputs report **F1(PA)** (Point-Adjust, following the field convention) alongside
**F1(PW)** (point-wise, unadjusted), plus precision/recall, `Hit@100%`, `Hit@150%` and NDCG.
Checkpoints are written to `checkpoints/<model>_<dataset>/model.ckpt`.

### Available models

`EPAADNet` (this work), and the following baselines implemented in `src/models.py`:
`TranAD`, `USAD`, `OmniAnomaly`, `LSTM_AD`, `MAD_GAN`, `MSCRED`, `CAE_M`, `DAGMM`,
`Attention`, `GDN`, `MTAD_GAT`, `TimesNet`, `DCdetector`.

> Baselines are provided for internal comparison only. To reproduce the numbers of any
> published baseline, please use that method's original codebase.

---

## Reproducing the experiments

All experiment entry points live in `scripts/`. Every script must be run **from the
repository root** so that `src/` is importable.

| Script | Produces |
|---|---|
| `scripts/run_baselines.py` | Point-wise F1 comparison of EPAAD-Net vs. 11 baselines across datasets |
| `scripts/benchmark_efficiency.py` | Time-per-record efficiency benchmark (imports real classes from `src.models`) |
| `scripts/benchmark_jetson.py` | Efficiency benchmark with dgl-free GDN / MTAD-GAT reimplementations |
| `scripts/benchmark_pointwise.py` | Point-wise (non-adjusted) F1 for EPAAD-Net / TimesNet / DCdetector |
| `scripts/eval_cross_dataset.py` | Cross-dataset and cross-machine generalization |
| `scripts/eval_robustness.py` | Distribution shift, anomaly ratio, Gaussian noise (SNR), missing data |
| `scripts/eval_sensitivity.py` | Hyperparameter sensitivity (window / LR / batch / dropout) |
| `scripts/eval_significance.py` | Multi-seed mean±std, 95% CI, paired t-test, LaTeX table |
| `scripts/profile_ablation.py` | Ablation variants: params + inference time |
| `scripts/profile_epaad.py` | FLOPs, params, memory, latency vs. batch, throughput vs. seq length |
| `scripts/profile_seqlen.py` | Latency vs. sequence length (10–500) across variants |
| `scripts/profile_all_models.py` | Params + latency for every model |
| `scripts/run_timesnet.py`, `scripts/run_dcdetector.py` | Standalone baseline runs |
| `scripts/experiment_mae.py`, `scripts/experiment_mae_cnnlstm.py` | Masked-autoencoder explorations |

Figures and diagrams:

| Script | Produces |
|---|---|
| `scripts/plot_feature_flow.py` | Feature-flow diagram of LTDMB → SCTM → TSADB |
| `scripts/plot_ablation_bar.py` | Grouped bar chart of ablation F1 |
| `scripts/plot_bar_xr.py` | Grouped bar chart read from `XR.xlsx` |
| `scripts/stitch_vertical.py`, `scripts/stitch_horizontal.py` | Qualitative input/output panel stitching |

### Ablation variants

Defined in `scripts/profile_ablation.py`: `Full`, `w/o LTDMB`, `w/o SCTM`, `w/o DM`,
`w/o Sin`, `w/o Cos`, `SED→Att`, `w/o TSADB`.

> `w/o CONV1D` and `w/o LTDMB` are the same ablation under two names; prefer `w/o LTDMB`.

---

## Repository layout

```
EPAAD-Net/
├── main.py                  # train / evaluate entry point
├── main_multiscale.py       # multi-window (10/20/30) variant
├── preprocess.py            # dataset → processed/*.npy
├── requirements.txt
├── src/                     # library code
│   ├── models.py            # EPAADNet + all baselines
│   ├── sctm.py              # SCTM — sine-cosine transform module
│   ├── tcn.py               # LTDMB — temporal convolutional backbone
│   ├── dlutils.py           # TSADB decoder layer, autoencoders, positional encoding
│   ├── constants.py         # per-dataset thresholds / LR / percentiles
│   ├── parser.py            # argparse
│   ├── pot.py, spot.py      # POT / SPOT thresholding
│   ├── diagnosis.py         # Hit@k, NDCG
│   ├── merlin.py            # MERLIN parameter-free detector
│   ├── plotting.py, utils.py
│   └── folderconstants.py
├── scripts/                 # experiment drivers (run from repo root)
├── docs/figures/            # figures used in the paper / README
├── results/                 # generated result artifacts (git-ignored)
├── data/                    # raw datasets (git-ignored)
├── processed/               # preprocessed splits (git-ignored)
└── checkpoints/             # trained weights (git-ignored)
```

---

## Acknowledgements

Parts of the training and evaluation harness are adapted from TranAD
(https://github.com/imperial-qore/TranAD), which is distributed under the
BSD 3-Clause License. The core EPAAD-Net architecture (SCTM, LTDMB, TSADB)
is an independent contribution. See `LICENSE` for full terms.


---

## Citation

If you use EPAAD-Net in your research, please cite:

```bibtex
@article{li2026fast,
  title={Fast time series anomaly detection using an efficient periodicity-aware neural network},
  author={Li, Pengfei and Ruan, Yinghao and Liu, Peishun and Dong, Junyu and Dong, Xinghui},
  journal={Neurocomputing},
  pages={135312},
  year={2026},
  publisher={Elsevier}
}
```
---

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for details.
Some training/evaluation utilities are adapted from TranAD
(https://github.com/imperial-qore/TranAD), also under BSD 3-Clause.

