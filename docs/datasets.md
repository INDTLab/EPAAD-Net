# Datasets

The raw datasets are **not** redistributed with this repository. Most are covered by
third-party licenses or explicit "request only" terms, so each one must be obtained from
its original source and placed under `data/<NAME>/` before running `preprocess.py`.

This file records the provenance of each dataset so that the licensing position of the
repository stays clear.

| Dataset | Directory | Source | Redistributable? |
|---|---|---|---|
| SMD | `data/SMD/` | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | MIT (Copyright (c) 2021 NetManAIOps-SMD) |
| SMAP, MSL | `data/SMAP_MSL/` | [telemanom](https://s3-us-west-2.amazonaws.com/telemanom/data.zip); labels from [`labeled_anomalies.csv`](https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv) | See upstream |
| SWaT | `data/SWaT/` | `series.json` from [JulienAu/Anomaly_Detection_Tuto](https://raw.githubusercontent.com/JulienAu/Anomaly_Detection_Tuto/master/Data/serie2.json) | See upstream |
| MBA | `data/MBA/` | [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via [PhysioBank ATM](https://archive.physionet.org/cgi-bin/atm/ATM) | PhysioNet terms |
| NAB | `data/NAB/` | [numenta/NAB](https://github.com/numenta/NAB) — `realKnownCause` subset | See upstream |
| UCR | `data/UCR/` | [KDD 2021 Multi-dataset Time-Series Anomaly Detection Competition](https://compete.hexagon-ml.com/practice/competition/39/) | Competition terms. Only the natural-source subsets (InternalBleeding, ECG) are used; synthetic sequences are excluded. |
| synthetic | `data/synthetic/` | [7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED) | See upstream |
| MSDS | `data/MSDS/` | [Zenodo record 3549604](https://zenodo.org/record/3549604) | **No** — distribution rights belong to the dataset authors; request access and place the CSVs in `data/MSDS/metrics/`, then run `data/MSDS/clean.py` to produce `train.csv` / `test.csv`. |
| WADI | `data/WADI/` | [iTRUST](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi) | **No** — distribution rights belong to iTRUST; request access. |
| FTSD | `data/FTSD/` | Refer to `data/FTSD/readme.md` for the Azure deployment procedure | Internal |

## References

- Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., Pei, D. *Robust anomaly detection for
  multivariate time series through stochastic recurrent neural network.* KDD 2019. (SMD)
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. *Detecting
  spacecraft anomalies using LSTMs and nonparametric dynamic thresholding.* KDD 2018. (SMAP/MSL)
- Moody, G.B., Mark, R.G. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med
  and Biol 20(3):45-50, 2001. (MBA)
- Ahmad, S., Lavin, A., Purdy, S., Agha, Z. *Unsupervised real-time anomaly detection for
  streaming data.* Neurocomputing 262:134-147, 2017. (NAB)
- Keogh, E., Dutta Roy, T., Naik, U., Agrawal, A. *Multi-dataset Time-Series Anomaly
  Detection Competition.* SIGKDD 2021. (UCR)
