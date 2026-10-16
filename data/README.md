# Dataset sources

The versioned Kaggle download/benchmark workflow is documented in [English](../docs/kaggle-benchmark.md) and [中文](../docs/kaggle-benchmark.zh-CN.md). Raw CSV caches are stored in `data/kaggle/` and excluded from Git. Downloaded version/license metadata and SHA-256 are verified before reuse and copied into benchmark results. Current real-data downloads await cloud network configuration; no Kaggle results are claimed yet.

## Historical fixtures

http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
http://www.is.umk.pl/projects/datasets.html


https://www.kaggle.com/c/data-science-london-scikit-learn  40 features


The following legacy table has no retained split, preprocessing, tuning or timing protocol. It is preserved as a historical note and must not be compared with the current held-out benchmark reports. In particular, `heart_scale` is already scaled upstream and cannot establish a raw-versus-scaled experiment.

DataSet | Method | AUC
------- | ------ | ---
heart   | FTRL-LR   |0.9109
heart   | EP-LR | 0.8982
heart | CART | 0.8231
heart | RDT | 0.9155
heart | RF | 0.9019
heart | GBDT | 0.9061
fourclass | FTRL-LR | 0.8281
fourclass | EP-LR | 0.7986
fourclass | CART | 0.9832
fourclass | RDT | 0.9925
fourclass | RF | 0.9947
fourclass | GBDT | 0.9958
