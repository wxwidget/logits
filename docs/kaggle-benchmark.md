# Kaggle benchmark

[中文说明](kaggle-benchmark.zh-CN.md)

This study connects the controlled synthetic benchmarks to public tabular data. It compares logits optimization and predictive quality; it does not introduce another model family or change the C++ algorithms.

**Status, 2026-10-16:** the downloader, runner and bilingual report generator have been exercised with explicitly labeled offline protocol fixtures. Actual Kaggle downloads and real-data measurements remain blocked by the cloud proxy's domain allowlist (HTTP CONNECT 403). No Kaggle version, license, checksum, performance number or winner has been invented. Existing synthetic reports are separate evidence.

## Datasets and research questions

| Case | Kaggle dataset / expected file | What it examines |
| --- | --- | --- |
| `breast_cancer` | [uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), `data.csv` | Thirty correlated numerical measurements in different units: GD/SGD/Adam versus L-BFGS/Newton-CG, before and after RMS scaling. `M` is positive, `B` negative; IDs are excluded. |
| `adult` | [uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income), `adult.csv` | Imbalanced income prediction with numerical/categorical features. Training-only one-hot encoding preserves CSR. `>50K` is positive; census sampling weight `fnlwgt` is excluded rather than treated as a predictor. |
| `insurance` | [mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance), `insurance.csv` | OLS/Ridge/Lasso/Elastic Net on age, BMI, children, sex, smoker and region. Compare regularization, coefficient sparsity and convergence; no target-derived features or interactions are added. |

These datasets are deliberately small enough for repeated objective verification. Adult adds a larger row count and sparse categorical representation; none of these alone demonstrates billion-row or extremely wide sparse scalability. Keep the existing synthetic `sparse_wide` experiment for controlled dimensional scaling.

## Download and run

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/kaggle_data.py
python benchmarks/kaggle_suite.py --output build/kaggle-benchmark
python benchmarks/kaggle_report.py \
  --input build/kaggle-benchmark --output build/kaggle-report
```

The runner defaults to seeds 42/43/44, three measured repetitions, 200 iterations/epochs and both raw/RMS features. It does not download implicitly. To check execution with actual cached data using a smaller budget:

```bash
python benchmarks/kaggle_suite.py --cases insurance --seeds 42 \
  --repeats 1 --iterations 2 --output build/kaggle-smoke
```

A two-iteration run is a protocol check and cannot establish performance. For compiler flags, pass `--build-cache /path/to/the/installed/wheel/build` to the full suite, as in the existing benchmark protocol. Linux `/proc` is required for the worker memory measurement.

Output includes `results.json`, `metrics.csv` and `traces.csv`; reports are `kaggle.en.md`, `kaggle.zh-CN.md`, a quality/cost figure, `summary.json` and a content-hash manifest. Each solver's coefficients are also retained in original feature/target units with feature names in the selection records. L1 nonzero counts use a `1e-6` threshold in optimization coordinates, so inspect coefficients when comparing scaling variants.

## Cache, provenance and reproducibility

Downloads use public Kaggle HTTPS APIs without reading credentials. The downloader first reads the current metadata and requires a declared version and license. It downloads that explicit `datasetVersionNumber`, validates ZIP paths/size, UTF-8 CSV schema and target mapping, then atomically writes `data/kaggle/<case>/`:

```text
manifest.json    # observed version/license, URLs, timestamp, sizes and SHA-256
metadata.json    # exact observed Kaggle metadata
<dataset>.csv   # original CSV
```

Existing caches are verified and reused without network calls. A changed CSV or metadata file is rejected. SHA-256 detects changes relative to the initial download; it is not a publisher signature. First-time downloads observe the current version, so retain the cache/manifests to reproduce the same experiment. Reports retain the versioned download URL and CSV hash even when raw files are not shared. The raw cache is gitignored; review the observed license before redistributing original data. No unverified dataset license is declared in the catalog.

If Kaggle changes the schema, license metadata, file name or public access behavior, the command fails explicitly. It never substitutes synthetic samples or a third-party mirror for an unavailable download. Alternate files require a reviewed adapter/catalog change and corresponding tests.

## Training and comparison protocol

- Split rows 70/15/15, with stratification for classification. Fit numeric missing-value medians, category vocabulary and RMS on training rows only; `?`/empty categorical values become a missing category. An all-missing training numeric column uses zero. Unknown held-out categories map to all-zero one-hot blocks; CSR is never centered. No rows are silently dropped.
- Numerical fields are explicitly allowlisted. Missing markers are imputed, but NaN/Inf spellings and malformed fields fail. Labels and IDs never enter the feature matrix. Encoded feature counts may differ across seeds because only training categories are learned.
- For regression, standardize the target using the training mean/std. Solve the documented mean-loss objective in those units; report RMSE and MAE back in original charge units. Dummy baselines use training prevalence or training target mean.
- Select penalties from `0.0001, 0.01, 0.1` using independent reference fits and validation log loss/MSE; OLS uses zero. Each model/scaling pair has its own selection. Test labels never choose parameters. Different scaling changes both geometry and the meaning of L1/L2, so raw/RMS quality differences are not a pure conditioning experiment.
- Logistic solvers share one selected L2. Existing training-objective step-size grids calibrate each optimizer; all trials/costs are retained. GD/SGD/Adam have three trials, L-BFGS/Newton-CG one. SGD/Adam use a constant step and batch size 256; their stopping statuses and objective drift remain visible.
- Independent SciPy Newton-CG/L-BFGS verifies logistic objectives in exactly transformed coordinates; NumPy/SciPy least squares and scikit-learn coordinate descent verify linear objectives. References must satisfy the documented `1e-5` stationarity check before selection. Preconditioned gradient norm is explicitly identified for logistic references.
- Use normalized gap `(F-Fref)/(1+|Fref|) <= 1e-5` for matched-target work/time. Errors, budget exhaustion and failed line searches remain distinct. A model materially below its reference invalidates its gap/target comparison. Missing target times remain censored; inspect hit counts before interpreting conditional timings.
- Single-thread workers measure warmed Python-facing fit latency, C++ history time, seven-call prediction median, row visits and whole-worker Linux VmHWM. Selection/calibration and preprocessing costs are separate. RSS includes the interpreter, preprocessing, data and warm-up; CPU frequency/background load are not isolated.
- Aggregate repetitions within each seed before aggregating across seeds. Reports give medians and seed IQR. Full traces retain repeat zero only; every repetition retains metrics, work and first-target evidence. Reports reject incomplete/duplicate experiments.

Classification metrics are log loss, ROC AUC, **average precision** (scikit-learn's non-interpolated definition), accuracy at a fixed 0.5 threshold and Brier score. Average precision is not the trapezoidal area under the precision-recall curve. Regression metrics are RMSE, MAE and R². Finite-budget test quality and independently optimized reference quality are reported separately.

Real data does not reveal true support or coefficient bounds. The suite does not manufacture support-recovery scores or claim a coefficient constraint is known to be correct. Insurance's baseline linear model omits interactions, so solver convergence alone cannot remove approximation error. Synthetic constraint studies remain in the separate linear suite.

## Cloud network prerequisite

The environment draft now preserves `api.github.com` and the existing package-manager preset while adding `www.kaggle.com`, `kaggle.com`, `api.kaggle.com` and `storage.googleapis.com`. The configuration tool reported `status: saved` and `requires_publish: true`. **Saving the draft did not activate these rules.** Review and save them in cloud environment settings, then publish the environment and retry the download command. GitHub tokens do not resolve a Kaggle CONNECT 403. No Kaggle token is requested for this public-download path.

## TDD evidence

`tests/test_kaggle_benchmark.py` initially failed at collection because the three new modules did not exist. Tests now exercise version-pinned downloads, observed-license requirements, atomic/unsafe archive handling, CSV/metadata checksum failures, target/schema validation, disjoint splits, train-only imputation/encoding/scaling, test-label independence of penalty selection, average precision with ties, seed aggregation and end-to-end worker/report execution. They use tiny generated CSV fixtures and perform no network requests.

The runner rejects fixtures unless explicitly invoked with `--allow-fixtures`. Such outputs are always labeled **PROTOCOL FIXTURE**, including their figures and both languages. Passing these tests validates the protocol, not Kaggle availability or measured real-data quality.
