"""Bilingual reports from completed Kaggle runs; never manufacture missing runs."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import kaggle_data, kaggle_suite
from benchmarks.report import fmt, stats, table


def validate_complete(data):
    config = data["config"]
    expected, selections = set(), set()
    for case, seed, scaling in product(config["cases"], config["seeds"], ("raw", "rms")):
        for model in kaggle_suite.names(case):
            selections.add((case, seed, scaling, model))
            for solver, repeat in product(kaggle_suite.solvers(model), range(config["repeats"])):
                expected.add((case, seed, scaling, model, solver, repeat))
    keys = ("case", "seed", "scaling", "model", "solver", "repeat")
    observed = Counter(tuple(r[k] for k in keys) for r in data["results"])
    selected = Counter(tuple(r[k] for k in keys[:4]) for r in data["selections"])
    if set(observed) != expected or any(n != 1 for n in observed.values()) or set(selected) != selections or any(n != 1 for n in selected.values()):
        raise ValueError("Incomplete or duplicate Kaggle experiment; finish all slots before reporting")
    if set(data["datasets"]) != set(config["cases"]):
        raise ValueError("Incomplete dataset provenance")
    if data["evidence_kind"] == "kaggle-download" and any(
            m.get("evidence_kind") != "kaggle-download" for m in data["datasets"].values()):
        raise ValueError("Cannot label fixture evidence as Kaggle measurements")


def summarize(data):
    groups = defaultdict(lambda: defaultdict(list))
    for row in data["results"]:
        groups[tuple(row[k] for k in ("case", "scaling", "model", "solver"))][row["seed"]].append(row)
    output = []
    for key, seeds in sorted(groups.items()):
        item = dict(zip(("case", "scaling", "model", "solver"), key))
        item.update(seeds=len(seeds), statuses=dict(Counter(r["status"] for rs in seeds.values() for r in rs)),
                    successful_seeds=sum(all(r.get("test") is not None for r in rs) for rs in seeds.values()),
                    reached_seeds=sum(all(r.get("target") is not None for r in rs) for rs in seeds.values()))
        def aggregate(get):
            per_seed = [stats([get(r) for r in rs]) for rs in seeds.values()]
            return stats([s["median"] for s in per_seed if s])
        for field in ("fit_seconds", "cpp_seconds", "objective_gap", "stationarity_norm", "nonzero_coefficients",
                      "prediction_rows_per_second", "worker_peak_rss_mib", "selection_seconds", "calibration_seconds"):
            item[field] = aggregate(lambda r: r.get(field))
        item["passes"] = aggregate(lambda r: r["row_visits"]/r["train_rows"] if "row_visits" in r else None)
        item["target_seconds"] = aggregate(lambda r: (r.get("target") or {}).get("seconds"))
        for metric in ("log_loss", "roc_auc", "average_precision", "accuracy", "brier", "rmse", "mae", "r2"):
            item[metric] = aggregate(lambda r: (r.get("test") or {}).get(metric))
        output.append(item)
    return output


def render(data, chinese=False):
    validate_complete(data)
    summaries = summarize(data)
    fixture = data["evidence_kind"] != "kaggle-download"
    lines = ["# Kaggle 真实数据评测" if chinese else "# Kaggle real-data benchmark", ""]
    if fixture:
        lines += ["**PROTOCOL FIXTURE — 仅验证流程，数据不是从 Kaggle 下载，不能作为算法效果或性能证据。**" if chinese else
                  "**PROTOCOL FIXTURE — execution checks only. These are not Kaggle downloads or algorithm quality/performance evidence.**", ""]
    lines += ["比较 logits 的同一逻辑回归模型的优化器，以及线性回归模型的正则化差异。所有表格先在每个种子内取重复中位数，再跨种子取中位数；方括号为种子间 IQR。" if chinese else
              "Compare optimizers of the same logits logistic model and regularization of linear models. Tables take repeat medians within seeds, then medians across seeds; brackets show the inter-seed IQR.", ""]
    lines += ["## 数据与来源" if chinese else "## Data and provenance", ""]
    lines += [table(["Dataset", "Task", "Version", "Observed license", "CSV SHA-256"], [
        [f"[{case}]({m['source_url']})", kaggle_data.CATALOG[case]["task"], m["version"],
         str(m["license"]).replace("|", "/").replace("\n", " "), m["csv_sha256"]]
        for case, m in data["datasets"].items()]), ""]
    for case in data["datasets"]:
        spec = kaggle_data.CATALOG[case]
        selection = next(r for r in data["selections"] if r["case"] == case)
        lines += [f"- **{case}**: {spec['purpose']} Rows={selection['rows']}; transformed p={selection['cols']}; "
                  f"storage={selection['storage']}; train/validation/test="
                  f"{selection['train_rows']}/{selection['validation_rows']}/{selection['test_rows']}."]
    lines += ["", "## 评测协议" if chinese else "## Protocol", ""]
    lines += [
        ("固定 70/15/15 分割，分类按标签分层。数值缺失值的中位数、类别字典和 RMS 仅拟合训练集；未知类别编码为全零，CSR 不中心化。Adult 的 fnlwgt 不用作特征或权重；ID 和标签不入特征。" if chinese else
         "Fixed 70/15/15 splits, stratified for classification. Numeric medians, category vocabularies and RMS are fit on training data only. Unseen categories map to zero; CSR is never centered. Adult fnlwgt is excluded as both predictor and weight; IDs and labels are excluded."), "",
        ("每种缩放独立用验证集选择正则化，分类最小化 log loss，回归最小化 MSE；候选为 0.0001/0.01/0.1，OLS 为 0。回归目标用训练集均值/标准差归一化用于求解，RMSE/MAE 返回原始费用单位。测试集不参与选择。" if chinese else
         "Select regularization separately per scaling using validation log loss (classification) or MSE (regression). Candidates are 0.0001/0.01/0.1, with OLS fixed at zero. Regression targets use training mean/std for optimization; RMSE/MAE are converted to original charge units. Test data never selects settings."), "",
        ("每个模型目标由独立 SciPy/NumPy/scikit-learn 参考解验证，选择成本单独记录。逻辑回归各优化器共用选定 L2；学习率按训练目标校准，SGD/Adam 固定学习率、batch=256，所有试验均保留。不同优化器搜索预算不完全相同。" if chinese else
         "Independent SciPy/NumPy/scikit-learn references verify each objective; selection cost is separate. Logistic solvers share the selected L2. Step sizes are calibrated by training objective, with constant-step SGD/Adam and batch=256; all trials are retained. Search budgets differ across optimizers."), "",
        ("计时：单线程，独立子进程，先预热，fit 含 Python 数据复制，核心时间取 C++ 历史；预测取 7 次中位数。RSS 为整个 worker 的 Linux VmHWM，含运行时、预处理与预热，并非求解器增量内存。完整轨迹仅保留首次重复。" if chinese else
         "Timing uses one thread, isolated workers and warm-up. Fit includes Python data copying; core time comes from C++ history. Prediction is the median of seven calls. RSS is whole-worker Linux VmHWM, including runtime, preprocessing and warm-up, not solver-incremental memory. Full traces retain the first repeat only."), "",
        (f"优化目标统一为 (F−Fref)/(1+|Fref|) ≤ {data['config']['target_gap']:g}；达到时间为核心历史时间。未达到者标为 —，不填 0；达标种子要求该种子所有重复均达标。各算法相同迭代预算不代表相同数据扫描次数。" if chinese else
         f"The shared optimization target is (F−Fref)/(1+|Fref|) ≤ {data['config']['target_gap']:g}; target time uses core history. Misses remain —, never zero. A reached seed requires every repeat to reach the target. Equal iteration budgets do not imply equal data passes."), ""]
    for case in data["datasets"]:
        classification = kaggle_data.CATALOG[case]["task"] == "classification"
        subset = [s for s in summaries if s["case"] == case]
        lines += [f"## {case}", ""]
        fields = ("log_loss", "roc_auc", "average_precision", "accuracy", "brier") if classification else ("rmse", "mae", "r2")
        lines += [table(["Scaling", "Model", "Solver", *fields, "nnz"], [
            [s["scaling"], s["model"], s["solver"], *[fmt(s, m, interval=True) for m in fields],
             fmt(s, "nonzero_coefficients", digits=1)] for s in subset]), ""]
        lines += [table(["Scaling", "Model", "Solver", "Fit ms [IQR]", "Core ms", "Passes", "Reached seeds",
                         "Target ms*", "Gap", "Predict rows/s", "RSS MiB", "LR tuning s", "Statuses"], [
            [s["scaling"], s["model"], s["solver"], fmt(s,"fit_seconds",1000,interval=True),
             fmt(s,"cpp_seconds",1000), fmt(s,"passes",digits=1), f"{s['reached_seeds']}/{s['seeds']}",
             fmt(s,"target_seconds",1000), fmt(s,"objective_gap",digits=6),
             fmt(s,"prediction_rows_per_second",digits=0), fmt(s,"worker_peak_rss_mib",digits=1),
             fmt(s,"calibration_seconds"),
             json.dumps(s["statuses"],sort_keys=True)] for s in subset]), ""]
        selections = [r for r in data["selections"] if r["case"] == case]
        groups = defaultdict(list)
        for r in selections:
            groups[(r["scaling"],r["model"])].append(r)
        rows = []
        for (scaling,model), selected in sorted(groups.items()):
            field = "log_loss" if classification else "rmse"
            rows.append([scaling,model,str([r["alpha"] for r in selected]),
                         f"{np.median([r['baseline_test'][field] for r in selected]):.4f}",
                         f"{np.median([r['reference_test'][field] for r in selected]):.4f}",
                         f"{np.median([r['selection_seconds'] for r in selected]):.4f}"])
        lines += [table(["Scaling","Model","Selected alpha per seed",f"Dummy {field}",
                         f"Reference {field}","Selection seconds"], rows), ""]
    lines += ["## 解读与局限" if chinese else "## Interpretation and limits", "",
              ("缩放后重新选择 L1/L2 会改变有效模型先验，raw/RMS 差异包含优化和正则化两方面，不能全部归因于条件数。小数据集的耗时受调用开销影响；这些实验不代表大规模或多线程扩展能力。" if chinese else
               "Scaling and reselecting L1/L2 changes the effective model prior. Raw/RMS differences mix optimization and regularization, and cannot all be attributed to conditioning. Small-dataset timing includes substantial call overhead; these experiments do not establish large-scale or multithreaded scaling."), "",
              ("逻辑回归的五个优化器共享同一假设空间。类别不平衡时联合看 average_precision、ROC AUC、log loss 和 Brier；accuracy 使用固定 0.5 阈值。Insurance 仅使用原始特征，没有交互项，线性模型的逼近误差不会因更快的优化自动消失。真实数据没有已知真系数，因此不计算支持集恢复率或虚构正确的参数边界。" if chinese else
               "The five logistic optimizers share one hypothesis space. Under imbalance, inspect average_precision, ROC AUC, log loss and Brier jointly; accuracy uses a fixed 0.5 threshold. Insurance uses original features without interactions, so faster optimization cannot remove linear-model approximation error. Real data has no known true coefficients; support recovery and purported ground-truth coefficient bounds are not reported."), "",
              ("*达标时间仅汇总实际达到者，必须结合达标种子数阅读，不能据此直接排名。数据为历史观察样本，随机分割不能证明部署时的跨时间、跨人群泛化或因果关系。许可证文本来自下载时 Kaggle 元数据，原始文件保留本地，不随报告重新分发。" if chinese else
               "*Target times summarize only observed hits and must be read with seed counts; they do not alone rank solvers. Historical observational data and random splits do not establish future, cross-population or causal validity. License text is observed Kaggle metadata; original files stay local and are not redistributed with reports."), "",
              "## Reproducibility", "", "```bash",
              "python benchmarks/kaggle_data.py",
              "python benchmarks/kaggle_suite.py --output build/kaggle-benchmark",
              "python benchmarks/kaggle_report.py --input build/kaggle-benchmark --output build/kaggle-report",
              "```", "", f"Seeds: {data['config']['seeds']}; repeats: {data['config']['repeats']}; "
              f"iterations: {data['config']['iterations']}. CPU: {data['machine']['cpu']}. "
              f"Python {data['machine']['python']}; logitlab {data['machine']['logitlab']}; "
              f"scikit-learn {data['machine']['sklearn']}.", "",
              "Complete dataset manifests, build/source hashes, split/scaler hashes, hyperparameter trials and statuses: `results.json`; "
              "measurements: `metrics.csv`; first-repeat histories: `traces.csv`. Paths and content hashes are recorded in `report-manifest.json`.", ""]
    return "\n".join(lines)


def figures(data, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logitlab-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/logitlab-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = summarize(data)
    fig, axes = plt.subplots(1,len(data["datasets"]),figsize=(6*len(data["datasets"]),4.5),squeeze=False)
    for ax,case in zip(axes[0],data["datasets"]):
        classification = kaggle_data.CATALOG[case]["task"] == "classification"
        for i,s in enumerate(s for s in summaries if s["case"] == case):
            x, y = s["fit_seconds"],s["log_loss" if classification else "rmse"]
            if x is None or y is None:
                continue
            marker = "o" if s["scaling"] == "rms" else "x"
            ax.errorbar(x["median"]*1000,y["median"],
                        xerr=[[1000*(x["median"]-x["q1"])],[1000*(x["q3"]-x["median"])]],
                        yerr=[[y["median"]-y["q1"]],[y["q3"]-y["median"]]],
                        fmt=marker,capsize=2,label=f"{s['model']}/{s['solver']}/{s['scaling']}")
        ax.set(xlabel="Fit milliseconds (log scale)",ylabel="Test log loss" if classification else "Test RMSE (charge units)",
               title=case,xscale="log")
        ax.grid(alpha=.2); ax.legend(fontsize=6,loc="best")
    fig.suptitle("PROTOCOL FIXTURE — not Kaggle measurements" if data["evidence_kind"] != "kaggle-download" else
                 "Kaggle: quality and fit cost (seed medians and IQR)")
    fig.tight_layout()
    fig.savefig(output/"quality-cost.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.input/"results.json").read_text())
    validate_complete(data)
    args.output.mkdir(parents=True, exist_ok=True)
    for chinese,filename in ((False,"kaggle.en.md"),(True,"kaggle.zh-CN.md")):
        text = render(data,chinese)
        text += "\n![Quality and cost](quality-cost.png)\n"
        (args.output/filename).write_text(text)
    figures(data,args.output)
    (args.output/"summary.json").write_text(json.dumps(summarize(data),indent=2)+"\n")
    files = [args.input/name for name in ("results.json","metrics.csv","traces.csv")]
    files += [Path(__file__), args.output/"quality-cost.png",args.output/"kaggle.en.md",args.output/"kaggle.zh-CN.md"]
    manifest = {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (args.output/"report-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
