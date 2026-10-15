"""Generate matching English/Chinese reports and scientific figures from raw data."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import numpy as np

SOLVERS = ("gd", "sgd", "adam", "lbfgs", "newton-cg")
COLORS = dict(zip(SOLVERS, ("#4477AA", "#EE6677", "#228833", "#AA8822", "#AA3377")))


def validate_complete(data):
    config = data["config"]
    expected = set()
    for case, seed, scaling, l2, solver in product(
            config["cases"], config["seeds"], ("raw", "rms"), config["l2"], SOLVERS):
        variants = ["baseline"]
        if config["ablations"] and scaling == "rms":
            if case == "sparse_wide" and solver in ("sgd", "adam") and l2 == .01:
                variants += [f"batch_{size}" for size in (32, 1024) if size != config["batch_size"]]
            if case == "dense_correlated" and solver == "lbfgs":
                variants += ["history_3", "history_30"]
        for variant, repeat in product(variants, range(config["repeats"])):
            expected.add((case, seed, scaling, l2, solver, variant, repeat))
    observed = Counter(tuple(row[key] for key in (
        "case", "seed", "scaling", "l2", "solver", "variant", "repeat")) for row in data["results"])
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise ValueError(f"Incomplete or duplicate experiment: {len(expected - set(observed))} "
                         "missing measurement slots. Finish the suite before generating a report.")
    refs = {(r["case"], r["seed"], r["scaling"], r["l2"]) for r in data["references"]}
    if refs != set(product(config["cases"], config["seeds"], ("raw", "rms"), config["l2"])):
        raise ValueError("Incomplete reference evidence.")


def stats(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return dict(median=float(median), q1=float(q1), q3=float(q3), n=len(values))


def summarize(rows, gap_target):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[k] for k in ("case", "l2", "scaling", "solver", "variant"))
        groups[key][row["seed"]].append(row)
    output = []
    for key, seeds in groups.items():
        s = dict(zip(("case", "l2", "scaling", "solver", "variant"), key))
        s.update(seeds=len(seeds), repeats=sum(len(r) for r in seeds.values()),
                 statuses=dict(Counter(r[0]["status"] for r in seeds.values())),
                 ever_target_seeds=sum(any(r.get("target") is not None for r in runs)
                                       for runs in seeds.values()),
                 final_target_seeds=sum(all(r.get("objective_gap") is not None and
                                           r["objective_gap"] <= gap_target for r in runs)
                                        for runs in seeds.values()),
                 reference_valid_seeds=sum(all(r.get("reference_valid", False) for r in runs)
                                           for runs in seeds.values()))
        for metric in ("fit_seconds", "cpp_seconds", "objective_gap", "gradient_norm", "epochs",
                       "row_visits", "full_evaluations", "batch_gradients", "hessian_vector_products",
                       "parameter_updates", "prediction_rows_per_second", "worker_peak_rss_mib",
                       "calibration_seconds"):
            medians = [stats([r.get(metric) for r in runs]) for runs in seeds.values()]
            s[metric] = stats([m["median"] for m in medians if m])
        for metric in ("log_loss", "accuracy", "roc_auc", "brier"):
            medians = [stats([r.get("test", {}).get(metric) for r in runs]) for runs in seeds.values()]
            s["test_" + metric] = stats([m["median"] for m in medians if m])
        s["total_passes"] = stats([
            np.median([r["row_visits"] / r["train_rows"] for r in runs if "row_visits" in r])
            for runs in seeds.values() if any("row_visits" in r for r in runs)])
        for metric in ("seconds", "row_visits"):
            medians = [stats([r["target"][metric] /
                              (r["train_rows"] if metric == "row_visits" else 1)
                              for r in runs if r.get("target") is not None])
                       for runs in seeds.values()]
            s["target_" + ("passes" if metric == "row_visits" else metric)] = stats(
                [m["median"] for m in medians if m])
        output.append(s)
    return output


def value(s, key):
    return s[key]["median"] if s.get(key) else None


def fmt(s, key, factor=1, digits=3, interval=False):
    a = s.get(key)
    if a is None:
        return "—"
    text = f"{a['median'] * factor:.{digits}f}"
    if interval:
        text += f" [{a['q1'] * factor:.{digits}f}, {a['q3'] * factor:.{digits}f}]"
    return text


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def figures(data, summaries, source, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logitlab-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/logitlab-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "logitlab-benchmark"})
    output.mkdir(parents=True, exist_ok=True)
    cases = data["config"]["cases"]
    lookup = {(s["case"], s["l2"], s["scaling"], s["solver"], s["variant"]): s for s in summaries}
    refs = {(r["case"], r["seed"], r["scaling"], r["l2"]): r for r in data["references"]}
    traces = defaultdict(list)
    seed = data["config"]["seeds"][0]
    with (source / "traces.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["seed"]) == seed and int(row["repeat"]) == 0 and row["variant"] == "baseline":
                traces[(row["case"], float(row["l2"]), row["scaling"], row["solver"])].append(row)

    def save(fig, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else {"Software": "LogitLab"}
            fig.savefig(path, dpi=160, bbox_inches="tight", metadata=metadata)
            if extension == "svg":
                # Matplotlib emits trailing spaces in multiline path attributes.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    synthetic = [c for c in cases if c != "heart"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, case in zip(axes.flat, synthetic):
        for solver in SOLVERS:
            for scaling in ("raw", "rms"):
                trace = traces.get((case, 0., scaling, solver), [])
                ref = refs.get((case, seed, scaling, 0.))
                if not trace or not ref or not ref["verified"]:
                    continue
                ax.plot([max(1e-6, float(r["seconds"])) for r in trace],
                        [max(1e-10, float(r["loss"]) - ref["objective"]) for r in trace],
                        color=COLORS[solver], ls="--" if scaling == "raw" else "-",
                        alpha=.85, label=f"{solver} / {scaling}")
        ax.axhline(data["config"]["target_gap"], color="#666666", lw=1, ls=":")
        ax.set(xscale="log", yscale="log", title=case,
               xlabel="C++ optimization time (s)", ylabel="Training objective gap")
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(synthetic):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Convergence: lambda=0, seed={seed}, repeat=0 (dashed: raw, solid: RMS)")
    fig.tight_layout()
    save(fig, "convergence")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, scaling in zip(axes, ("raw", "rms")):
        matrix = np.full((len(cases), len(SOLVERS)), np.nan)
        for i, case in enumerate(cases):
            for j, solver in enumerate(SOLVERS):
                s = lookup.get((case, 0., scaling, solver, "baseline"))
                gap = value(s, "objective_gap") if s else None
                if gap is not None:
                    matrix[i, j] = np.log10(max(1e-10, gap))
                if s:
                    ax.text(j, i, f"{s['final_target_seeds']}/{s['seeds']}", ha="center", va="center",
                            color="black" if np.isnan(matrix[i, j]) or matrix[i, j] > -6 else "white")
        im = ax.imshow(matrix, vmin=-10, vmax=0, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(SOLVERS)), xticklabels=SOLVERS, yticks=range(len(cases)),
               yticklabels=cases, title=f"{scaling}: final gap; text = target seeds")
    fig.colorbar(im, ax=list(axes), label="log10(median final objective gap)", shrink=.8)
    save(fig, "scaling")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, case in zip(axes.flat, cases):
        for solver in SOLVERS:
            for scaling in ("raw", "rms"):
                s = lookup.get((case, .01, scaling, solver, "baseline"))
                if s and s["test_log_loss"] and s["fit_seconds"]:
                    x, y = value(s, "fit_seconds") * 1000, value(s, "test_log_loss")
                    timing, loss = s["fit_seconds"], s["test_log_loss"]
                    ax.errorbar(x, y, xerr=[[x - timing["q1"] * 1000], [timing["q3"] * 1000 - x]],
                                yerr=[[y - loss["q1"]], [loss["q3"] - y]],
                                fmt="none", color=COLORS[solver], alpha=.3, capsize=2, lw=1)
                    ax.scatter(value(s, "fit_seconds") * 1000, value(s, "test_log_loss"),
                               color=COLORS[solver], marker="x" if scaling == "raw" else "o",
                               s=55, label=f"{solver} / {scaling}")
        ax.set(xscale="log", title=case, xlabel="Full fit time, median (ms)",
               ylabel="Test log loss, median")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].set_visible(True)
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles, labels, fontsize=10, ncol=2, loc="center")
    fig.suptitle("Prediction and cost: lambda=0.01 (x: raw, circle: RMS; bars: seed IQR)")
    fig.tight_layout()
    save(fig, "quality-performance")


PROTOCOL = {
    "en": [
        "Five optimizers fit one binary logistic regression model. This compares convergence, prediction and cost; the hypothesis class is the same.",
        "Stratified 70/15/15 train/validation/test splits are shared within each seed. RMS uses training rows only, without centering; CSR stays sparse and zero-RMS columns use scale 1. Validation metrics are retained, but neither validation nor test labels select configurations.",
        "Two tracks: λ=0 isolates coordinate conditioning since nonzero feature rescaling preserves attainable predictions. λ=0.01 studies the practical pipeline: isotropic L2 after scaling changes the penalty on original coefficients. Its predictive changes cannot be attributed to conditioning alone. There is no λ search.",
        "Calibration: GD rates 0.05/0.5/5; SGD 0.001/0.01/0.1; Adam 0.0001/0.001/0.01. Select the lowest FINAL TRAINING objective and refit from zero. L-BFGS/Newton-CG each use one configuration with line search. Search budgets differ; calibration time is separate. This bounded grid cannot establish a fully tuned winner.",
        "The independent SciPy Newton-CG/L-BFGS-B reference has a separately written objective, gradient and exact HVP. RMS preconditioning transforms the penalty to λ/scale², preserving the original objective even on raw features. A preconditioned full-gradient norm ≤1e-5 validates the reference; materially better library results invalidate gap/target statistics. Target: objective gap ≤1e-4.",
        "Fit timing includes Python conversion, C++ validation and workspace allocation. C++ trace/target time excludes these. Small training and prediction warm-ups precede timing. Prediction uses seven calls per fit, includes conversion, excludes preprocessing. All solvers share the same prediction kernels; small throughput differences do not establish an optimizer-specific predictor advantage. RSS is whole isolated-worker peak across repetitions, including interpreter/data/warm-up, not incremental solver memory. Linux uses post-exec VmHWM to exclude inherited launcher high-water marks. Workers run sequentially.",
        "Aggregate repetitions within each seed first, then report median [Q1,Q3] across seed medians. Repeated timing is not an independent dataset. Three seeds are descriptive evidence, not significance tests. ROC-AUC uses average ranks for ties; accuracy threshold is 0.5; Brier is mean squared probability error. Test log loss clips probabilities to [1e-15,1−1e-15].",
    ],
    "zh-CN": [
        "五种优化器使用同一个二分类逻辑回归模型。本报告比较收敛、预测效果与成本，模型表达能力相同。",
        "按类别分层划分训练/验证/测试集，比例约 70/15/15，同一种子共享划分。RMS 仅在训练集拟合且不减均值，CSR 保持稀疏，零 RMS 列尺度为 1。保留验证集指标，但验证集与测试集标签均不参与配置选择。",
        "两条轨道：λ=0 时非零特征缩放不改变可表达的预测，用于研究数值条件；λ=0.01 研究实际流程，缩放坐标中的各向同性 L2 改变了原始系数的惩罚。因此预测效果变化不能全部归因于收敛改善。本实验未搜索 λ。",
        "校准：GD 步长 0.05/0.5/5，SGD 为 0.001/0.01/0.1，Adam 为 0.0001/0.001/0.01。选择最终训练目标最小的配置，从零重训。L-BFGS/Newton-CG 使用线搜索，各运行一个配置。搜索预算不同，校准成本单列。有限网格不能证明充分调参后的最优算法。",
        "独立 SciPy Newton-CG/L-BFGS-B 参考采用另外编写的目标、梯度和精确 HVP；RMS 预条件同步将惩罚变换为 λ/scale²，保持原始目标。预条件坐标中的完整梯度范数 ≤1e-5 才接受参考解；库中结果显著低于参考时禁用 gap/达标统计。目标为训练目标差 ≤1e-4。",
        "fit 时间包含 Python 转换、C++ 校验和工作区分配，C++ trace/首次达标时间排除这些成本。计时前预热小规模训练及预测。预测吞吐取七次调用中位数，包含转换，不含预处理。所有优化器共用预测内核，细小吞吐差异不能解释为某优化器预测实现更快。RSS 是独立进程多次重复的整体峰值，包含解释器、数据和预热，不能当作算法增量内存。Linux 使用 exec 后的 VmHWM 排除继承的启动进程峰值。进程顺序运行。",
        "先对同一种子的重复计时取中位数，再对种子中位数统计 median [Q1,Q3]；重复计时不是独立数据集，三个种子只支持描述性结论。ROC-AUC 采用平均秩处理并列；Accuracy 阈值 0.5；Brier 为概率均方误差；测试 log loss 将概率裁剪到 [1e-15,1−1e-15]。",
    ],
}

DESCRIPTIONS = {
    "dense_control": ("Comparable-scale Gaussian control.", "尺度相近的高斯稠密对照。"),
    "dense_multiscale": ("Factors 0.01–100; label signal generated before rescaling.", "倍率 0.01–100；标签信号在缩放前生成。"),
    "dense_correlated": ("Four-feature correlated groups, noise 0.05, opposing coefficients. RMS does not whiten.",
                         "每四列高度相关，噪声 0.05，系数互相抵消；RMS 无法白化。"),
    "sparse_wide": ("CSR density 0.01 (quick: 0.08), factors 0.01–100; sparse traversal and dense updates.",
                    "CSR 密度 0.01（quick 为 0.08），倍率 0.01–100；考察稀疏遍历和稠密更新。"),
    "heart": ("Bundled LIBSVM heart_scale; +1→1/−1→0, unspecified coordinates→0, no imputation; already scaled upstream.",
              "仓库 LIBSVM heart_scale；+1→1/−1→0，未给出坐标为 0，不做缺失值填补；上游已经缩放。"),
}


def findings(summaries, zh):
    t = lambda en, cn: cn if zh else en
    lookup = {(s["case"], s["l2"], s["scaling"], s["solver"]): s
              for s in summaries if s["variant"] == "baseline"}
    lines = []
    raw = lookup.get(("dense_multiscale", 0., "raw", "lbfgs"))
    rms = lookup.get(("dense_multiscale", 0., "rms", "lbfgs"))
    if raw and rms:
        lines.append(t(
            f"- **Scaling resolves unit imbalance:** multiscale L-BFGS final-target seeds "
            f"{raw['final_target_seeds']}/{raw['seeds']} → {rms['final_target_seeds']}/{rms['seeds']}; "
            f"median test log loss {fmt(raw, 'test_log_loss')} → {fmt(rms, 'test_log_loss')}.",
            f"- **Scaling 缓解量级失衡：**多尺度 L-BFGS 最终达标种子数 "
            f"{raw['final_target_seeds']}/{raw['seeds']} → {rms['final_target_seeds']}/{rms['seeds']}；"
            f"测试 log loss 中位数 {fmt(raw, 'test_log_loss')} → {fmt(rms, 'test_log_loss')}。"))
    correlated = [lookup.get(("dense_correlated", 0., "rms", solver))
                  for solver in ("gd", "sgd", "adam", "lbfgs", "newton-cg")]
    if all(correlated):
        counts = ", ".join(f"{s['solver']} {s['final_target_seeds']}/{s['seeds']}" for s in correlated)
        lines.append(t(f"- **Scaling does not remove correlation:** correlated RMS final-target seeds: {counts}.",
                       f"- **Scaling 不消除相关性：**相关特征 RMS 最终达标种子数：{counts}。"))
    raw = lookup.get(("sparse_wide", 0., "raw", "lbfgs"))
    rms = lookup.get(("sparse_wide", 0., "rms", "lbfgs"))
    regularized = lookup.get(("sparse_wide", .01, "rms", "lbfgs"))
    if raw and rms and regularized:
        lines.append(t(
            f"- **Optimization and generalization differ:** sparse L-BFGS test log loss "
            f"raw λ=0 {fmt(raw, 'test_log_loss')}, RMS λ=0 {fmt(rms, 'test_log_loss')}, "
            f"RMS λ=0.01 {fmt(regularized, 'test_log_loss')}. Check the training gaps and regularization "
            "together; a smaller training objective does not guarantee a better test score.",
            f"- **收敛与泛化不同：**稀疏 L-BFGS 测试 log loss：原始 λ=0 {fmt(raw, 'test_log_loss')}，"
            f"RMS λ=0 {fmt(rms, 'test_log_loss')}，RMS λ=0.01 {fmt(regularized, 'test_log_loss')}。"
            "应结合训练目标差和正则解读，训练目标更低不保证测试效果更好。"))
    return lines


def render(data, summaries, lang, raw_link):
    zh = lang == "zh-CN"
    t = lambda en, cn: cn if zh else en
    config, machine = data["config"], data["machine"]
    baseline = [s for s in summaries if s["variant"] == "baseline"]
    references = data["references"]
    reproduction = ("python benchmarks/suite.py --cases " + " ".join(config["cases"]) +
                    " --seeds " + " ".join(map(str, config["seeds"])) +
                    " --l2 " + " ".join(map(str, config["l2"])) +
                    f" --epochs {config['epochs']} --batch-size {config['batch_size']}" +
                    f" --repeats {config['repeats']} --target-gap {config['target_gap']}" +
                    (" --quick" if config["quick"] else "") +
                    (" --no-ablations" if not config["ablations"] else "") +
                    " --output build/benchmark-suite")
    lines = [t("# LogitLab optimizer benchmark", "# LogitLab 优化器评测报告"), "",
             t("[中文版](benchmark.zh-CN.md)", "[English](benchmark.en.md)"), "",
             PROTOCOL[lang][0], "",
             t("## Main findings", "## 主要发现"), "",
             *findings(summaries, zh), "",
             t("## Evidence and reproduction", "## 数据与复现"), "",
             t(f"{len(data['results'])} measured fits; {len(config['seeds'])} seeds, "
               f"{config['repeats']} timed repeats per configuration. ",
               f"共 {len(data['results'])} 次实测训练，{len(config['seeds'])} 个种子，每个配置重复 {config['repeats']} 次。") +
             t(f"References passing gradient check: {sum(r['verified'] for r in references)}/{len(references)}.",
               f"独立参考解通过梯度检查：{sum(r['verified'] for r in references)}/{len(references)}。"), "",
             f"[JSON]({raw_link}/results.json) · [Metrics CSV]({raw_link}/metrics.csv) · "
             f"[Traces CSV]({raw_link}/traces.csv) · [Summary JSON](summary.json) · [Report manifest](report-manifest.json)", "",
             "~~~bash", "python -m pip install '.[test,benchmark]'",
             reproduction,
             "python benchmarks/report.py --input build/benchmark-suite --output build/benchmark-report", "~~~", "",
             t("These reproduce the recorded experiment settings. A small protocol check uses --quick --cases dense_control "
               "--seeds 42 --epochs 2 --repeats 1. Quick results are not performance evidence.",
               "以上复现本次实验设置。快速协议检查可用 --quick --cases dense_control --seeds 42 --epochs 2 --repeats 1。快速模式不能用于性能结论。"), "",
             t(f"Actual settings: λ={config['l2']}; seeds={config['seeds']}; quick={config['quick']}.",
               f"实际设置：λ={config['l2']}；种子={config['seeds']}；quick={config['quick']}。"), "",
             f"- CPU: {machine['cpu']}; affinity: {machine['cpu_affinity']}.",
             f"- Python {machine['python']}; NumPy {machine['numpy']}; SciPy {machine['scipy']}; LogitLab {machine['logitlab']}.",
             f"- {machine['compiler'].splitlines()[0]}; {machine['memory_total']}.",
             t("- Single-thread core/BLAS, portable Release build. CPU frequency is not isolated. "
               "JSON retains commit, dirty-tree state, source/binary hashes and NumPy build metadata.",
               "- 核心与 BLAS 单线程，可移植 Release 构建，未隔离 CPU 频率。JSON 保存提交、工作区状态、源码/二进制哈希和 NumPy 构建信息。"), "",
             t("## Protocol", "## 评测协议"), ""]
    for paragraph in PROTOCOL[lang][1:]:
        lines.extend([paragraph, ""])
    lines.extend([t(f"Budget per trial: {config['epochs']} epochs/outer steps, batch={config['batch_size']} "
                    "(except ablations), tolerance=1e-6, zero initialization.",
                    f"每次训练预算：{config['epochs']} 个 epoch/外层步，batch={config['batch_size']}（参数对照除外），容差 1e-6，零初始化。"), ""])
    scenario_rows = []
    for case in config["cases"]:
        ref = next(r for r in references if r["case"] == case)
        scenario_rows.append([case, f"{ref['rows']} × {ref['cols']}", ref["storage"], DESCRIPTIONS[case][int(zh)]])
    lines.extend([table([t("Scenario", "场景"), t("Shape", "规模"), t("Storage", "存储"), t("Purpose", "目的")], scenario_rows), "",
                  t("Heart provenance: [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart); "
                    "file checksum in JSON. Already-scaled heart is not a raw clinical scaling experiment. Its small timings "
                    "include substantial interface overhead; current splits cannot be compared to historical data/README.md AUC.",
                    "Heart 来源：[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart)，JSON 有文件校验和。"
                    "已经缩放的 Heart 不能作为原始临床特征缩放实验，小规模计时受到接口开销影响，当前划分不能与 data/README.md 的历史 AUC 直接比较。"), "",
                  t("## Scaling and convergence", "## Scaling 与收敛"), "",
                  "![Final gaps and target counts](figures/scaling.png)", "",
                  t("λ=0. Color: log10 median final gap; text: FINAL target seeds / seeds. Plotting floor: 1e-10.",
                    "λ=0。颜色：最终目标差中位数的 log10；数字：最终达标种子数/总种子数。仅绘图设下限 1e-10。"), "",
                  "![Convergence curves](figures/convergence.png)", "",
                  t("First seed/repeat examples, not averages. Dashed: raw; solid: RMS. Dotted: target. "
                    "SGD/Adam can cross the target and later move away.",
                    "首个种子、首次重复的示例，不是平均曲线。虚线：原始；实线：RMS；点线：目标。SGD/Adam 可能暂时达标后再次偏离。"), "",
                  t("## Prediction and cost", "## 预测效果与成本"), "",
                  "![Quality versus training time](figures/quality-performance.png)", "",
                  t("λ=0.01; x: raw, circle: RMS; bars: seed IQR. A shorter fit is not faster convergence "
                    "if its target was missed. Overlapping seed ranges limit conclusions about small score differences.",
                    "λ=0.01；叉号：原始，圆点：RMS；误差线：种子 IQR。未达标时，总训练时间短不能证明收敛更快。"
                    "种子区间重叠时，不能对细小效果差异下确定结论。"), ""])
    for case in config["cases"]:
        lines.extend([f"### {case}", ""])
        rows = [s for s in baseline if s["case"] == case]
        rows.sort(key=lambda s: (s["l2"], s["scaling"], SOLVERS.index(s["solver"])))
        quality = [[s["l2"], s["scaling"], s["solver"], fmt(s, "test_log_loss", interval=True),
                    fmt(s, "test_roc_auc"), fmt(s, "test_accuracy", 100, 1), fmt(s, "test_brier"),
                    f"{value(s, 'objective_gap'):.2e}" if s["objective_gap"] else "—",
                    f"{s['final_target_seeds']}/{s['seeds']}"] for s in rows]
        quality_headers = (["λ", "缩放", "优化器", "测试 Log loss [Q1,Q3]", "ROC-AUC", "准确率 %",
                            "Brier", "最终目标差", "最终达标"] if zh else
                           ["λ", "Scaling", "Solver", "Test log loss [Q1,Q3]", "ROC-AUC", "Accuracy %",
                            "Brier", "Final gap", "Final target"])
        lines.extend([table(quality_headers, quality), ""])
        costs = [[s["l2"], s["scaling"], s["solver"], fmt(s, "fit_seconds", 1000, 2, True),
                  fmt(s, "target_seconds", 1000, 2), f"{s['ever_target_seeds']}/{s['seeds']}",
                  fmt(s, "target_passes", digits=1), fmt(s, "total_passes", digits=1),
                  fmt(s, "prediction_rows_per_second", 1e-6), fmt(s, "worker_peak_rss_mib", digits=1),
                  fmt(s, "calibration_seconds", 1000, 2)] for s in rows]
        cost_headers = (["λ", "缩放", "优化器", "训练 ms [Q1,Q3]", "首次达标 C++ ms", "曾达标",
                         "达标遍历", "总遍历", "预测 百万行/s", "进程峰值 MiB", "校准 ms"] if zh else
                        ["λ", "Scaling", "Solver", "Fit ms [Q1,Q3]", "Target C++ ms", "Ever target",
                         "Target passes", "Total passes", "Predict M rows/s", "Worker RSS MiB", "Calibration ms"])
        lines.extend([table(cost_headers, costs), ""])
    lines.extend([t("A dash is missing/unreached, never zero or the budget endpoint. Target time/passes are "
                    "conditional on reaching: read the count. A row visit is one objective/gradient, batch, curvature "
                    "or HVP traversal of a row; passes=row visits/train rows. This proxy is not FLOPs and excludes "
                    "dense vector-update costs. Rejected line searches and stochastic monitoring passes are counted.",
                    "破折号表示缺失/未达标，不能当作 0 或预算截止时间。首次达标时间和遍历量仅统计曾达标种子，须结合数量阅读。"
                    "row visit 是目标/梯度、batch、曲率或 HVP 的一次行遍历；passes=row visits/训练行数。该代理不是 FLOPs，未包含稠密向量更新成本。"
                    "被拒绝的线搜索与随机优化器监测遍历均计入。"), "",
                  t("## Algorithm ablations", "## 算法参数对照"), ""])
    relevant = {(s["case"], s["l2"], s["scaling"], s["solver"]) for s in summaries if s["variant"] != "baseline"}
    ablations = [s for s in summaries if (s["case"], s["l2"], s["scaling"], s["solver"]) in relevant]
    ablations.sort(key=lambda s: (s["case"], s["l2"], s["solver"], s["variant"]))
    ablation_headers = (["场景", "λ", "优化器", "配置", "训练 ms", "测试 Log loss", "最终目标差", "更新次数", "最终达标"]
                        if zh else ["Case", "λ", "Solver", "Variant", "Fit ms", "Test log loss", "Final gap",
                                    "Updates", "Final target"])
    lines.extend([table(ablation_headers,
                        [[s["case"], s["l2"], s["solver"], s["variant"], fmt(s, "fit_seconds", 1000, 2),
                          fmt(s, "test_log_loss"), f"{value(s, 'objective_gap'):.2e}" if s["objective_gap"] else "—",
                          fmt(s, "parameter_updates", digits=0), f"{s['final_target_seeds']}/{s['seeds']}"]
                         for s in ablations]), "",
                  t("Baseline batch=256/history=10. Sparse RMS λ=0.01 compares batch 32/256/1024; correlated RMS "
                    "compares history 3/10/30. Stochastic rates are recalibrated per batch. Smaller batches make more "
                    "dense O(p) updates despite CSR. More L-BFGS history costs O(mp) memory/work and is not always better.",
                    "默认 batch=256/history=10。稀疏 RMS、λ=0.01 比较 batch=32/256/1024；相关特征 RMS 比较 history=3/10/30。"
                    "每种 batch 重新校准步长。CSR 也会因较小 batch 执行更多 O(p) 稠密更新；更多 L-BFGS 历史增加 O(mp) 内存与工作量，未必更好。"), "",
                  t("## Findings and recommendations", "## 结果与建议"), ""])
    for solver in SOLVERS:
        pair = [next((s for s in baseline if s["case"] == "dense_multiscale" and s["l2"] == 0
                      and s["scaling"] == scaling and s["solver"] == solver), None) for scaling in ("raw", "rms")]
        if all(pair):
            raw, rms = pair
            lines.append(t(f"- Multiscale **{solver}**: final target raw {raw['final_target_seeds']}/{raw['seeds']} "
                           f"→ RMS {rms['final_target_seeds']}/{rms['seeds']}; test log loss "
                           f"{fmt(raw, 'test_log_loss')} → {fmt(rms, 'test_log_loss')}.",
                           f"- 多尺度 **{solver}**：最终达标 原始 {raw['final_target_seeds']}/{raw['seeds']} "
                           f"→ RMS {rms['final_target_seeds']}/{rms['seeds']}；测试 log loss "
                           f"{fmt(raw, 'test_log_loss')} → {fmt(rms, 'test_log_loss')}。"))
    lines.extend(["", t("Use scaled L-BFGS as a batch reference; assess Newton-CG's precision against its extra HVP "
                        "passes. GD is a conditioning baseline. SGD/Adam expose batch/step/noise tradeoffs: check "
                        "final objective and held-out loss together. RMS balances magnitudes but does not whiten "
                        "correlations. Choose deployment regularization using a separate validation protocol.",
                        "可用缩放后的 L-BFGS 建立批量参考，评估 Newton-CG 的精度是否值得额外 HVP 遍历。GD 是数值条件基线；"
                        "SGD/Adam 用于研究 batch、步长与噪声，应同时检查最终训练目标和留出集 loss。RMS 平衡量级但不白化相关性。"
                        "部署时应通过独立验证协议选择正则。"), "",
                  t("## Limits and verification", "## 局限与验证"), "",
                  t("Synthetic labels and one 270-row real fixture do not establish industrial-scale performance. "
                    "No streaming, multicore scaling or incremental memory is measured. Seeds change both synthetic "
                    "data and splits. Local cloud timing, early stopping, constant-step noise, bounded grids and "
                    "coordinate-dependent gradient norms limit ranking claims. There is no universal winner.",
                    "合成标签和一个 270 行真实样本不能证明工业规模性能；未测流式训练、多核扩展或增量内存。种子同时改变合成数据与划分。"
                    "云机计时、提前停止、恒定步长噪声、有限网格和坐标相关的梯度范数均限制排名结论，没有通用最优算法。"), "",
                  t("Tests cover split disjointness, train-only RMS/CSR, transformed regularizer and gradient, AUC "
                    "ties, target censoring, actual work including failed line searches, and quick CLI. Reports are "
                    "generated from raw artifacts; both versions share figures with English labels.",
                    "测试覆盖划分不重叠、训练集 RMS/CSR、变换正则与梯度、AUC 并列、未达标处理、实际工作量及失败线搜索，并检查快速 CLI。"
                    "报告从原始产物生成，两种语言共用英文标注图表。"), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.input / "results.json").read_text())
    validate_complete(data)
    summaries = summarize(data["results"], data["config"]["target_gap"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    figures(data, summaries, args.input, args.output / "figures")
    link = Path(os.path.relpath(args.input.resolve(), args.output.resolve())).as_posix()
    for lang in ("en", "zh-CN"):
        (args.output / f"benchmark.{lang}.md").write_text(render(data, summaries, lang, link))
    import matplotlib
    manifest = dict(report_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    results_sha256=hashlib.sha256((args.input / "results.json").read_bytes()).hexdigest(),
                    traces_sha256=hashlib.sha256((args.input / "traces.csv").read_bytes()).hexdigest(),
                    matplotlib=matplotlib.__version__, numpy=np.__version__)
    (args.output / "report-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
