"""Generate bilingual regression reports, preserving tuning and convergence limits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.linear_suite import MODELS, SOLVERS, models
from benchmarks.report import fmt, stats, table, value

COLORS = dict(zip((*MODELS, "wls", "nonnegative", "box-fixed"),
                 ("#4477AA", "#EE6677", "#228833", "#AA8822", "#AA3377", "#66AABB", "#777777")))
DEFAULT_SOLVER = lambda model: "cg" if model in ("ols", "ridge", "wls") else "fista"


def validate_complete(data):
    c = data["config"]
    expected, references = set(), set()
    for case, seed, scaling in product(c["cases"], c["seeds"], ("raw", "rms")):
        for model in models(case):
            references.add((case, seed, scaling, model))
            solvers = SOLVERS if model in ("ols", "ridge", "wls") else ("proximal-gd", "fista")
            expected.update((case, seed, scaling, model, solver, repeat)
                            for solver, repeat in product(solvers, range(c["repeats"])))
    seen = Counter(tuple(row[key] for key in ("case", "seed", "scaling", "model", "solver", "repeat"))
                   for row in data["results"])
    if set(seen) != expected or any(count != 1 for count in seen.values()):
        raise ValueError(f"Incomplete experiment: {len(expected-set(seen))} missing slots, or duplicate measurements")
    refs = [tuple(r[key] for key in ("case", "seed", "scaling", "model")) for r in data["selections"]]
    if set(refs) != references or len(refs) != len(references):
        raise ValueError("Incomplete model-selection evidence")


def summarize(data):
    groups = defaultdict(lambda: defaultdict(list))
    for row in data["results"]:
        groups[tuple(row[k] for k in ("case", "scaling", "model", "solver"))][row["seed"]].append(row)
    summaries = []
    for key, seeds in groups.items():
        s = dict(zip(("case", "scaling", "model", "solver"), key))
        s.update(seeds=len(seeds), repeats=sum(len(r) for r in seeds.values()),
                 maximum_constraint_violation=max(r["constraint_violation"] for runs in seeds.values() for r in runs),
                 converged_seeds=sum(all(r["converged"] for r in runs) for runs in seeds.values()),
                 target_seeds=sum(all(r["objective_gap"] is not None and r["objective_gap"] <= data["config"]["target_gap"]
                                      for r in runs) for runs in seeds.values()),
                 ever_target_seeds=sum(any(r.get("target") is not None for r in runs) for runs in seeds.values()),
                 reference_valid_seeds=sum(all(r["reference_valid"] for r in runs) for runs in seeds.values()),
                 statuses=dict(Counter(runs[0]["status"] for runs in seeds.values())),
                 selected_alphas=sorted({r["alpha"] for runs in seeds.values() for r in runs}))
        for metric in ("fit_seconds", "cpp_seconds", "iterations", "objective_gap", "stationarity_norm",
                       "constraint_violation", "nonzero_coefficients", "coefficient_rmse", "support_f1",
                       "prediction_rows_per_second", "worker_peak_rss_mib", "selection_seconds",
                       "hessian_vector_products", "parameter_updates"):
            per_seed = [stats([r.get(metric) for r in runs]) for runs in seeds.values()]
            s[metric] = stats([a["median"] for a in per_seed if a])
        for group, metric, name in (("test", "rmse", "test_rmse"), ("test", "mae", "test_mae"),
                                    ("test", "r2", "test_r2"), ("test_weighted", "rmse", "weighted_rmse"),
                                    ("reference_test", "rmse", "reference_rmse")):
            s[name] = stats([np.median([r[group][metric] for r in runs]) for runs in seeds.values()])
        s["total_passes"] = stats([np.median([r["row_visits"]/r["train_rows"] for r in runs]) for runs in seeds.values()])
        target = [stats([r["target"]["seconds"] for r in runs if r.get("target") is not None]) for runs in seeds.values()]
        s["target_seconds"] = stats([v["median"] for v in target if v])
        summaries.append(s)
    return summaries


def figures(data, summaries, source, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logitlab-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/logitlab-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "logitlab-linear"})
    output.mkdir(parents=True, exist_ok=True)
    cases = data["config"]["cases"]
    lookup = {(s["case"], s["scaling"], s["model"], s["solver"]): s for s in summaries}
    def save(fig, name):
        for extension in ("png", "svg"):
            path = output/f"{name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else {"Software": "LogitLab"}
            fig.savefig(path, dpi=160, bbox_inches="tight", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, case in zip(axes.flat, cases):
        for j, model in enumerate(models(case)):
            s = lookup[(case, "rms", model, DEFAULT_SOLVER(model))]
            loss = s["test_rmse"]; median = loss["median"]
            ax.bar(j, median, color=COLORS[model], alpha=.75)
            ax.errorbar(j, median, yerr=[[median-loss["q1"]], [loss["q3"]-median]], color="black", capsize=3)
            ax.scatter(j, value(s, "reference_rmse"), color="black", s=20, zorder=3)
        ax.set(xticks=range(len(models(case))), xticklabels=models(case), title=case, ylabel="Test RMSE")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.15)
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    fig.suptitle("RMS pipelines: bars = measured RMSE (seed IQR); black dots = independent reference")
    fig.tight_layout()
    save(fig, "model-quality")

    if "heteroscedastic" in cases:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric, label in zip(axes, ("test_rmse", "weighted_rmse"),
                                     ("Unweighted test RMSE", "Precision-weighted test RMSE")):
            for j, model in enumerate(models("heteroscedastic")):
                s = lookup[("heteroscedastic", "rms", model, DEFAULT_SOLVER(model))]
                score = s[metric]; median = score["median"]
                ax.bar(j, median, color=COLORS[model], alpha=.75)
                ax.errorbar(j, median, yerr=[[median-score["q1"]], [score["q3"]-median]],
                            color="black", capsize=3)
            ax.set(xticks=range(len(models("heteroscedastic"))),
                   xticklabels=models("heteroscedastic"), ylabel=label)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.15)
        fig.suptitle("Known heteroscedasticity: WLS and two distinct predictive risks (seed IQR)")
        fig.tight_layout()
        save(fig, "weighted-quality")

    markers = {"cg": "o", "proximal-gd": "s", "fista": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for ax, case in zip(axes.flat, cases):
        for s in summaries:
            if s["case"] == case and s["scaling"] == "rms":
                x, y = value(s, "fit_seconds")*1000, value(s, "test_rmse")
                duration, error = s["fit_seconds"], s["test_rmse"]
                ax.errorbar(x, y,
                            xerr=[[x-duration["q1"]*1000], [duration["q3"]*1000-x]],
                            yerr=[[y-error["q1"]], [error["q3"]-y]],
                            fmt="none", color=COLORS[s["model"]], alpha=.25, lw=.8, capsize=2)
                ax.scatter(value(s, "fit_seconds")*1000, value(s, "test_rmse"), s=50,
                           color=COLORS[s["model"]], marker=markers[s["solver"]])
        ax.set(xscale="log", title=case, xlabel="Full fit median (ms)", ylabel="Test RMSE")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.), numticks=8))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda number, _: f"{number:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    handles = [Line2D([], [], color=color, marker="o", ls="", label=model) for model, color in COLORS.items()]
    handles += [Line2D([], [], color="black", marker=marker, ls="", label=solver) for solver, marker in markers.items()]
    fig.legend(handles=handles, ncol=5, loc="lower center")
    fig.suptitle("RMS quality and cost: same selected penalty across solvers; bars = seed IQR")
    fig.tight_layout(rect=(0, .08, 1, .96))
    save(fig, "quality-performance")

    first_seed = data["config"]["seeds"][0]
    references = {(r["case"], r["scaling"]): r["reference"] for r in data["selections"]
                  if r["seed"] == first_seed and r["model"] == "ols"}
    traces = defaultdict(list)
    with (source/"traces.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["seed"]) == first_seed and row["model"] == "ols":
                traces[(row["case"], row["scaling"], row["solver"])].append(row)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, case in zip(axes.flat, cases):
        for solver, color in zip(SOLVERS, ("#4477AA", "#EE6677", "#228833")):
            for scaling in ("raw", "rms"):
                ref = references[(case, scaling)]
                trace = traces[(case, scaling, solver)]
                ax.plot([max(1e-7, float(h["seconds"])) for h in trace],
                        [max(1e-12, (float(h["loss"])-ref["objective"])/(1+abs(ref["objective"]))) for h in trace],
                        color=color, ls="--" if scaling == "raw" else "-", label=f"{solver}/{scaling}")
        ax.axhline(data["config"]["target_gap"], color="#777777", ls=":")
        ax.set(xscale="log", yscale="log", title=case, xlabel="C++ time (s)", ylabel="Normalized objective gap")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"OLS convergence, seed={first_seed}, repeat=0; dashed raw / solid RMS")
    fig.tight_layout()
    save(fig, "convergence")


def render(data, summaries, zh, link):
    t = lambda en, cn: cn if zh else en
    config, machine = data["config"], data["machine"]
    refs = data["selections"]
    lookup = {(s["case"], s["scaling"], s["model"], s["solver"]): s for s in summaries}
    lines = [t("# Linear regression: models, constraints and optimization", "# 线性回归：模型、参数约束与优化评测"), "",
             t("[中文版](linear.zh-CN.md)", "[English](linear.en.md)"), "",
             t("OLS, Ridge, Lasso and Elastic Net use the same C++ dense/CSR least-squares kernels. "
               "WLS isolates known precision weights; nonnegative/box-fixed fits test declared domain priors. "
               "Model regularization and optimizer convergence are separate axes.",
               "OLS、Ridge、Lasso、Elastic Net 使用同一组 C++ 稠密/CSR 最小二乘内核。WLS 检验已知精度权重，非负/边界固定模型检验预先声明的领域先验。"
               "正则化模型差异与求解器收敛差异分别评估。"), "",
             t("## Main observations", "## 主要观察"), "",
             *observations(summaries, zh), "",
             *( [t("**Quick protocol check only; not performance evidence.**",
                    "**当前是快速协议检查，不能用于性能结论。**"), ""] if config["quick"] else []),
             t("## Evidence and reproduction", "## 评测规模与复现"), "",
             t(f"{len(data['results'])} measured fits; {len(config['seeds'])} seeds × {config['repeats']} repetitions. "
               f"{sum(s['reference']['verified'] for s in refs)}/{len(refs)} selected references verified.",
               f"共 {len(data['results'])} 次实测训练；{len(config['seeds'])} 个种子，每个配置 {config['repeats']} 次重复。"
               f"选中参考解 {sum(s['reference']['verified'] for s in refs)}/{len(refs)} 通过验证。"), "",
             f"[Raw JSON]({link}/results.json) · [Metrics CSV]({link}/metrics.csv) · [Traces CSV]({link}/traces.csv) · "
             "[Summary](summary.json) · [Report manifest](report-manifest.json)", "",
             "~~~bash", "python -m pip install '.[test,benchmark]'",
             "python benchmarks/linear_suite.py --cases " + " ".join(config["cases"]) +
             " --seeds " + " ".join(map(str, config["seeds"])) +
             f" --max-iter {config['max_iter']} --repeats {config['repeats']} --target-gap {config['target_gap']}" +
             (" --quick" if config["quick"] else "") + " --output build/linear-benchmark",
             "python benchmarks/linear_report.py --input build/linear-benchmark --output build/linear-report", "~~~", "",
             f"- CPU: {machine['cpu']}; affinity: {machine['cpu_affinity']}.",
             f"- Python {machine['python']}; NumPy {machine['numpy']}; SciPy {machine['scipy']}; "
             f"scikit-learn {machine['sklearn']}; LogitLab {machine['logitlab']}.",
             t("- Core/BLAS one thread; source/binary/data/split/scaler hashes are in raw JSON. "
               "Actual build flags are recorded when --build-cache is supplied. CPU frequency/background load are not isolated.",
               "- C++/BLAS 单线程；原始 JSON 保存源码、二进制、数据、划分和 scaler 哈希。"
               "提供 --build-cache 时记录实际构建参数；未隔离 CPU 频率和后台负载。"), "",
             t("## Protocol", "## 协议"), "",
             t("Random seeded 70/15/15 train/validation/test splits; identical splits across models/solvers. "
               "RMS is learned from training rows, never centers CSR. No test labels enter penalty selection. "
               "Ridge/Lasso/Elastic Net search α={0.0001,0.001,0.01,0.1} using independently solved training "
               "objectives and validation MSE. Elastic Net fixes l1_ratio=0.5. Other models have no penalty search. "
               "The same selected objective is then given to every applicable C++ solver.",
               "按种子随机划分训练/验证/测试集约 70/15/15，所有模型与求解器共享划分。RMS 仅在训练集拟合，CSR 不减均值。"
               "测试标签不参与选参。Ridge/Lasso/Elastic Net 用独立参考实现拟合训练集，在 α={0.0001,0.001,0.01,0.1} 中按验证 MSE 选择；"
               "Elastic Net 固定 l1_ratio=0.5。其他模型没有正则搜索。随后所有适用 C++ 求解器使用相同选中目标。"), "",
             t("Objective: sum(qᵢ(yᵢ−xᵢw−b)²)/2 + l1||w||₁ + l2||w||²/2, q sums to 1. "
               "Intercept is unpenalized. Ridge l2=α; Lasso l1=α; Elastic Net l1=l2=α/2. "
               "Unlike sklearn Ridge's sum-loss α, this Ridge coefficient is a mean-loss penalty. "
               "Scaling changes isotropic regularization: α is reselected per preprocessing pipeline. "
               "OLS raw/RMS preserves attainable predictions; constraints are transformed back to the same original units.",
               "目标：sum(qᵢ(yᵢ−xᵢw−b)²)/2 + l1||w||₁ + l2||w||²/2，q 总和为 1，截距不惩罚。"
               "Ridge 的 l2=α，Lasso 的 l1=α，Elastic Net 的 l1=l2=α/2。这里 Ridge 使用均值损失尺度，与 sklearn Ridge 的求和损失 α 不同。"
               "缩放改变各向同性正则含义，因此每条预处理流程重新选 α。OLS 原始/RMS 的可表达预测不变，约束按同一原始单位变换。"), "",
             t("References: independently formulated, column-preconditioned NumPy SVD / SciPy sparse LSMR "
               "for OLS/Ridge/WLS; sklearn cyclic coordinate descent for L1; scipy bounded least squares "
               "with fixed coordinates eliminated for constraints. Preconditioning transforms penalties/bounds too. "
               "Reference validity requires unit-step proximal mapping norm ≤1e-5 and violation ≤1e-10.",
               "参考实现：独立构造并做列预条件的 NumPy SVD / SciPy 稀疏 LSMR 验证 OLS/Ridge/WLS；"
               "sklearn 循环坐标下降验证 L1；SciPy 有界最小二乘消去固定坐标后验证约束。预条件同步变换正则和边界。"
               "参考有效条件为单位步长近端映射范数 ≤1e-5、约束违反 ≤1e-10。"), "",
             t(f"Budget {config['max_iter']} updates, tolerance 1e-7. CG is Jacobi-preconditioned and matrix-free; "
               "it supports smooth unconstrained objectives. Proximal GD and monotone restarted FISTA support all "
               "penalties/bounds. Stop criterion is the unit-step proximal gradient-mapping norm. Target: "
               f"(F−Fref)/(1+|Fref|) ≤ {config['target_gap']:g}; convergence and target attainment are distinct.",
               f"预算 {config['max_iter']} 次更新，容差 1e-7。CG 使用 Jacobi 预条件，不形成 XᵀX，仅适用于光滑无约束问题；"
               "近端 GD 与单调重启 FISTA 支持全部正则和边界。停止条件为单位步长近端梯度映射范数；"
               f"达标条件为 (F−Fref)/(1+|Fref|) ≤ {config['target_gap']:g}，与算法的收敛状态分别记录。"), "",
             t("Fit wall time includes Python conversion/constraints/weight setup and core validation. C++ history "
               "excludes setup. Small fit/prediction warm-ups; seven prediction calls per repeat. All methods share "
               "prediction kernels. RSS is post-exec worker VmHWM, including interpreter, data and all repetitions, "
               "not incremental solver memory. Selection cost is independent-reference fit/verification cost, "
               "excluding one-time imports, validation scoring and test scoring; it is not C++ tuning cost.",
               "fit 墙钟时间含 Python 转换、约束/权重设置及核心校验；C++ history 排除准备成本。计时前做小规模训练与预测预热，"
               "每次重复预测七次，所有方法共用预测内核。RSS 为 exec 后 worker VmHWM，含解释器、数据和全部重复，不是算法增量内存。"
               "选参成本为独立参考拟合及验证成本，排除首次导入、验证集评分和测试评分；它不是 C++ 调参成本。"), "",
             t("First aggregate repeats inside each seed, then median [Q1,Q3] across seeds. Repeated timing "
               "is not an independent dataset; three seeds are descriptive, not significance tests. Target times "
               "are conditional on reaching; a dash never means zero or the budget cutoff. Full traces retain repeat=0; "
               "all repeats retain timing/metrics/target counters. Missing configurations block report generation.",
               "先汇总同一种子内重复，再对种子中位数统计 median [Q1,Q3]；重复计时不是独立数据，三个种子只支持描述性结论。"
               "首次达标时间仅统计曾达标种子，破折号不表示 0 或预算终点。完整轨迹保留 repeat=0；全部重复保留计时、指标与达标计数。"
               "实验缺项时禁止生成完整报告。"), ""]
    scenarios = []
    descriptions = {
        "dense_control": t("Independent Gaussian features, sparse signed truth.", "独立高斯特征、稀疏有符号系数。"),
        "dense_correlated": t("Groups of four correlated columns, noise 0.05; tests shrinkage and ill conditioning.", "每四列强相关、噪声 0.05；研究收缩与病态条件。"),
        "dense_multiscale": t("Units differ by factors 0.01–100; coefficient transformation preserves the signal.", "量级倍率 0.01–100，系数同步变换以保持信号。"),
        "sparse_wide": t("CSR density 0.01 (quick 0.15), sparse truth, noisy high-dimensional estimation.", "CSR 密度 0.01（quick 为 0.15），稀疏真系数、高维噪声估计。"),
        "heteroscedastic": t("Known sigma=0.15+3|x0|; WLS uses precision=1/sigma², others are unweighted.", "已知 sigma=0.15+3|x0|；WLS 使用 1/sigma²，其他模型不加权。"),
        "positive_prior": t("True coefficients in [0,0.5], w0=0.2 declared before responses. Box priors are oracle domain knowledge.", "真实系数在 [0,0.5]，w0=0.2 在响应生成前声明；边界为已知领域先验。")}
    for case in config["cases"]:
        s = next(s for s in refs if s["case"] == case)
        scenarios.append([case, f"{s['rows']} × {s['cols']}", s["storage"], descriptions[case]])
    lines += [table([t("Scenario", "场景"), t("Shape", "规模"), t("Storage", "存储"), t("Purpose", "目的")], scenarios), "",
              t("All datasets are synthetic. Priors and noise precisions are known by construction, not "
                "estimated from test responses. Results do not establish industrial-scale or real-domain accuracy.",
                "全部为合成数据；先验和噪声精度由生成机制给定，不从测试响应估计。结果不能证明工业规模性能或真实领域精度。"), "",
              t("## Default pipelines: quality and reference agreement", "## 默认流程：效果与参考解"), "",
              t("Predeclared solver: CG for OLS/Ridge/WLS, FISTA for L1/constraints. This choice does not inspect "
                "test performance. Black dots are reference predictions, not a lower bound on test error.",
                "预先规定 OLS/Ridge/WLS 使用 CG，L1/约束使用 FISTA，不根据测试效果选择求解器。黑点为参考预测，不是测试误差下界。"), "",
              "![Model quality](figures/model-quality.png)", ""]
    overview = [s for s in summaries if s["scaling"] == "rms" and s["solver"] == DEFAULT_SOLVER(s["model"])]
    headers = (["场景", "模型", "求解器", "测试 RMSE [Q1,Q3]", "参考 RMSE", "R²", "非零系数", "Support F1", "训练 ms"]
               if zh else ["Case", "Model", "Solver", "Test RMSE [Q1,Q3]", "Reference RMSE", "R²", "Nonzero", "Support F1", "Fit ms"])
    lines += [table(headers, [[s["case"], s["model"], s["solver"], fmt(s, "test_rmse", interval=True),
                              fmt(s, "reference_rmse"), fmt(s, "test_r2"), fmt(s, "nonzero_coefficients", digits=0),
                              fmt(s, "support_f1"), fmt(s, "fit_seconds", 1000, 2)] for s in overview]), "",
              t("Support uses |original-coordinate coefficient|>1e-6; intercept is excluded. Coefficient "
                "RMSE and support F1 use synthetic truth and are evaluation-only, never tuning criteria.",
                "Support 使用 |原始坐标系数|>1e-6，不含截距。系数 RMSE 和 Support F1 使用合成真值，仅作评估，不参与选参。"), "",
              t("Violation is measured against each model's declared constraints. For an unconstrained model, "
                "zero violation is vacuous and does not imply that it satisfies the physical prior.",
                "约束违反量针对各模型自己声明的约束。无约束模型的违反量为 0 是空集性质，不表示它满足领域先验。"), "",
              *([t("### Known heteroscedasticity", "### 已知异方差"), "",
                 "![Weighted and unweighted risks](figures/weighted-quality.png)", "",
                 t("The two panels evaluate different risks. Precision-weighted error emphasizes lower-noise "
                   "observations. WLS need not improve unweighted error in every split.",
                   "两幅图评估不同风险：精度加权误差更重视低噪声观测，WLS 不保证每次划分的未加权误差都改善。"), ""]
                if "heteroscedastic" in config["cases"] else []),
              t("## Optimizer cost and scaling", "## 求解成本与 scaling"), "",
              "![Quality and performance](figures/quality-performance.png)", "",
              "![OLS convergence](figures/convergence.png)", "",
              t("Convergence plots show the first seed/repeat, not averages; floor 1e-12 is for plotting only. "
                "Equal update limits do not mean equal work. Unreached fits must not be ranked as faster convergence.",
                "收敛图为首种子、首次重复的示例，不是平均曲线；1e-12 下限仅用于绘图。相同更新上限不代表相同计算量。"
                "未达标但耗时短的结果不能宣称收敛更快。"), ""]
    for case in config["cases"]:
        rows = [s for s in summaries if s["case"] == case]
        rows.sort(key=lambda s: (s["scaling"], models(case).index(s["model"]), SOLVERS.index(s["solver"])))
        quality_headers = (["缩放", "模型", "求解器", "选中 α", "测试 RMSE", "加权 RMSE", "MAE", "R²", "系数 RMSE", "最大约束违反"]
                           if zh else ["Scale", "Model", "Solver", "Selected α", "Test RMSE", "Weighted RMSE", "MAE", "R²", "Coef RMSE", "Max violation"])
        lines += [f"### {case}", "", table(quality_headers, [
            [s["scaling"], s["model"], s["solver"], "/".join(f"{a:g}" for a in s["selected_alphas"]),
             fmt(s, "test_rmse"), fmt(s, "weighted_rmse"), fmt(s, "test_mae"), fmt(s, "test_r2"),
             fmt(s, "coefficient_rmse"), f"{s['maximum_constraint_violation']:.1e}"] for s in rows]), ""]
        cost_headers = (["缩放", "模型", "求解器", "训练 ms [Q1,Q3]", "归一化目标差", "最终达标", "曾达标", "首次达标 C++ ms",
                         "数据遍历", "预测 百万行/s", "进程峰值 MiB", "参考选参 ms"]
                        if zh else ["Scale", "Model", "Solver", "Fit ms [Q1,Q3]", "Normalized gap", "Final target", "Ever target",
                                    "Target C++ ms", "Data passes", "Predict M rows/s", "RSS MiB", "Reference selection ms"])
        lines += [table(cost_headers, [
            [s["scaling"], s["model"], s["solver"], fmt(s, "fit_seconds", 1000, 2, True),
             f"{value(s, 'objective_gap'):.2e}" if s["objective_gap"] else "—", f"{s['target_seeds']}/{s['seeds']}",
             f"{s['ever_target_seeds']}/{s['seeds']}", fmt(s, "target_seconds", 1000, 2),
             fmt(s, "total_passes", digits=1), fmt(s, "prediction_rows_per_second", 1e-6),
             fmt(s, "worker_peak_rss_mib", digits=1), fmt(s, "selection_seconds", 1000, 2)] for s in rows]), ""]
    lines += [t("## Interpretation and limits", "## 结果解读与局限"), ""]
    for model in MODELS:
        a = lookup.get(("dense_multiscale", "raw", model, DEFAULT_SOLVER(model)))
        b = lookup.get(("dense_multiscale", "rms", model, DEFAULT_SOLVER(model)))
        if a and b:
            lines.append(t(f"- Multiscale {model}/{a['solver']}: final target raw {a['target_seeds']}/{a['seeds']} "
                           f"→ RMS {b['target_seeds']}/{b['seeds']}; test RMSE {fmt(a, 'test_rmse')} → {fmt(b, 'test_rmse')}.",
                           f"- 多尺度 {model}/{a['solver']}：最终达标 原始 {a['target_seeds']}/{a['seeds']} → "
                           f"RMS {b['target_seeds']}/{b['seeds']}；测试 RMSE {fmt(a, 'test_rmse')} → {fmt(b, 'test_rmse')}。"))
    for case, baseline, other in (("sparse_wide", "ols", "lasso"), ("heteroscedastic", "ols", "wls"),
                                  ("positive_prior", "ols", "box-fixed")):
        a = lookup.get((case, "rms", baseline, DEFAULT_SOLVER(baseline)))
        b = lookup.get((case, "rms", other, DEFAULT_SOLVER(other)))
        if a and b:
            lines.append(t(f"- {case}: {baseline} RMSE {fmt(a, 'test_rmse')} vs {other} {fmt(b, 'test_rmse')}; "
                           f"weighted RMSE {fmt(a, 'weighted_rmse')} vs {fmt(b, 'weighted_rmse')}.",
                           f"- {case}：{baseline} RMSE {fmt(a, 'test_rmse')}，{other} {fmt(b, 'test_rmse')}；"
                           f"加权 RMSE 分别为 {fmt(a, 'weighted_rmse')}、{fmt(b, 'weighted_rmse')}。"))
    lines += ["", t("CG preconditioning can already correct column magnitudes, so scaling does not have the same "
                    "effect on every solver. RMS does not whiten correlation. L1 encourages zeros, Ridge shrinks all "
                    "coefficients, and valid priors reduce the feasible set; misspecified constraints can introduce bias. "
                    "Hyperparameter grids, early stopping, three seeds and synthetic mechanisms limit winner claims.",
                    "CG 预条件本身可校正列量级，因此 scaling 对不同求解器影响不同；RMS 不白化相关性。L1 促进零系数，Ridge 收缩全部系数，"
                    "正确先验缩小可行域，错误约束会引入偏差。有限参数网格、提前停止、三个种子和合成机制均限制优胜排名结论。"), "",
              t("Data passes count row visits including CG diagonal preconditioning, HVPs and all objective/gradient "
                "line-search evaluations. This is not FLOPs; dense parameter work and sparse row work differ. "
                "Common OLS analysis is tested separately against SciPy/independent covariance formulas: "
                "classical/HC3 uncertainty is restricted to converged, full-rank, unpenalized unconstrained dense fits.",
                "数据遍历计入 CG 对角预条件、HVP 和全部目标/梯度线搜索评估；这不是 FLOPs，稠密参数和稀疏行操作成本不同。"
                "OLS 分析另用 SciPy/独立协方差公式测试：classical/HC3 只适用于收敛、满秩、无正则、无约束的稠密拟合。"), ""]
    return "\n".join(lines)


def observations(summaries, zh):
    t = lambda en, cn: cn if zh else en
    lookup = {(s["case"], s["scaling"], s["model"], s["solver"]): s for s in summaries}
    output = []
    for case, a_name, b_name in (("sparse_wide", "ols", "lasso"),
                                ("heteroscedastic", "ols", "wls"),
                                ("positive_prior", "ols", "box-fixed")):
        a = lookup.get((case, "rms", a_name, DEFAULT_SOLVER(a_name)))
        b = lookup.get((case, "rms", b_name, DEFAULT_SOLVER(b_name)))
        if a and b:
            output.append(t(
                f"- **{case}:** {a_name} test RMSE {fmt(a, 'test_rmse')} vs {b_name} {fmt(b, 'test_rmse')}; "
                f"weighted RMSE {fmt(a, 'weighted_rmse')} vs {fmt(b, 'weighted_rmse')}.",
                f"- **{case}：**{a_name} 测试 RMSE {fmt(a, 'test_rmse')}，{b_name} {fmt(b, 'test_rmse')}；"
                f"加权 RMSE 分别为 {fmt(a, 'weighted_rmse')}、{fmt(b, 'weighted_rmse')}。"))
    a = lookup.get(("dense_multiscale", "raw", "ols", "fista"))
    b = lookup.get(("dense_multiscale", "rms", "ols", "fista"))
    if a and b:
        output.append(t(
            f"- **Scaling:** multiscale OLS/FISTA final-target seeds raw {a['target_seeds']}/{a['seeds']} "
            f"→ RMS {b['target_seeds']}/{b['seeds']}. The CG preconditioner is evaluated separately.",
            f"- **Scaling：**多尺度 OLS/FISTA 最终达标 原始 {a['target_seeds']}/{a['seeds']} → "
            f"RMS {b['target_seeds']}/{b['seeds']}。CG 自带预条件的表现单独比较。"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.input/"results.json").read_text())
    validate_complete(data)
    summaries = summarize(data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    figures(data, summaries, args.input, args.output/"figures")
    link = Path(os.path.relpath(args.input.resolve(), args.output.resolve())).as_posix()
    for lang in ("en", "zh-CN"):
        (args.output/f"linear.{lang}.md").write_text(render(data, summaries, lang == "zh-CN", link))
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        "report_source_sha256": Path(__file__), "report_helpers_sha256": Path(__file__).with_name("report.py"),
        "results_sha256": args.input/"results.json", "traces_sha256": args.input/"traces.csv"}.items()}
    (args.output/"report-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
