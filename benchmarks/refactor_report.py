"""Publish complete paired measurements and quality audits in both languages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import refactor


def validate(data):
    config = data["config"]
    if (
        not data.get("eligible_for_performance_claim")
        or config["quick"]
        or len(set(config["seeds"])) < 3
        or config["repeats"] < 3
    ):
        raise ValueError("Protocol/quick runs cannot establish performance acceptance")
    expected = {
        (case, storage, seed, repeat)
        for case in refactor.PRIMARY + refactor.SECONDARY
        for storage in ("dense", "csr")
        for seed in config["seeds"]
        for repeat in range(config["repeats"])
    }
    keys = ("case", "storage", "seed", "repeat")
    for label in ("before", "after"):
        observed = [tuple(row[k] for k in keys) for row in data[label]]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("Incomplete declared seeds/repeats/workloads")
    return refactor.compare(data["before"], data["after"])


def statistics(rows, field):
    # Treat each data seed equally, independently of its number of repeats.
    medians = [
        np.median([r[field] for r in rows if r["seed"] == seed])
        for seed in sorted({r["seed"] for r in rows})
    ]
    q1, median, q3 = np.quantile(medians, [0.25, 0.5, 0.75])
    return dict(median=float(median), q1=float(q1), q3=float(q3))


def summaries(data, comparison):
    output = []
    for item in comparison["cases"]:
        row = dict(item)
        for label in ("before", "after"):
            selected = [
                r
                for r in data[label]
                if r["case"] == item["case"] and r["storage"] == item["storage"]
            ]
            row[label + "_details"] = {
                field: statistics(selected, field)
                for field in (
                    "cpp_seconds",
                    "prediction_seconds",
                    "worker_peak_rss_mib",
                    "heldout_quality",
                )
            }
            row[label + "_maximum_stationarity"] = max(
                r["stationarity"] for r in selected
            )
            row[label + "_statuses"] = sorted({r["status"] for r in selected})
        output.append(row)
    return output


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def ms(values):
    return f"{values['median'] * 1000:.3f} [{values['q1'] * 1000:.3f}, {values['q3'] * 1000:.3f}]"


def render(data, comparison, summary, quality, raw, chinese):
    speed = comparison["primary_geometric_speedup"]
    status = (
        ("通过" if comparison["passed"] else "未达到")
        if chinese
        else ("PASS" if comparison["passed"] else "NOT MET")
    )
    heading = (
        "代码质量重构与性能验收"
        if chinese
        else "Runtime refactoring: quality and performance acceptance"
    )
    lines = [
        f"# {heading}",
        "",
        f"**2× gate: {status}; geometric mean full-fit speedup = {speed:.4f}×.**",
        "",
    ]
    lines += [
        "基线为原始 3374ad2 / 0.4.0，候选为重构后的 0.4.1。所有 18 个主场景均使用相同数据、目标函数和完整梯度精度，且已通过独立数值检查。SGD/Adam 的 4 个固定预算场景单独报告。"
        if chinese
        else "Baseline: original 3374ad2 / 0.4.0. Candidate: refactored 0.4.1. All 18 primary workloads share identical data, objectives and full-gradient accuracy, verified independently. Four fixed-budget SGD/Adam workloads are reported separately.",
        "",
        "汇总规则：先取每个种子内重复中位数，再跨种子取中位数及 IQR。加速比是基线/候选中位数之比，主指标为 18 项加速比的几何平均。完整 fit 包括输入复制、校验和初始化；不是只计最快的 C++ 内核片段。"
        if chinese
        else "Aggregation: repeat medians within each seed, then median/IQR across seeds. Speedup is the ratio of baseline/candidate medians; the primary gate is the geometric mean across all 18 ratios. Full fit includes copying, validation and initialization, rather than only the fastest native kernel segment.",
        "",
        "## 代码质量" if chinese else "## Code quality",
        "",
        table(
            ["Audit", "Before", "After", "CI budget"],
            [
                [
                    "Maximum C++ function complexity",
                    quality["before"]["max_cpp"],
                    quality["after"]["max_cpp"],
                    15,
                ],
                [
                    "Maximum Python runtime function complexity",
                    quality["before"]["max_python"],
                    quality["after"]["max_python"],
                    15,
                ],
            ],
        ),
        "",
        "数据契约、存储/数值内核、模型目标、求解器状态和绑定注册分别负责独立职责；Python 公开模块只负责导出，估计器直接依赖转换/校验模块。审查范围是现代运行时、公开头文件和绑定，历史实验与生成报告不计入该函数门禁。完整函数清单随原始数据保留。"
        if chinese
        else "Data contracts, storage/numeric kernels, response objectives, optimizer state and binding registration have separate responsibilities. The Python facade exports estimators; estimators depend directly on conversion/validation modules. The function gate covers the modern runtime, public headers and bindings, excluding historical experiments and generated reports. Full function inventories are retained.",
        "",
        "![Complexity audit](complexity.png)",
        "",
        "## 主场景：收敛到相同精度"
        if chinese
        else "## Primary workloads: matched convergence accuracy",
        "",
    ]
    primary = [r for r in summary if not r["secondary"]]
    secondary = [r for r in summary if r["secondary"]]
    lines += [
        table(
            [
                "Storage",
                "Model/solver",
                "Before fit ms [IQR]",
                "After fit ms [IQR]",
                "Speedup",
                "Max stationarity before/after",
            ],
            [
                [
                    r["storage"],
                    r["case"],
                    ms(r["before"]),
                    ms(r["after"]),
                    f"{r['speedup']:.3f}×",
                    f"{r['before_maximum_stationarity']:.2e} / {r['after_maximum_stationarity']:.2e}",
                ]
                for r in primary
            ],
        ),
        "",
        "![Full-fit speedup](speedup.png)",
        "",
        "## 随机算法：固定预算"
        if chinese
        else "## Stochastic algorithms: fixed budget",
        "",
        "SGD/Adam 使用相同种子、30 个 epoch、batch=256 和固定学习率。此处比较同预算的实际时间及结果一致性，不把预算耗尽称为收敛，也不计入 2× 收敛验收指标。"
        if chinese
        else "SGD/Adam share seeds, 30 epochs, batch=256 and constant learning rates. This track compares time and result consistency at fixed work budgets; budget exhaustion is not convergence and these ratios do not enter the 2× convergence gate.",
        "",
        table(
            [
                "Storage",
                "Model/solver",
                "Before ms [IQR]",
                "After ms [IQR]",
                "Speedup",
                "Before/after statuses",
            ],
            [
                [
                    r["storage"],
                    r["case"],
                    ms(r["before"]),
                    ms(r["after"]),
                    f"{r['speedup']:.3f}×",
                    f"{','.join(r['before_statuses'])} / {','.join(r['after_statuses'])}",
                ]
                for r in secondary
            ],
        ),
        "",
        "## 核心、预测与内存" if chinese else "## Core time, prediction and memory",
        "",
        "以下均为种子中位数。预测耗时含输入转换；RSS 为整个 worker 的 Linux VmHWM，包含解释器、数据生成、预热、NumPy 独立验算和数据，不能解释为求解器增量内存。"
        if chinese
        else "Values below are seed medians. Prediction includes input conversion. RSS is whole-worker Linux VmHWM, including interpreter, generated data, warm-up and independent NumPy checks; it is not incremental solver memory.",
        "",
        table(
            [
                "Storage",
                "Model/solver",
                "Core ms before/after",
                "Predict ms before/after",
                "RSS MiB before/after",
            ],
            [
                [
                    r["storage"],
                    r["case"],
                    " / ".join(
                        f"{r[label + '_details']['cpp_seconds']['median'] * 1000:.3f}"
                        for label in ("before", "after")
                    ),
                    " / ".join(
                        f"{r[label + '_details']['prediction_seconds']['median'] * 1000:.3f}"
                        for label in ("before", "after")
                    ),
                    " / ".join(
                        f"{r[label + '_details']['worker_peak_rss_mib']['median']:.1f}"
                        for label in ("before", "after")
                    ),
                ]
                for r in summary
            ],
        ),
        "",
        "## 测试集效果" if chinese else "## Held-out quality",
        "",
        "以下为各种子中位数；二项使用 log loss，高斯/泊松使用 MSE。每一对重复的效果差均单独检查，不能通过平均抵消不合格结果。"
        if chinese
        else "Seed medians below use Binomial log loss or Gaussian/Poisson MSE. Every paired repeat is checked independently; aggregation cannot hide a quality failure.",
        "",
        table(
            ["Storage", "Model/solver", "Quality before", "Quality after"],
            [
                [
                    r["storage"],
                    r["case"],
                    f"{r['before_details']['heldout_quality']['median']:.8f}",
                    f"{r['after_details']['heldout_quality']['median']:.8f}",
                ]
                for r in summary
            ],
        ),
        "",
        "## 协议、来源与边界" if chinese else "## Protocol, provenance and limits",
        "",
    ]
    machine = data["machine"]
    lines += [
        f"- CPU: {machine['cpu']}; affinity: {machine['affinity']}; threads: {machine['threads']}.",
        f"- Python {machine['python']}; NumPy {machine['numpy']}; SciPy {machine['scipy']}; libc {machine['libc']}.",
        f"- Measured: {machine['measured_at']}; seeds: {data['config']['seeds']}; repeats: {data['config']['repeats']}.",
        "- Dense: 24,000 × 96; CSR: 24,000 × 2,000, density 1%; training 80%, held-out 20%.",
        "- L2=.1; linear proximal solvers additionally use L1=.01. Unpenalized intercept; primary tolerance=1e-6.",
        "- SGD learning rate=.03; Adam=.003. Primary budgets: logistic/GLM 300, linear 1,000; actual work retained.",
        f"- Baseline runtime: {data['artifacts']['before']['runtime']}; candidate: {data['artifacts']['after']['runtime']}.",
        "",
        "每次主拟合必须同时满足收敛状态和独立重算的梯度/近端映射范数 ≤1.00001e−6。配对目标差 ≤1e−9、系数差 ≤3e−5（另有1e−5相对项）、测试指标差 ≤2e−5。指标为二项 log loss，或高斯/泊松 MSE；这些阈值没有因优化而放宽。"
        if chinese
        else "Every primary fit must both report convergence and satisfy an independently recomputed gradient/proximal mapping norm ≤1.00001e−6. Paired objectives agree within 1e−9, coefficients within 3e−5 absolute plus 1e−5 relative tolerance, and held-out metrics within 2e−5. Quality is Binomial log loss or Gaussian/Poisson MSE. Optimization does not relax these thresholds.",
        "",
        "二进制、源文件和数据校验和、实际编译配置均保留。两边使用相同 Release 编译器和基础优化级别，候选通过运行时检测使用可选 SIMD/系统向量数学，未启用 fast-math 或额外线程。部分求解器减少了重复扫描/内层工作，因此不是相同指令数的微基准。"
        if chinese
        else "Native binaries, sources, datasets and actual build configurations are hashed. Both artifacts use the same Release compiler and base optimization level. The candidate runtime-detects optional SIMD/system vector math without fast-math or extra threads. Some solvers avoid scans or unnecessary inner work, so this is not an equal-instruction microbenchmark.",
        "",
        "结果来自固定规模的合成数据和当前 CPU。虽固定了单核与线程数，仍未隔离 CPU 频率和宿主机负载；三个种子不能覆盖所有数据条件。几何平均达标不代表每个场景都达标；可选后端不可用时应重新测量。标量与向量数学可能有末位舍入差异。"
        if chinese
        else "These measurements use fixed-size synthetic data on the recorded CPU. CPU frequency and host contention remain uncontrolled despite core/thread pinning; three seeds do not cover every data condition. A geometric-mean pass does not imply every workload passes. Remeasure when optional backends are unavailable. Scalar/vector math can differ in final-bit rounding.",
        "",
        f"[Raw measurements and histories]({raw}/results.json) · [Baseline audit]({raw}/quality-before.json) · [Candidate audit]({raw}/quality-after.json)",
        "",
        "Design, exact commands and ownership/numerical contracts: [refactoring guide](../../../docs/refactoring.md). "
        "Content hashes for this report and its generator are in `report-manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def figures(comparison, quality, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logitlab-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/logitlab-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "logitlab-refactor-v1"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    primary = [r for r in comparison["cases"] if not r["secondary"]]
    for axis, storage in zip(axes, ("dense", "csr")):
        rows = [r for r in primary if r["storage"] == storage]
        axis.barh(
            [r["case"] for r in rows], [r["speedup"] for r in rows], color="#4477AA"
        )
        axis.axvline(1, color="#888888", linewidth=1)
        axis.axvline(2, color="#CC6677", linestyle="--", label="2× gate")
        axis.set(title=storage, xlabel="Full-fit speedup vs 0.4.0")
        axis.grid(axis="x", alpha=0.2)
        axis.legend()
    fig.suptitle("All 18 matched-convergence workloads")
    fig.tight_layout()
    save_figure(fig, output / "speedup")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    axis.bar(
        x - 0.18,
        [quality["before"]["max_cpp"], quality["before"]["max_python"]],
        width=0.36,
        label="0.4.0",
        color="#999999",
    )
    axis.bar(
        x + 0.18,
        [quality["after"]["max_cpp"], quality["after"]["max_python"]],
        width=0.36,
        label="0.4.1",
        color="#4477AA",
    )
    axis.axhline(15, color="#CC6677", linestyle="--", label="CI budget")
    axis.set(
        xticks=x,
        xticklabels=["Modern C++ / bindings", "Python runtime"],
        ylabel="Maximum function complexity",
    )
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output / "complexity")
    plt.close(fig)


def save_figure(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    svg = stem.with_suffix(".svg")
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.input / "results.json").read_text())
    comparison = validate(data)
    quality = {
        label: json.loads((args.input / f"quality-{label}.json").read_text())
        for label in ("before", "after")
    }
    summary = summaries(data, comparison)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = os.path.relpath(args.input, args.output)
    for chinese, filename in ((False, "refactor.en.md"), (True, "refactor.zh-CN.md")):
        report = render(data, comparison, summary, quality, raw, chinese)
        attempts = args.input / "development-attempts.md"
        if attempts.exists():
            label = (
                "包含未达标版本的开发实验记录"
                if chinese
                else "Development experiments, including failed gates"
            )
            report += f"\n[{label}]({raw}/development-attempts.md)\n"
        (args.output / filename).write_text(report)
    (args.output / "summary.json").write_text(
        json.dumps(dict(comparison=comparison, summary=summary), indent=2) + "\n"
    )
    figures(comparison, quality, args.output)
    inputs = [
        args.input / name
        for name in ("results.json", "quality-before.json", "quality-after.json")
    ]
    inputs += [Path(__file__), Path(refactor.__file__)]
    inputs += sorted(args.input.glob("attempt-*.results.json.gz"))
    if (args.input / "development-attempts.md").exists():
        inputs.append(args.input / "development-attempts.md")
    files = (
        inputs
        + list(args.output.glob("*.md"))
        + list(args.output.glob("*.png"))
        + list(args.output.glob("*.svg"))
        + [args.output / "summary.json"]
    )
    manifest = {
        os.path.relpath(p, args.output): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (args.output / "report-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
