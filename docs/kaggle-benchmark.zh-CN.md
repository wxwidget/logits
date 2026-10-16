# Kaggle 真实数据评测

[English documentation](kaggle-benchmark.md)

**当前状态（2026-10-16）：下载器、评测和中英文报告流程已用明确标记的离线夹具验证；真实 Kaggle 下载仍被云环境代理白名单拦截（CONNECT 403），尚未产生真实数据评测结果。** 没有虚构数据集版本、许可证、校验和、效果或性能数字。

## 数据选择

| 数据集 | 研究内容 | 特征与标签 |
| --- | --- | --- |
| [Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) | 不同数值单位、特征相关性下，GD/SGD/Adam 与 L-BFGS/Newton-CG 的收敛差异，以及 RMS 缩放的影响 | 30 个数值特征，排除 ID；`M` 为正类、`B` 为负类 |
| [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) | 类别不平衡、类别特征 one-hot 后的 CSR、稀有类别和概率质量 | 仅在训练集学习类别；`>50K` 为正类；排除抽样权重 `fnlwgt`，不把它作为特征或样本权重 |
| [Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) | OLS、Ridge、Lasso、Elastic Net 的效果、稀疏性和求解速度 | 年龄、BMI、子女数、性别、吸烟状态、地区；预测费用 `charges`，不加入目标派生特征或交互项 |

这些集合用于反复核验真实数据上的优化与预测表现。它们不能单独证明超大规模扩展能力；极宽 CSR 的可控研究继续使用已有 `sparse_wide` 合成实验。真实数据没有已知真系数，因此不编造支持集恢复率或“正确”的参数约束；约束能力仍由已有合成评测验证。

## 一键复现

```bash
python -m pip install '.[test,benchmark]'
python benchmarks/kaggle_data.py
python benchmarks/kaggle_suite.py --output build/kaggle-benchmark
python benchmarks/kaggle_report.py \
  --input build/kaggle-benchmark --output build/kaggle-report
```

默认三个种子 `42/43/44`、每项三次测量、200 次迭代或 epoch，并比较 raw/RMS。已有缓存时，快速流程检查可以运行：

```bash
python benchmarks/kaggle_suite.py --cases insurance --seeds 42 \
  --repeats 1 --iterations 2 --output build/kaggle-smoke
```

两次迭代只能检查流程，不能得出性能结论。完整评测可传 `--build-cache /path/to/installed/wheel/build` 记录实际编译参数。内存测量要求 Linux `/proc`。

输出保留 `results.json`、`metrics.csv`、`traces.csv`，以及中英文报告 `kaggle.en.md`、`kaggle.zh-CN.md`、效果/耗时图、汇总 JSON 和报告内容校验清单。原始单位的系数与截距随每次拟合保存；特征名保存在对应的模型选择记录中。非零数使用优化坐标中 `1e-6` 的阈值，跨缩放解释时应同时检查系数。

## 数据来源与缓存

下载器访问 Kaggle 公开 API，不读取凭据。首先读取真实元数据中的版本和许可证，再用明确的 `datasetVersionNumber` 下载。缺少版本或许可证、返回登录 HTML、ZIP 路径不安全或超限、CSV 列名或标签不匹配时均明确失败，不替换成镜像或合成数据。

缓存位于 `data/kaggle/<case>/`，包含原始 CSV、原始 `metadata.json` 和 `manifest.json`。清单记录版本、许可证、来源/下载 URL、时间、文件长度及 SHA-256；重复使用前核对 CSV 和元数据校验和。首次观察的哈希用于检测之后的本地变更，并非发布者签名。首次下载会读取当前版本，复现同一实验应保留缓存及清单；原始文件不进入 Git，报告保留版本化下载链接和校验和。未核实的许可证不会预填进目录。

## 避免数据泄漏与公平比较

- 固定训练/验证/测试为 70/15/15，分类按标签分层。数值缺失中位数、类别字典、RMS 都只从训练集拟合；类别 `?`/空值作为缺失类别，未知类别编码为全零；CSR 不中心化。训练中全缺失的数值列填零。异常的 NaN/Inf 字符串、非数值内容、非法标签直接失败，不静默丢弃样本。
- 回归目标用训练集均值和标准差归一化用于优化，RMSE/MAE 换回原始费用单位。常量基线只使用训练集标签均值或正类率。模型与优化器都使用相同分割，测试集不选择参数。
- Ridge/Lasso/Elastic Net 和逻辑回归 L2 使用验证集从 `0.0001/0.01/0.1` 选择，OLS 固定为零。每种缩放独立选择；因此 raw/RMS 的效果差异包含正则化含义变化，不能全部解释为条件数改善。
- 每个候选由独立 SciPy/NumPy/scikit-learn 参考解验证。各逻辑优化器共享同一个选定 L2，再按训练目标校准学习率；GD/SGD/Adam 各三次候选，L-BFGS/Newton-CG 一次，搜索成本单独保留。SGD/Adam 为固定学习率、batch 256，可能耗尽预算或振荡。
- 统一比较归一化目标差距 `(F−Fref)/(1+|Fref|) ≤ 1e-5`，保留错误、未收敛和线搜索失败。达标时间从 C++ 历史读取，未达到者留空，不填零；相同迭代数不代表相同扫描次数。逻辑回归参考解明确报告预条件坐标的梯度范数。
- 单线程、独立子进程、预热后重复拟合；fit 含 Python 输入复制，核心耗时取 C++ 历史，预测取七次中位数。预处理、参考选择和学习率搜索成本分开记录。RSS 是整个 worker 的 Linux VmHWM，包含解释器、预处理、数据和预热，不等同于求解器增量内存；未隔离 CPU 频率和后台负载。
- 先在每个种子内汇总重复，再跨种子汇总中位数与 IQR；完整历史仅保留首次重复，其余重复保留效果、工作量和达标证据。报告拒绝缺失或重复实验。类别编码后的列数可能随训练集种子变化。

分类联合报告 log loss、ROC AUC、**average precision**、固定 0.5 阈值的 accuracy 和 Brier，避免不平衡场景仅看 accuracy。average precision 使用 scikit-learn 的非插值定义，不等于梯形积分的 PR 曲线面积。回归报告 RMSE、MAE、R²，并同时给出常量基线与独立参考解的测试效果。

Insurance 的基础线性模型没有交互项，更快的求解不能消除模型逼近误差。历史观察数据的随机分割也不能证明未来或其他人群中的泛化与因果关系。达标时间只汇总实际达到的重复，必须结合达标数阅读，不能仅凭该列排名。

## 当前云环境阻塞

已保存网络配置草稿，保留原有 `api.github.com` 和包管理器预设，增加 `www.kaggle.com`、`kaggle.com`、`api.kaggle.com`、`storage.googleapis.com`。工具确认 `status: saved`、`requires_publish: true`。**保存草稿没有激活规则：需要在云环境设置中检查并保存更改，再发布环境，然后重试下载。** GitHub token 无法解决 Kaggle 代理 CONNECT 403；当前公开下载方式没有申请 Kaggle token。

## TDD 验证边界

测试先因三个新模块不存在而失败，再实现下载和评测。离线测试覆盖：固定下载版本、要求真实许可证元数据、危险 ZIP/错误文件/HTML 拒绝、CSV 与元数据校验和、列名和目标校验、互斥分割、训练集专属填补/编码/缩放、测试标签不影响正则化选择、带并列分数的 average precision、种子聚合，以及 worker 到中英文报告与图的完整执行。

测试夹具默认被真实评测拒绝；仅 `--allow-fixtures` 可启用，输出和图必须标记 **PROTOCOL FIXTURE**。这些测试通过只能证明流程可执行，不能证明真实数据下载或算法评测已经完成。
