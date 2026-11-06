# Intermediate candidates / 中间候选版本

These are complete earlier experiments on intermediate source/binary variants, using the unchanged 2× gate. The first four failed; the fifth passed timing but was superseded after a cross-compiler overflow fix. They are retained for audit and are not substituted for the final candidate. No numerical threshold or workload was changed.

以下为中间代码版本的完整实验，使用同一个 2× 门禁。前四轮未通过，第五轮性能通过，但随后发现需修复跨编译器溢出行为，因此不作为最终版本证据。保留原始数据供审查，不用它们替代最终候选，不修改数值阈值或场景。

| Attempt | Geometric mean | Gate | Raw data |
| --- | --- | --- | --- |
| 1 | 1.955829× | NOT MET | [attempt-1.results.json.gz](attempt-1.results.json.gz) |
| 2 | 1.917366× | NOT MET | [attempt-2.results.json.gz](attempt-2.results.json.gz) |
| 3 | 1.941797× | NOT MET | [attempt-3.results.json.gz](attempt-3.results.json.gz) |
| 4 | 1.989371× | NOT MET | [attempt-4.results.json.gz](attempt-4.results.json.gz) |
| 5 | 2.064578× | PASS, superseded by overflow fix | [attempt-5.results.json.gz](attempt-5.results.json.gz) |
| 6 | 1.965264× | NOT MET, overflow fix | [attempt-6.results.json.gz](attempt-6.results.json.gz) |

Each gzip expands to the full JSON: all paired measurements, histories, accuracy checks, source/binary hashes and build/machine metadata. The final `results.json` uses the final source variant and a fresh complete run.

每份 gzip 解压后均为完整 JSON，包含全部配对测量、历史、精度检查及源文件/二进制/构建/机器信息。最终 results.json 对应最终代码的另一轮完整测量。

One partial run was discarded because its start overlapped a source-wheel build. It is in [attempt-interrupted.results.json.gz](attempt-interrupted.results.json.gz); it was never treated as eligible evidence. The final run began after all build/test processes completed.

另有一次部分运行因开头与源码 wheel 构建重叠而作废，数据保留于上述链接，未作为验收证据。最终测量在全部构建/测试进程结束后单独启动。
