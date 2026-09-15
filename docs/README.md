# 项目文档

- [2026-09-15：七个 Qwen 模型的费用与审核效果 benchmark](benchmarks/2026-09-15/README.md)：98 条固定数据、完整分母指标、API 用量、公开价格与复算方法。
- [选型结论与错例分析](benchmarks/2026-09-15/analysis.md)：费用与误判的取舍、输出契约问题和待复查记录。

工程分层、启动和 HTTP API 使用方法见[项目 README](../README.md)。

## 历史报告的读取与复算

2026-09-15 的报告、错例分析和 5 份 CSV/JSON 数据从提交 `f5b54af` 原样迁入，
保留原始数值、代码摘要和协议。历史协议以 [protocol.json](benchmarks/2026-09-15/protocol.json) 为准；
报告中引用的配置和执行命令描述发布时的代码版本。当前 `config/benchmark.toml` 用于准备新实验。

在重构分支读取原工作树中保存的回执，可以执行：

```bash
.tools/bin/just benchmark-report \
  --work /home/xuran/projects/data_review/data_citadel/artifacts/experiments/model_benchmark_20260915 \
  --json
```

该命令只读已保存结果，不调用模型。迁入的文档无需原始 MCAP 或媒体缓存即可阅读；
复算需要上面实验目录中的 `experiment.json`、`inputs/` 和 `models/`。
需要重新生成报告时，使用 `--publish 新目录`，保留原发布文件。
