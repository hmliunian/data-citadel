# Data Citadel

机器人采集审核的最小 A/B 对照试验。目前只验证任务 `DL-8GY1IC`，尚不能据此宣称全量清洗有效。需求与阶段记录见 [agent.md](agent.md)，开发约定见 [AGENTS.md](AGENTS.md)，真实结果见 [试验报告](pilot_reports/20260909_dl8gy1ic.md)。

- A：同任务专家图像 + 待测图像，一次审核。
- B：先缓存每位专家的逐帧描述，再结合待测图像审核。
- 专家限定同一 `task_code`、high、Accepted，并有原审核记录；数量由 `--experts` 配置，不固定为三条或五条。
- 三路相机每 2 秒采样并补首尾，保留实际时间。唯一标签为 `correct` / `incorrect`；证据不足为 `needs_review`、标签为空；运行失败单列。
- 原数据只读；GT 只用于本地展示和评测，不进入模型。

## 查看本轮结果

```bash
cd /home/xuran/projects/data_review/data_citadel
.venv/bin/python -m pilot --run-dir artifacts/pilot_20260909_dl8gy1ic serve --port 8765
```

打开 http://127.0.0.1:8765 。页面按 GT / Predict 排列，可切换 ID 和 A/B，展开专家，点击三路关键帧定位各自视频。

本轮网页/API **只读已有结果，不发起付费审核**：`GET /health`、`GET /api/results`、`GET /api/episodes/{episode_id}`、`GET /media/{episode_id}/{asset_name}`。不提供任意本地文件或源 MCAP 下载；独立集在冻结且已有对应审核结果之前不可浏览。服务仅绑定本机，不宜直接公开。

## 复现一个新试验

数据格式为 `task-summary.json`、`tasks/<task_code>/api_tags/`、`tasks/<task_code>/data/<id>/episode.mcap` 和 `receipts/`。默认源目录为 `/home/xuran/xuran_projects/data_review/datasets/20260907_afternoon`。

使用新的输出目录，不能覆盖已有清单或冻结结果。以下 `run` 命令会实际调用模型；其余命令不调用模型。

```bash
.venv/bin/python -m pilot --run-dir artifacts/new_pilot prepare --task-code DL-8GY1IC --experts 3
.venv/bin/python -m pilot --run-dir artifacts/new_pilot sample --split experts
.venv/bin/python -m pilot --run-dir artifacts/new_pilot sample --split development
.venv/bin/python -m pilot --run-dir artifacts/new_pilot run --split development --route A
.venv/bin/python -m pilot --run-dir artifacts/new_pilot run --split development --route B
# 查看开发结果后冻结；冻结以后不得再调参并复用该独立集。
.venv/bin/python -m pilot --run-dir artifacts/new_pilot freeze
.venv/bin/python -m pilot --run-dir artifacts/new_pilot sample --split holdout
.venv/bin/python -m pilot --run-dir artifacts/new_pilot run --split holdout --route A
.venv/bin/python -m pilot --run-dir artifacts/new_pilot run --split holdout --route B
.venv/bin/python -m pilot --run-dir artifacts/new_pilot report
```

当前选择器为这个小样本试验准备：开发和独立集各 3 正 + 1 负，并排除整个 high 候选池及重复 MCAP。样本不足会明确报错，不是通用全量划分器。先统一抽帧再并行 A/B；B 同一路线顺序执行，避免并发重复生成专家描述。

凭据顺序：`QWEN_API_KEY` / `DASHSCOPE_API_KEY` → `QWEN_API_KEY_FILE` → 项目 `Qwen-api/qwen_api_key.txt`（只含 key）。可设 `QWEN_MODEL`、`QWEN_BASE_URL`；默认 `qwen-vl-max`、北京 DashScope 兼容端点。凭据及大体积产物不进入 Git。

每次 HTTP 请求/响应和重试单独归档，图像请求日志保存哈希而不是 base64。结果保存在 `results/`；`report` 生成新的 CSV、JSONL 和汇总，统计所有归档调用的用量。估价不是实际账单，详见汇总中的定价来源。失败后显式加 `--retry-failed` 才会重试已失败样本。

## 全量现状回放

沿用当前 A/B 判法，排除整个 high 专家池，固定全量清单；这是与原 GT 的对照，不是新的独立验证。仅 `run` 付费调用，续跑默认跳过已有结果。

```bash
.venv/bin/python -m pilot.full --run-dir artifacts/new_full prepare --experts 3
.venv/bin/python -m pilot.full --run-dir artifacts/new_full run --first-per-task
.venv/bin/python -m pilot.full --run-dir artifacts/new_full run
.venv/bin/python -m pilot.full --run-dir artifacts/new_full report
```

报告含整体及各任务的正确率、误放行/误拒、覆盖率、待复核/失败和调用费用；使用 CSV/JSONL 查看，现有单任务网页不读取全量清单。

## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check pilot pilot_tests
PILOT_RUN_DIR=artifacts/pilot_20260909_dl8gy1ic .venv/bin/python -m pytest -q pilot_tests/test_real_media.py
```

普通测试使用假客户端；最后一项核对三条真实专家 MP4 的帧数/定位、已归档模型消息的 GT 隔离及真实结果 API/视频 Range，不调用模型。浏览器可在本机复查，本轮 Firefox 检查记录见报告。工程测试通过不等于模型效果达标；独立报告必须同时看误放行、误拒、自动覆盖率及分母。
