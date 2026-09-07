<p align="center"><img src="./data_citadel.png" alt="Data Citadel" width="100%"></p>

# Data Citadel

面向机器人原子操作的严格视频审核服务。输入一次采集的 `episode_id`，读取本地 JSON/MCAP，以五条人工确认的专家视频作为参考，返回 `correct`、`incorrect` 或 `uncertain`，以及错误类型和时间证据。

当前支持全部十个动作：`A_001 A_002 A_003 A_004 A_005 A_009 A_011 A_012 A_013 A_015`。本阶段实现原子操作；长程任务尚未实现。具体需求见 [agent.md](agent.md)，开发约束见 [AGENTS.md](AGENTS.md)，验证记录见 [里程碑](docs/MILESTONES.md)。

## 启动

使用本项目独立的 Python 3.12 / uv 环境：

```bash
cd /home/xuran/xuran_projects/data_review/data_citadel
uv sync
uv run data-citadel inventory
uv run data-citadel serve
```

浏览器打开 `http://127.0.0.1:8000`，可选择动作和采集记录、预览画面并发起审核。交互式 API 文档位于 `/docs`。页面仅适合本地或受控内网；当前没有用户登录机制。

默认数据目录是项目旁的 `../datasets`，按以下结构索引：

```text
datasets/atomic/<action_id>/<category>/<episode_id>/
├── <episode_id>.json
├── episode.mcap
└── verification.json
```

原始数据只读。只纳入有校验收据的完整采集组；检查文件大小、修改时间、JSON 哈希及收据配对，解码时检查 MCAP chunk CRC。无收据的下载中目录不进入索引；损坏的已验证组会报错。每次进程启动建立一个本地数据快照，新增下载完成后重启服务更新索引。

## 专家与测试集

按原子动作生成专家和独立测试清单：

```bash
uv run data-citadel prepare --output-dir config
```

生成 `config/experts.json`、`config/evaluation.json`、`config/report.json`。再次生成时需使用新的输出目录，避免覆盖审批和旧实验。

`action_id` 是唯一的原子任务划分依据，例如搬运篮子和搬运小熊都属于 A_003。`task_code` 是数据源指令编号，`episode_id` 是一次采集；物体、指令编号和采集员不同不会拆成不同任务。每个动作内按 episode_id 稳定排序，前五条已有人工正确记录用于专家参考，随后五条作为独立正确测试。错误样本也按动作每类取五条；余量列入 `exploratory`，缺额写入报告，不能把样本不足当成分类效果已通过。

按用户确认，专家沿用数据源已有人工审核：样本必须位于 `correct`、状态为 `Accepted`，且有 `task.review.reviewer` 原审核人记录。生成器将来源记为 `approval_source: dataset_review`，满五条后启用；运行时再次核对源记录，不伪造二次审批。原审核人和审核时间保留在结果溯源中，不进入模型输入。设 `approved: false` 可禁用某组；手工登记仍支持 `approval_source: manual` 和真实 `reviewer`。专家帧可用网页预览或命令导出：

```bash
uv run data-citadel sample <episode_id> --interval 1 --output-dir artifacts/expert-preview-1
```

专家配置要求恰好五条独立正确样本；待测记录不能同时是自己的专家。当前策略是 `same_action_v2`；旧 `same_task_same_collector_v1` 清单需要重新 `prepare` 到新目录后切换，不能直接沿用旧拆分。配置变更后重启服务。缺少可用专家时任务分项为 `uncertain`，通用审核仍执行，已有明确通用错误仍可输出 `incorrect`。

## 模型配置

默认使用视觉模型 `qwen-vl-max`，而非纯文本 `qwen-max`。凭据优先级：

1. 环境变量 `QWEN_API_KEY` 或 `DASHSCOPE_API_KEY`。
2. `QWEN_API_KEY_FILE` 指定的文件。
3. 项目内 `Qwen-api/qwen_api_key.txt`；兼容原文档的 `../dataset_checker/Qwen-api/qwen_api_key.txt`。

凭据文件只应包含 key 本身。代码仅在发起请求时读取凭据，日志与提交不包含 key。可通过 `QWEN_MODEL`、`QWEN_BASE_URL` 调整模型和地域端点；key 必须与地域匹配。其他配置为 `DATASET_ROOT`、`CITADEL_ARTIFACTS_DIR`、`CITADEL_EXPERTS_PATH`、`CITADEL_CAMERA_TOPIC`，实现见 `settings.py`。

每条完整审核包含通用、任务两次模型请求。任务请求按顺序提供五条专家的各自指令和视频帧，再提供待测指令和帧；Qwen 从专家学习同类动作的过程，再按待测自己的物体和目标状态判断，不要求物体外观相同。这是同一次请求中的示例指导，不是额外的独立专家学习 API 阶段。HTTP 408/429/5xx 和网络异常有有限重试，请求最多 250 张图。超过图像预算会明确报错。

## 审核与抽帧

```text
episode_id → JSON/MCAP 完整性检查 → 主视角解码与抽帧
                                      ├→ 通用审核
                                      └→ 同 action_id 的五条专家 → 任务审核
                                                缺专家时：任务 uncertain
                                                     ↓
                                         严格规则合并 → 结果与证据
```

MCAP 中的 FlatBuffers schema 用于解析 `foxglove.CompressedVideo` 和 `foxglove.JointStates`。视频连续解码后按实际采集时间选择图像。

| 方案 | 行为 |
| --- | --- |
| `uniform` | 待测视频每 2 秒抽帧，补首尾；专家每 1 秒抽帧 |
| `keyframes` | 提取左右夹爪位置变化的开合起止时刻，补首尾并去重 |
| 连续运动摘要 | 所有解码帧参与低分辨率差分，提供超过 5 秒的低运动候选区间 |

默认只使用 `/camera/coracam_head/left_h264/video`。夹爪事件是采样启发式，阈值说明在 `media/sampling.py`；低运动不等于动作错误。夹爪数据不可用、帧预算缩减、视频覆盖缺口等情况会输出 warning，并阻止数据进入 ground truth 候选。

只有完整性、通用质量和任务完成检查都明确通过、有实际候选时间证据、没有采样警告且达到配置阈值时，才输出 `correct`。模型自报的置信度不代表经过统计校准，也不能保证零误判。

错误代码包括 `data_missing`、`blurred`、`content_mismatch`、`incomplete_action`、`repeated_retry`、`annotation_error`、`other`。遮挡或无意义静止用 `other` 并附具体原因；不确定的错误类别不强行归类。重试样本单独记录 `retry_outcome`，确认最终成功且有过程证据时标记 `retain_retry_sample`。此标记不使其成为普通 ground truth。

人工审核状态、拒绝原因、目录类别和采集员信息不进入模型提示词。Qwen 失败、无法解码或不支持的 schema 为运行错误，不伪装为语义上的 `incorrect`。

## API 与命令行

```bash
curl -X POST http://127.0.0.1:8000/v1/reviews \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":"<32位episode_id>","strategy":"uniform"}'

uv run data-citadel review <episode_id> --strategy keyframes
```

响应包含 `verdict`、`error_types`、`findings`、`ground_truth_candidate`、`retain_retry_sample`、分项判断和 `provenance`。模型、提示词、决策规则、专家配置的版本/哈希和实际抽帧时间被保留；审核结果保存到 `artifacts/reviews/`。

网页结果按 **GT 在上、Predict 在下** 展示：GT 是原人工审核标签；Predict 只显示一个最终标签（中文与代码），不并列多个预测。存在多个已确认错误时，采用 `error_types` 中第一项；完整错误和 Qwen 原始分项保留在展开详情，不改变严格审核结果。可另行展开五条专家的指令、输入帧数和任务判断。

GT 没有来源标签时显示“未提供”；模型的 `ground_truth_candidate` 只是严格策略筛选的候选，不替代原人工标签。GT 仅用于页面对照，不进入模型请求。

其他接口：`GET /health`、`GET /v1/inventory`、`GET /v1/episodes`、`GET /v1/episodes/{id}/frames`。HTTP 404 表示记录不存在，422 表示请求或数据问题，502 表示模型调用失败。

## 评测与验证

```bash
uv run data-citadel evaluate config/evaluation.json \
  --output-dir artifacts/evaluation-uniform --strategy uniform
uv run data-citadel evaluate config/evaluation.json \
  --output-dir artifacts/evaluation-keyframes --strategy keyframes

uv run pytest -q
CITADEL_REAL_MCAP=1 uv run pytest -q tests/test_repository.py tests/test_media.py
uv run ruff check src tests
```

评测输出 `predictions.jsonl` 和 `summary.json`，按动作及 baseline/exploratory 分组，报告正确精确率/召回率、错误误放行率、不确定比例、覆盖率、各类支持数及多标签混淆矩阵。没有预测正确样本时精确率为 `null`；服务异常单列并计入覆盖率分母。支持 `--limit N` 做小批量验证。真实 MCAP 测试无外部模型请求；普通测试使用合成数据与注入的模型客户端。

模型评测与工程测试不同：测试通过说明实现满足已覆盖的行为约束；分类效果需要有人工审核依据的专家集及独立人工测试集验证。

## 代码分层

`models/settings` 定义共享数据与配置；`repository/media` 负责读取和采样；`experts` 管理专家；`qwen/review` 完成模型调用与判定；`api/cli/runtime` 共享应用入口；`evaluation` 独立处理人工标签与指标。新增代码遵循 [concise-code skill](.agents/skills/concise-code/SKILL.md)。
