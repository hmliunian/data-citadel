# Data Citadel

UMI-T 独立原子任务测试窗口与 FastAPI。当前配置为抓取；代码不依赖旧项目实现。

三路相机按真实时间对齐，每 1 秒采样并保留首尾，整幅横向拼接。每张拼接帧紧邻真实时间说明，完整有序序列一次输入 Qwen，避免额外首尾帧造成固定帧率计时偏差。

夹爪位置与四个指尖触觉按原始时间戳读取，使用 0.1 秒区间极值补充视频帧间变化，辅助识别失败后重夹。触觉用测点力向量模长之和表示相对受力；单位未标定，不能单凭力值证明悬空或失败。

任务指令、物体图和场景图按实际 task_code 获取。分别检查物体、场景、主镜头可见性、画质、动作、失败重试和完整性，返回原因、时间证据及悬空时长。

抓取成功要求基本抓取后受控悬空约 2 秒，末态仍悬空。允许自由位置、路径、角度、调整和换手。默认时间容差 0.2 秒是待 GT 校准的试验参数；缺少足够时长、首尾或识别证据时待复核。模型调用/格式错误单独记为处理失败。

**运行**

当前项目已有环境与开发数据清单：

```bash
cd /home/xuran/projects/data_review/data_citadel
.venv/bin/python -m citadel --work-dir artifacts/gripper_v1 serve
```

窗口：`http://127.0.0.1:8766/`；接口文档：`/docs`。筛选支持误放、误拒和待复核，冻结后同时展示开发集与留出集。
主镜头、左腕和右腕均提供从 MCAP 导出的完整 MP4，可播放或下载；拼接视频为每 1 秒采样的模型视图。
有夹爪信号时显示位置与触觉曲线，点击曲线可把各路视频定位到同一采集时刻。
用 `/?filter=errors` 查看全部错例，追加 `&episode=记录ID` 可直达单条记录，并对照 GT 原因与模型结论。

远程访问可通过 SSH 转发：

```bash
ssh -N -L 8766:127.0.0.1:8766 xuran-5090-7f
```

新环境安装：`python3 -m venv .venv`，再运行 `.venv/bin/python -m pip install -e '.[dev]'`。

新建试验使用新目录，不覆盖已有清单：

```bash
.venv/bin/python -m citadel --work-dir artifacts/new_run prepare \
  --dataset /home/xuran/projects/data_review/datasets/DL-NA52UU
.venv/bin/python -m citadel --work-dir artifacts/new_run serve
```

当前默认模型为 `qwen3.8-max-0902`，非思考模式；模型与 prompt 的实测取舍见 [VALIDATION.md](VALIDATION.md)。
Qwen 凭据从 `QWEN_API_KEY` / `DASHSCOPE_API_KEY` 或 `Qwen-api/qwen_api_key.txt` 读取。
`QWEN_MODEL`、`QWEN_BASE_URL`、`QWEN_API_KEY_FILE` 可覆盖默认值；凭据不入 Git。
修改代码、规则或模型设置后重启服务；已有结果按配置区分。
只有审核操作调用模型，准备视频不调用模型。同版本结果会复用，执行失败可显式重试。

**验证与扩展**

- `review EPISODE_ID [--retry-failed]`：单条审核。
- `run --split development [--workers 3] [--limit N]`：批量审核开发集，默认最多 3 条并发。
- `report --split development`：导出完整分母统计及 JSONL，包含未运行、待复核、处理失败、误放、误拒和已知 token 用量。
- `freeze`：开发集全部处理且无执行错误后冻结代码、prompt、模型设置和任务参考资源，之后才开放 `holdout`。
- `run --split holdout`：留出集验证。修改冻结内容需新建试验；已看过的记录不能再宣称为独立测试集。

实际数据 98 条，固定划分开发集 68 条、留出集 30 条。GT 仅用于划分和结果对照，不进入模型输入。调用记录、媒体、结果与报告保存在 `artifacts/`，原始数据只读。请求记录保存图片摘要，不保存密钥或签名下载 URL。

新增原子任务在 `config/tasks.json` 添加规则：`task_codes`、`action_ids`、`display_name`、`success`、`allowed`、`failures`，需要保持时长的任务再添加 `hold_seconds/hold_tolerance_s`。不同任务可能复用动作编号，因此同时限定任务代码；无匹配或多重匹配时停止审核。

```bash
mkdir -p artifacts
.venv/bin/python -m pytest -q --basetemp artifacts/tests
.venv/bin/python -m ruff check citadel tests
```

开发遵循 [.agents/skills/concise-code/SKILL.md](.agents/skills/concise-code/SKILL.md)：代码精简，每个功能验证后及时本地提交。真实试跑结论见 [VALIDATION.md](VALIDATION.md)。
