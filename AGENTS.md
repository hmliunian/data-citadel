# Data Citadel 开发约定

本文件是工程约束入口；[agent.md](agent.md) 保存业务需求和已确认决策，
[concise-code](.agents/skills/concise-code/SKILL.md) 规定代码精简、验证和提交流程。
相同规则只维护一份，其他文档引用。开始修改前读取这三处。

## 工作范围

- `data_citadel` 是完整项目；源码、配置、工具和运行产物按本仓库组织，产物不迁往父目录。
  独立 Git worktree 是同一项目的另一个工作副本，目录名可以不同。
- 沿用用户最近授权的分支，不自动切回 `main`。当前分支有评测运行时，在独立 worktree 的新分支实施重构；
  原工作树的代码、配置、环境、产物和进程保留原状，不原地切换分支。
- 每个 worktree 单独建立 `.venv`、`.cache` 和 `artifacts`；不通过链接共用可写运行目录。
  原始数据可以共用只读路径。验证使用假模型和独立测试输出，不启动与现有评测竞争的真实审核。
- 保留工作区中其他任务的改动，提交时只暂存本次明确修改的内容。
- 工程重构遵守 `agent.md` 的业务边界；原始数据只读，GT 不进入模型，普通测试使用假客户端。

## 文件位置

新增文件先按职责放入已有目录；不要在仓库根目录或 `artifacts/` 下按日期、修复次数散建代码目录。

| 内容 | 位置 |
| --- | --- |
| 领域模型、判定与信号规则 | `citadel/domain/` |
| 审核、实验和任务调度用例、依赖协议 | `citadel/application/` |
| 模型、资源、MCAP、存储与执行器适配 | `citadel/infrastructure/` |
| HTTP 路由、公开结构、服务生命周期 | `citadel/server/` |
| 配置加载、依赖组装、命令入口 | `citadel/configuration.py`、`bootstrap.py`、`__main__.py`；包初始化为 `__init__.py` |
| 独立 HTTP 客户端、浏览器界面 | `citadel_client/`、`gui/` |
| 配置、任务规则和自然语言 prompt | `config/`，prompt 在 `config/prompts/` |
| 需要保留、复跑的工具与实验脚本 | `scripts/`，纳入 Git |
| 测试代码与固定测试素材 | `tests/` |
| 评测说明、错例分析与可复算统计 | `docs/` |
| 测试输出、一次性调试文件 | `.cache/tests/`、`.cache/gui-tests/`、`.cache/scratch/` |
| 服务运行数据 | 默认 `artifacts/server/`，部署路径通过服务配置指定 |
| 新实验结果 | `artifacts/experiments/<实验名>/` |
| 已结束且可搬移的历史实验 | `artifacts/archive/<实验名>/` |

`citadel/server/` 保存 HTTP 服务源码；`artifacts/server/` 保存服务运行数据。
新实验由正式模块从原始数据准备输入，媒体和资源缓存放在本轮实验内部；不要求历史实验缓存才能运行。

持续使用的代码必须纳入 Git，不得藏在被忽略的目录中。一次性脚本复用时移入 `scripts/`，
提取共用实现并补上运行说明。新实验用代码版本和配置快照记录来源，避免反复复制整套源码。
需要新增目录职责时，同步更新本文件和对应检查；不得用放宽检查掩盖违规。

## 依赖方向

下表限制仓库内部导入；各层可以导入本层模块。

| 模块 | 可以依赖的其他内部模块 |
| --- | --- |
| `citadel/domain/` | 无 |
| `citadel/application/` | `domain`、`configuration` |
| `citadel/infrastructure/` | `domain`、`configuration`、`application.ports` 中的协议 |
| `citadel/server/` | `application`、`domain`、`configuration` |
| `citadel/configuration.py` | `domain` |
| `citadel_client/` | 不导入服务端，通过 HTTP 调用 |

- 具体依赖在 `bootstrap.py` 组装，`__main__.py` 连接启动入口；业务层不反向导入它们。
- 领域和应用层使用标准库工具及 Pydantic；网络、数据库、媒体解码和模型供应商依赖放在基础设施层。
  不在领域或应用层执行这些 I/O，也不直接调用文件存储。
- `citadel/` 不导入 `citadel_client/`；两者都不导入 `scripts/` 或 `tests/`。
- 正式代码、工具和测试不得导入 `artifacts/`、`.cache/` 或历史源码快照。
  不得通过修改 `sys.path`、`PYTHONPATH`、动态加载或源码链接绕过边界。
- Prompt 文本及可配置的任务、模型、服务参数集中维护在 `config/`；同一业务规则只实现一份。

## 产物生命周期

- 实验开始时记录代码版本、有效配置、数据范围和运行命令；完成或中断时记录状态、已知模型用量及结果。
  使用已有清单、快照和调用记录即可，不另建重复记录体系。
- `calls/`、`results/`、配置快照和评测报告是证据，保留来源且不得覆盖；凭据不进入产物或 Git。
- `.cache/` 内容必须可重新生成；测试临时文件使用 pytest 的 `tmp_path`。
  不依赖上一次测试输出；并行测试须使用不同的临时目录。
- 历史源码快照作为只读证据保存，不适配新功能，不作为新流程的执行依赖。
- 旧实验目录在完成使用和引用核查前保留原位，不继续向其中开发新功能。
  归档前检查运行进程、配置、报告引用及共享媒体；搬移后同步引用并验证。
  不按目录名称批量删除服务数据或历史证据。

## 验证与交付

- 使用 uv 管理环境和锁文件，使用 `just` 入口运行开发、测试与检查。
- `just check` 执行源码位置检查、Python 静态导入检查、Ruff 和 Git 空白检查。
  位置检查覆盖 Git 已跟踪和未被忽略的新源码；导入检查支持绝对、相对和函数内导入。
- 静态检查不执行代码；动态加载、文件读写副作用、业务语义和忽略目录中的临时代码仍需审查与相关测试验证。
- 交付前检查新增文件位置、临时输出、相关文档和验证结果；提交要求遵循 `concise-code`。
