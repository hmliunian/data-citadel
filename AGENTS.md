# Data Citadel 开发约定

编写、修改或审查本项目代码时，读取并遵循 [.agents/skills/concise-code/SKILL.md](.agents/skills/concise-code/SKILL.md)。用户要求：代码要精简，不要冗余。

需求见 agent.md；当前先完成一个代表任务的最小验证。datasets 是项目外的原始只读输入。

专家只从相同 task_code 的 quantify=high 样本选择，数量由实验配置决定，禁止跨任务共用。专家候选池、开发集和独立测试集互斥。第一阶段只输出 correct / incorrect；证据不足为待复核，不输出错误类别。具体用手、任务一致性和重试规则遵循 agent.md 的已确认内容。

每个里程碑在相关测试通过后做本地 Git 提交，报告提交号与验证结果。模型调用测试默认使用可注入的假客户端；实际模型验证应明确报告调用范围与结果。
