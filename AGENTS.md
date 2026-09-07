# Data Citadel 开发约定

编写、修改或审查本项目代码时，读取并遵循 [.agents/skills/concise-code/SKILL.md](.agents/skills/concise-code/SKILL.md)。用户要求：代码要精简，不要冗余。

需求见 agent.md；本阶段覆盖全部 10 个原子动作。datasets 是项目外的原始只读输入。

每个里程碑在相关测试通过后做本地 Git 提交，报告提交号与验证结果。模型调用测试默认使用可注入的假客户端；实际模型验证应明确报告调用范围与结果。
