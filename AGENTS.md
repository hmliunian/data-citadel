# Data Citadel 开发约定

需求以 [agent.md](agent.md) 为准。开发时读取 [.agents/skills/concise-code/SKILL.md](.agents/skills/concise-code/SKILL.md)。

代码精简、不冗余；每个新功能相关测试通过后及时本地 commit，报告提交号与验证结果。仅在本项目内修改，直接在 main 工作。

新流程独立实现。原始数据只读，GT 不进入模型；实际模型试验记录范围、用量和结果，普通测试使用假客户端。
