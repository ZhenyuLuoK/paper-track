# paper-track

paper tracking — 跟踪**智能体（Agent）**与**强化学习应用（RL Applications）**两个领域的最新论文（顶会 / 顶尖企业 / 顶级科研院校预印本）。每日按主题聚类产出带图像分析的中文简报。

## 目录结构

- `briefs/` — 带日期的论文简报（Markdown），含摘要、方法、相关工作、**图像分析**（动机 / 架构 / 实验结果）、实验与结果、结论及「so-what」，以及「该关注谁」。
- `papers/` — 论文原文 PDF（文件名含 arXiv id 与简称）。
- `figures/<arxiv_id>/` — 从各论文 PDF 抽取的关键图（动机 / 架构 / 结果），供简报内嵌与图像分析引用。

## 简报索引

| 日期 | 领域 | 主题 | 简报 | 收录论文 |
|------|------|------|------|---------|
| 2026-07-14 | 智能体 | CUA / 工具调用 / 长程状态 / 流程合规 / 全模态取证 | [2026-07-14-agent-papers-brief.md](briefs/2026-07-14-agent-papers-brief.md) | ScaleCUA、StructAgent、MM-ToolSandBox、Omni-Decision、Compile-Then-Page |
| 2026-07-14 | 智能体 · 第二批 | 机器人控制 / 弃权校准 / 工具侧记忆 / 多智能体探索 / 存储足迹 | [2026-07-14-agent-papers-brief-b.md](briefs/2026-07-14-agent-papers-brief-b.md) | VIA、AgentAbstain、ToolAtlas、MACE、AgentFootprint |
| 2026-07-14 | 强化学习应用 | 灵巧操作 sim-to-real / 生成式控制策略 / 可验证物理奖励 / 离线到在线 / LLM RL 稳定性 | [2026-07-14-rl-papers-brief.md](briefs/2026-07-14-rl-papers-brief.md) | REGRIND、VINE、RLVP、Robo-ValueRL、ARMOR |

## 收录论文原文

### 智能体（2026-07-14 批次）

| arXiv | 简称 | 机构 | PDF | 图 |
|-------|------|------|-----|----|
| 2607.11185 | ScaleCUA | 清华大学 · Z.AI | [PDF](papers/2607.11185_ScaleCUA.pdf) | [figures](figures/2607.11185_ScaleCUA/) |
| 2607.11388 | StructAgent | UCSD · Aether AI | [PDF](papers/2607.11388_StructAgent.pdf) | [figures](figures/2607.11388_StructAgent/) |
| 2607.11818 | MM-ToolSandBox | Apple | [PDF](papers/2607.11818_MM-ToolSandBox.pdf) | [figures](figures/2607.11818_MM-ToolSandBox/) |
| 2607.11433 | Omni-Decision | 阿里通义 · 中科院 · 清华 | [PDF](papers/2607.11433_OmniDecision.pdf) | [figures](figures/2607.11433_OmniDecision/) |
| 2607.11346 | Compile, Then Page | 香港理工 · 港大 | [PDF](papers/2607.11346_CompileThenPage.pdf) | [figures](figures/2607.11346_CompileThenPage/) |

### 智能体 · 第二批（2026-07-14）

| arXiv | 简称 | 机构 | PDF |
|-------|------|------|-----|
| 2607.11119 | VIA | 斯坦福大学 | [PDF](papers/2607.11119_VIA.pdf) |
| 2607.10059 | AgentAbstain | UIUC | [PDF](papers/2607.10059_AgentAbstain.pdf) |
| 2607.11126 | ToolAtlas | 北京大学 · 微软 | [PDF](papers/2607.11126_ToolAtlas.pdf) |
| 2607.11250 | MACE（Multi-Agent Exploration） | UW-Madison · UCSB | [PDF](papers/2607.11250_MultiAgentExplore.pdf) |
| 2607.11149 | AgentFootprint | 香港理工大学 | [PDF](papers/2607.11149_AgentFootprint.pdf) |

### 强化学习应用（2026-07-14 批次）

| arXiv | 简称 | 机构 | PDF | 图 |
|-------|------|------|-----|----|
| 2607.11874 | REGRIND | 康奈尔 · Amazon FAR | [PDF](papers/2607.11874_REGRIND.pdf) | [figures](figures/2607.11874_REGRIND/) |
| 2607.10369 | VINE | AgiBot · 港科大 · 北大 | [PDF](papers/2607.10369_VINE.pdf) | [figures](figures/2607.10369_VINE/) |
| 2607.10474 | RLVP | MIT | [PDF](papers/2607.10474_RLVP.pdf) | [figures](figures/2607.10474_RLVP/) |
| 2607.09866 | Robo-ValueRL | 人大 · 北大 · 微软研究院 · X-Humanoid | [PDF](papers/2607.09866_RoboValueRL.pdf) | [figures](figures/2607.09866_RoboValueRL/) |
| 2607.10481 | ARMOR | 中科大 · 北大 · 达特茅斯 | [PDF](papers/2607.10481_ARMOR.pdf) | [figures](figures/2607.10481_ARMOR/) |

## 方法与口径

- **检索源**：arXiv API（`cs.AI / cs.CL / cs.CV / cs.LG / cs.RO / cs.MA` 等）；智能体批次取当日提交，强化学习批次取近 7 天窗口。
- **聚类**：按**主题/论断**聚类而非按来源聚类，聚类名尽量用「趋势变化」表述。
- **信号优先**：剔除来源不明初创与「换基准复述旧结论」型论文，优先「生产/可落地」信号（开源全套或明确工程准则）。
- **图像分析**：每篇的动机/架构/结果图由脚本从 PDF 按图注定位裁剪至 `figures/<arxiv_id>/`，简报内嵌并结合图注与正文解读。
