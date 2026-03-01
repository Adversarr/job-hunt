# 项目：LLM 算法面试知识库

## 概述
本项目用于构建 LLM 算法面试知识库，包含面试问答文档。

## 输入文件
- `overview.md` - 知识库大纲（15 章主题域）
- `questions.md` - 真实面试问题收集（字节/阿里/腾讯/美团等）

## 输出目录
`docs/` - 按 chapter 组织的 markdown 文档

## 文档模板
每个文档必须包含：
- 一句话结论
- 核心定义/公式
- 为什么（2-3 个因果链）
- 怎么做（可落地步骤）
- 权衡分析
- 高频追问（至少 5 个）
- 常见错误（至少 3 个）
- 反问面试官的问题
- 自测题
- 标签
- 相关文档

详细模板见 `.opencode/agents/write-document.md`

## 技术栈
- CUDA + PyTorch + NumPy
- Transformers / DeepSpeed / FlashAttention / vLLM
- 代码示例使用 Python + PyTorch

## 内容要求
- 精确：公式、代码、配置参数必须准确
- 详细：覆盖"是什么、为什么、怎么做、best-practice"
- 关联：添加文档间交叉引用
- 可检索：标签体系强制执行

## 验收标准
- 每个 overview 子主题都有对应文档
- 每个 questions.md 问题都能映射到具体文档
- 文档遵循统一模板
- 标签体系完整
- 文档间有交叉引用

## 生成进度

### 已完成文档（56 篇）

#### 第 0 章：知识库约定
- `docs/00-约定/README.md`

#### 第 1 章：Transformer 与基础组件
- `docs/01-Transformer基础/01-Transformer总览.md`
- `docs/01-Transformer基础/02-Attention机制.md`
- `docs/01-Transformer基础/03-FFN与归一化.md`
- `docs/01-Transformer基础/04-Tokenizer与Embedding.md`
- `docs/01-Transformer基础/05-手撕Transformer组件.md`

#### 第 2 章：训练数据与训练流水线
- `docs/02-训练数据流水线/01-数据处理全链路.md`
- `docs/02-训练数据流水线/02-tokenize与packing.md`
- `docs/02-训练数据流水线/03-训练Loop.md`
- `docs/02-训练数据流水线/04-训练异常排查.md`

#### 第 3 章：SFT 与参数高效微调
- `docs/03-SFT与LoRA/01-SFT目标与边界.md`
- `docs/03-SFT与LoRA/02-LoRA原理.md`
- `docs/03-SFT与LoRA/03-显存与吞吐估算.md`

#### 第 4 章：RLHF / 对齐
- `docs/04-RLHF对齐/01-RLHF总览.md`
- `docs/04-RLHF对齐/02-PPO算法.md`
- `docs/04-RLHF对齐/03-DPO算法.md`
- `docs/04-RLHF对齐/04-GRPO算法.md`
- `docs/04-RLHF对齐/05-偏好数据设计.md`
- `docs/04-RLHF对齐/06-奖励黑客.md`

#### 第 5 章：长上下文
- `docs/05-长上下文/01-上下文窗口与外推.md`
- `docs/05-长上下文/02-RoPE体系.md`
- `docs/05-长上下文/03-YaRN.md`
- `docs/05-长上下文/04-多轮对话Position ID.md`

#### 第 6 章：模型架构对比
- `docs/06-模型架构对比/01-主流架构共同点.md`
- `docs/06-模型架构对比/02-Qwen LLaMA DeepSeek对比.md`
- `docs/06-模型架构对比/03-MoE专题.md`

#### 第 7 章：分布式训练与 ZeRO
- `docs/07-分布式训练ZeRO/01-并行策略总览.md`
- `docs/07-分布式训练ZeRO/02-通信瓶颈定位.md`
- `docs/07-分布式训练ZeRO/03-DeepSpeed ZeRO.md`

#### 第 8 章：数值精度与量化
- `docs/08-数值精度量化/01-训练精度选择.md`
- `docs/08-数值精度量化/02-推理量化路线.md`
- `docs/08-数值精度量化/03-稳定性专题.md`

#### 第 9 章：推理与 Infra
- `docs/09-推理Infra/01-Prefill与Decode.md`
- `docs/09-推理Infra/02-KV Cache核心.md`
- `docs/09-推理Infra/03-Paged Attention.md`
- `docs/09-推理Infra/04-Batching与调度.md`
- `docs/09-推理Infra/05-Speculative Decoding.md`
- `docs/09-推理Infra/06-多卡推理TP PP.md`
- `docs/09-推理Infra/07-性能Profiling与故障.md`

#### 第 10 章：FlashAttention
- `docs/10-FlashAttention/01-FlashAttention原理.md`
- `docs/10-FlashAttention/02-FlashAttention工程.md`

#### 第 11 章：评估体系
- `docs/11-评估体系/01-分类检索指标.md`
- `docs/11-评估体系/02-生成评估.md`
- `docs/11-评估体系/03-创造性评估.md`
- `docs/11-评估体系/04-内部评测体系设计.md`

#### 第 12 章：幻觉与 RAG
- `docs/12-幻觉RAG/01-幻觉与RAG.md`
- `docs/12-幻觉RAG/02-系统能力.md`

#### 第 13 章：手撕/算法题
- `docs/13-手撕算法题/01-回溯题型.md`
- `docs/13-手撕算法题/02-滑窗题型.md`
- `docs/13-手撕算法题/03-DP题型.md`
- `docs/13-手撕算法题/04-图DFS-BFS题型.md`
- `docs/13-手撕算法题/05-字符串解析题型.md`
- `docs/13-手撕算法题/06-工程手撕题型.md`

#### 第 14 章：项目经历
- `docs/14-项目经历/01-项目经历讲述框架.md`
- `docs/14-项目经历/02-研究拷打.md`

#### 第 15 章：覆盖索引
- `docs/15-覆盖索引/README.md`

### 完成状态
✅ 15 章主题域全部覆盖
✅ 所有 questions.md 问题已映射到文档
✅ 文档遵循统一模板
✅ 标签体系完整
✅ 文档间有交叉引用