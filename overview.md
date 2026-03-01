下面给你一个“面试知识库（LLM 算法 / 对齐 / 训练 / 推理 infra）”的大纲模板，按可检索、可复用、可扩展来设计，确保能覆盖你收集的所有题型与追问方向（包括 DPO/SFT/RLHF、FlashAttention、ZeRO/并行、长上下文、KV cache、量化、手撕代码、项目拷打等）。

---
1. 知识库使用约定（首页）
0.1 导航结构
- 以“主题域（Domain）→ 页面（Page）→ 卡片（Cards）”三层组织
- 每个页面固定包含：
  - 一句话结论（面试 15 秒版）
  - 核心定义/公式
  - 关键实现细节/工程坑
  - 高频追问 & 标准回答
  - 反问面试官的高质量问题
  - 自测题（口述 + 手写）
0.2 标签体系（建议强制执行）
- 训练：#pretrain #SFT #DPO #PPO #GRPO #RM #RLHF
- 架构：#Transformer #MHA #GQA #MQA #MoE #MLA #RoPE #YaRN
- 工程：#FlashAttention #DeepSpeed #ZeRO #3DParallel #TP #PP #DP #Profiling
- 推理：#prefill #decode #kv_cache #paged_attention #batching #spec_decode
- 数值：#fp16 #bf16 #amp #int8 #int4 #quant
- 题型：#handwrite #derive #leetcode #system_design
- 公司/团队：可选 #字节 #阿里 #腾讯 #美团 #百度（方便回看面经差异）
0.3 统一“答题卡片模板”
- 结论：一句话
- 为什么：2–3 个关键因果链
- 怎么做：可落地步骤（含伪代码/配置/指标）
- 权衡：收益/代价/适用边界
- 追问：至少 5 个
- 常见错误：至少 3 个

---
1. Transformer 与基础组件（必考底座）
覆盖：Transformer forward、非线性来源、MHA/MQA/GQA、手写 FFN/Decoder layer/mask、embedding/tokenizer、字符串边界题等。
1.1 Transformer 总览
- Decoder-only vs Encoder-Decoder：结构差异、训练目标、应用场景（生成/理解/多模态）
- forward 链路拆解：Embedding → (RoPE/位置编码) → Attention → FFN → Residual + Norm → logits
1.2 Attention 机制全家桶
- Scaled Dot-Product Attention：公式、softmax 稳定性（max-trick）、mask 的实现方式
- MHA：为何多头、头数减少会怎样、是否“提升效率”的严格解释（算子形状与并行性）
- MQA / GQA：省的是什么、代价是什么（质量 vs 吞吐 vs 显存），以及在推理侧与 KV cache 的关系
1.3 FFN / SwiGLU / RMSNorm
- FFN 维度对齐（d_model, d_ff, gating），手写结构要点
- RMSNorm vs LayerNorm：计算/稳定性/实现细节
1.4 Tokenizer 与 Embedding
- tokenizer 实现方式：BPE、SentencePiece（Unigram/BPE）、WordPiece；差异与适用场景
- embedding：id→向量（查表）、tie embedding、是否参与微调与原因
- 中文/多语 tokenizer 常见坑（长度、标点、空格、特殊符号）
1.5 手撕专题（基础）
- 只能用 numpy 手写 MHA：形状推导、mask、softmax 稳定性、batch 维度处理
- 手撕 Transformer decoder layer + causal mask（上三角 -inf 加到 score）

---
2. 训练数据与训练流水线（从 txt 到 SFT）
覆盖：从预处理到 tokenize、forward、loss、参数更新的细节描述。
2.1 数据处理全链路
- 数据清洗：去重、脏词、编码、异常长度、重复段落、泄漏（test contamination）
- 结构化：instruction / input / output、多轮对话格式、system/user/assistant role
2.2 tokenize 与 packing
- padding vs packing、动态长度、sample packing 对吞吐/损失的影响
- 多轮对话的 loss mask（只算 assistant token / 只算最后一轮 / 全轮权重）
2.3 训练 loop（面试可口述到“算子级”）
- forward：logits、shift、label mask
- loss：Cross-Entropy（含 label smoothing 可选）
- backward：梯度、梯度裁剪、AMP/GradScaler
- optimizer：AdamW、weight decay 的处理（哪些参数不做 decay）
2.4 常见训练异常排查
- loss 震荡：数据分布、学习率、batch、混合精度溢出、梯度爆炸/消失
- NaN/inf：优先查 softmax、layernorm、loss scaling、异常样本

---
3. SFT 与参数高效微调（LoRA / QLoRA / 全参）
覆盖：LoRA 原理、初始化、全参 vs LoRA 对比、显存估算等。
3.1 SFT：目标与边界
- SFT 的作用：指令遵循、风格对齐、能力迁移
- 只用 SFT 可以吗：能到什么程度，为什么 RLHF 仍有价值
3.2 LoRA / Adapter
- 低秩分解为何只放在特定矩阵（QKV/O/FFN）
- A/B 初始化策略、rank/alpha/dropout 选择经验
- QLoRA：NF4、double quant、paged optimizer（面试常问“为什么能训得动”）
3.3 显存与吞吐估算（微调视角）
- 参数、梯度、优化器状态（Adam 一般 2×state）、activation checkpointing
- “项目里微调用了多少显存”如何解释得自洽

---
4. RLHF / 对齐：PPO、DPO、GRPO 与偏好数据
覆盖：RLHF 流程、PPO/DPO loss、DPO vs SFT、GRPO、DPO 数据构造/采样/配比、奖励黑客等。
4.1 RLHF 全流程（可画图口述）
- SFT → 采样 → 偏好标注/合成 → 训练奖励模型（RM）→ RL（PPO/GRPO）或 直接偏好优化（DPO）
- KL 约束的意义、reference model 的角色
4.2 PPO（面试常考点）
- policy ratio、clip、优势函数、KL penalty 的常见实现
- PPO 缺点与工程痛点：不稳定、采样成本、奖励模型偏差放大等
4.3 DPO（从“公式→直觉→实现”三段讲清）
- DPO vs SFT：目标差异（偏好相对排序 vs 拟合参考答案）
- DPO on-policy/off-policy 的口径与实现现实（数据收集方式 + 是否在线更新）
- chosen/rejected reward 都下降：从 logit margin、reference 偏置、数据噪声解释与排查路径
4.4 GRPO（以及“为什么不用 PPO”）
- GRPO 与 PPO 的关键差异点：稳定性、实现复杂度、采样效率的论证框架
- DeepSeek 为什么选 GRPO：你需要能回答“改了什么、解决了什么、代价是什么”
4.5 偏好数据设计（你题库里的核心难点）
- DPO/SFT 数据如何配比：单轮 vs 多轮，为什么这么配（覆盖/难度/噪声/标注成本）
- 多轮 DPO 样本采样策略：
  - 按轮次采样（早轮/中轮/末轮权重）
  - 按错误类型采样（事实/指令偏离/安全/格式）
  - 按对话长度分桶（短/中/长上下文）
- “第一轮对话做 DPO”：prompt / chosen / rejected 如何定义；是否保留 system；如何控制曝光偏差（exposure bias）
4.6 奖励黑客（Reward Hacking）
- 常见模式：投机取巧、重复、模板化、迎合 RM 偏好
- 防护：
  - reward model 训练：对抗样本、hard negative、数据多样性
  - 训练策略：KL、拒答策略、长度惩罚/归一化
  - 线上兜底：规则、判别器、RAG 校验、可回退策略（fallback）

---
5. 长上下文：RoPE、YaRN、外推与滑窗
覆盖：YaRN、RoPE 外推、多轮 position id、窗口滑动 pos 计算等。
5.1 上下文窗口与外推问题
- “只在短文本训过，长文本怎么外推”：训练侧（continued pretrain/packing）与推理侧（RoPE scaling/NTK/YaRN）
- 扩展位置编码的副作用：注意力模式改变、近邻偏置、长程依赖退化等
5.2 RoPE 体系
- RoPE 基本形式、频率尺度、相对位置信息来源
- RoPE scaling 各路线：线性/NTK/YaRN 的差异点与适用边界
5.3 YaRN（要能讲“直觉 + 机制 + 实际落地”）
- YaRN 的动机：保持短上下文能力同时扩展长上下文
- 与滑动窗口注意力/稀疏注意力对比：吞吐、显存、效果
5.4 多轮对话 position id 工程细节
- 连续对话 position id 处理策略
- 窗口滑动时 pos 如何重映射；对 cache/精度的影响

---
6. 模型架构对比：Qwen / LLaMA / DeepSeek、MoE、MLA
覆盖：架构差异、MoE vs Dense、DeepSeekMoE、MLA 等。
6.1 主流架构共同点
- Decoder-only + RMSNorm + SwiGLU + RoPE（大体趋势）
- GQA 的动机与收益
6.2 Qwen / LLaMA / DeepSeek 对比框架
- 结构层：attention 变体、是否 MoE、是否 MLA
- 训练层：数据、tokenizer、多语/中文策略
- 推理层：长上下文策略、部署生态（vLLM/TensorRT-LLM 适配点）
6.3 MoE 专题（面试高频）
- top-k routing、负载均衡 loss、专家容量（capacity factor）
- 难点：负载不均、训练不稳定、通信开销
- “为什么效果好/可借鉴点”：细粒度专家、共享专家等思路（结合你题库口径组织）

---
7. 分布式训练与显存优化：3D 并行、DeepSpeed ZeRO
覆盖：DP/DDP、3D 并行通信瓶颈、ZeRO-1/2/3、显存估算与瓶颈定位等。
7.1 并行策略总览
- DP / DDP：梯度同步（all-reduce）
- TP：张量切分；常见通信（all-reduce / all-gather）
- PP：流水并行；bubble 与 micro-batch
- 3D 并行：DP×TP×PP 的组合与瓶颈画像
7.2 通信瓶颈定位（回答要“能落地”）
- 哪些环节最卡：梯度 all-reduce、参数 all-gather、激活重算、optimizer step
- profiling 指标：iteration time 分解、通信/计算占比、链路抖动原因
7.3 DeepSpeed ZeRO
- ZeRO-1/2/3 分别切什么：optimizer state / gradients / parameters
- ZeRO-3 显存为什么还是大：activation、KV cache（推理）、fragmentation、通信缓冲区等
- “每张卡占用显存估算题”的答题模板（参数→梯度→状态→激活→碎片/峰值）

---
8. 数值精度与量化：FP16/BF16/INT8/INT4
覆盖：FP16 vs BF16、混合精度策略、INT4 精度损失场景与补救、哪些层不能乱量化等。
8.1 训练精度选择
- FP16：精度更细但动态范围小，易 overflow/underflow
- BF16：动态范围更大，训练更稳（常用于大模型 AMP）
- FP32：关键累计/归一化/softmax 可保 FP32（看框架实现）
8.2 推理量化路线
- INT8：权重量化/激活量化；常见副作用（质量下降、长文本更明显）
- INT4：outlier 权重/激活更敏感；分组量化、混合精度保关键层、QLoRA 微调补救
- “为什么有的层不能乱量化”：LayerNorm/输出层/注意力投影中的敏感路径
8.3 稳定性专题
- softmax 稳定性、NaN/inf 排查顺序（先数值，再数据，再并行）

---
9. 推理与 Infra：prefill/decode、KV cache、批处理、调度与线上故障
这一章基本对应你题库里“推理 infra”整块：KV cache 存什么、为什么没 Q cache、显存怎么估、paged attention、continuous batching、speculative decoding、TP/PP 推理瓶颈、profiling、OOM/吞吐抖动/串台等。
9.1 prefill vs decode
- 计算特性：prefill 更“算密集”，decode 更“带宽/访存/小 batch”敏感
- 为什么 decode 更慢：kernel 形态、KV 读写、同步与调度开销
9.2 KV cache 核心
- cache 里存什么：每层的 K/V（按 head 切分），形状与增长项（batch、seq、layer、head_dim）
- 为什么没有 Q cache：Q 只用于当前步，复用价值低
- 显存估算公式与“batch 一大就 OOM”的解释路径
9.3 paged attention / prefix cache / prompt cache
- paged attention 解决什么：避免大块连续显存、降低碎片、支持动态长度
- prefix/prompt cache 命中率：什么场景高、重排/拼接是否影响命中
9.4 batching 与调度
- 动态 batching vs continuous batching：吞吐/延迟权衡
- 合批为什么提吞吐、何时伤延迟、不同长度队列怎么排
- 线上“最怕的调度坑”：长尾、抖动、跨机通信、抢占
9.5 speculative decoding
- draft/target 怎么配、接受率低的后果、什么场景收益最大、为何有时更慢
- 与动态 batching 的协同策略、线上 ROI 评估指标
9.6 多卡推理（TP/PP）与通信
- TP 下 KV cache 怎么存、为什么通信拖慢 decode、all-reduce/all-gather 常见位置
9.7 性能 profiling 与线上故障手册
- 关键指标：tokens/s、TTFT、TPOT、显存峰值、cache hit rate、queue latency
- 故障剧本：
  - 吞吐掉一半怎么查、OOM 但看着还够、偶发慢定位、热更新后效果变差、输出循环/重复、串台来源

---
10. FlashAttention 与算子级优化
覆盖：FlashAttention vs 普通 attention，QK^T/softmax/PV 如何处理、对 prefill 更有用的原因、decode 常用 kernel 等。
10.1 FlashAttention 原理分解
- IO 视角：为什么普通 attention 被 HBM 带宽卡住
- block-wise 计算：QK^T 分块、在线 softmax（streaming）、再算 PV
- 数值稳定性：block 内 max/exp 累积的实现要点
10.2 工程追问库
- 为什么对 prefill 更有用：prefill 的矩阵更大更适合块化；decode 的形状更“瘦”
- 会不会影响精度：理论等价但实现需注意数值误差与精度路径

---
11. 评估体系：指标、基准、内部评测与创造性
覆盖：acc/precision/f1、样本不均衡影响、生成评测、创造性评估、benchmark 局限与内部评估设计。
11.1 分类/检索指标
- acc、precision、recall、f1 的定义与适用
- 类不均衡：为什么 acc 不可靠；micro/macro 的选择
11.2 生成评估
- 自动指标：ROUGE/BLEU、BERTScore、事实性（QA-based / entailment-based）
- 人评：偏好评测、pairwise、rubric 设计
11.3 创造性评估
- 新颖度、多样性、重复度（Self-BLEU 等思路）、“自动指标的局限”如何讲得专业
11.4 内部评测体系设计
- 离线集：覆盖业务意图、长尾、对抗
- 在线：A/B、护栏指标（拒答率/幻觉率/投诉率/成本）

---
12. 幻觉、RAG 与系统能力（产品级理解）
覆盖：幻觉原因、RAG 是否能彻底解决、单 LLM 能否完成“订票订酒店”以及需要哪些系统能力等。
12.1 幻觉的根因分层
- 训练分布：数据噪声/缺失
- 推理机制：next-token、曝光偏差
- 对齐副作用：过度迎合/编造
12.2 缓解策略
- RAG：召回、重排、引用、事实校验；为什么不能“彻底解决”
- 事实监督/工具调用/约束解码/自一致性
12.3 系统能力
- 任务分解、工具链（搜索/订票/支付）、权限与安全、状态机与记忆、失败回滚

---
13. 手撕/算法题专题库（按题型而非按题号）
覆盖：括号生成、最长子串、最大乘积子数组、零钱兑换、岛屿数量、井字棋、路径转换、字符串解码边界、多头自注意力等。
13.1 模板化题型
- 回溯：括号生成（剪枝点、合法性不变量）
- 滑窗：最长无重复子串（窗口维护、哈希/数组）
- DP：
  - 最大乘积子数组（同时维护 max/min）
  - 零钱兑换（完全背包：min/计数两套）
- 图/DFS/BFS：岛屿数量（递归栈/并查集对比）
- 字符串解析：解码/边界（空串、溢出、前导零、多位数）
13.2 “工程手撕题”
- 井字棋胜负判断：输入解析、状态合法性、获胜线枚举
- 只能 numpy 写 MHA：形状 + mask + softmax + 复杂度
13.3 每题页面结构
- 题意澄清问题清单（先问什么）
- 复杂度目标
- 常见坑
- 可扩展追问（流式输入/大规模/并行）

---
14. 项目经历与研究拷打：讲法、验证与复盘
覆盖：最重要贡献、如何验证有效性、对比学习负样本难题、实验显著性、结果不好怎么调等。
14.1 项目讲述框架（可直接背）
- 背景与目标：业务/指标/约束
- 你的贡献：明确“你做了什么”而不是“我们做了什么”
- 方法选择：为什么选它、与备选方案对比
- 验证闭环：离线→在线、ablation、统计显著性
- 失败与修正：遇到什么问题、怎么定位、怎么改、学到什么
14.2 对比学习专题
- 正负样本太像/拉不开：hard negative mining、温度系数、margin、采样分布重塑、loss 变体（InfoNCE/Triplet 等答题口径）
14.3 反问清单
- 团队在对齐/推理侧的核心挑战是什么
- 线上指标与约束（延迟/成本/质量）如何权衡
- 你入职后 3 个月的成功标准

---
15. 覆盖索引（把“问题清单”映射到知识库页面）
为了确保“一个问题都不漏”，建议在知识库加一页 《题库→页面映射表》，按你现有清单做索引，例如：
- 推理/infra 整块 → 第 9 章（prefill/decode、KV cache、batching、spec decode、profiling、OOM/串台）
- FlashAttention → 第 10 章（原理、分块 softmax、prefill vs decode）
- ZeRO/3D 并行/通信瓶颈/显存估算 → 第 7 章
- RLHF/PPO/DPO/GRPO + 偏好数据设计 + 奖励黑客 → 第 4 章
- 长上下文（YaRN/RoPE/滑窗 pos） → 第 5 章
- Tokenizer/Embedding/Transformer 基础/手撕 MHA → 第 1 章
- 指标与评估/内部评测 → 第 11 章

---
如果你希望把这个大纲直接落到 Notion/Obsidian/语雀的目录结构里，我可以按你偏好的工具输出一份可直接导入的目录树（含页面命名、标签、每页必填字段）。
