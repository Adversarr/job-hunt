# RLHF 全流程总览

## 一句话结论
RLHF 通过"人类偏好对齐"突破 SFT 的天花板，流程为 **SFT 建立 base → RM 学习偏好 → PPO/DPO/GRPO 优化策略**，核心是用 KL 约束防止模型偏离预训练能力，是 LLM 安全性和有用性的关键对齐技术。

## 核心定义/公式

### RLHF 完整流程
```
预训练模型 (Base Model)
     ↓
   SFT (有监督微调)
     ↓
  采样与标注 (Sampling & Annotation)
     ↓
  训练奖励模型 (Reward Model Training)
     ↓
  ┌─────────┴──────────┐
  ↓                     ↓
PPO/GRPO 流程        DPO 流程
(RM + 策略优化)      (直接偏好优化)
```

### PPO Loss（带 KL 约束）
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：策略比率
- $\hat{A}_t$：优势函数（通过 RM 计算奖励）
- $\epsilon$：clip 参数，通常 0.1~0.2
- $\beta$：KL 约束系数，通常 0.05~0.1

### DPO Loss
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right]$$

其中：
- $y_w$：chosen（偏好回答）
- $y_l$：rejected（非偏好回答）
- $\beta$：KL 约束系数

### Reward Model Loss
$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

### KL 散度（约束项）
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

### 代码示例（RLHF 流程伪代码）

```python
# ===== 阶段 1: SFT =====
sft_model = train_sft(base_model, sft_data)
# 数据: {"prompt": "...", "response": "..."}

# ===== 阶段 2: 采样与标注 =====
# 2.1 采样：用 SFT 模型生成多个回答
prompts = ["介绍一下北京", "如何学习编程", ...]
for prompt in prompts:
    responses = sft_model.generate(
        prompt, 
        n=4,  # 每个 prompt 生成 4 个回答
        temperature=0.7
    )
    # responses: [response_1, response_2, response_3, response_4]

# 2.2 标注：人类/LLM 标注偏好
preference_pairs = annotate_preferences(responses)
# 输出: {"prompt": "...", "chosen": response_1, "rejected": response_3}

# ===== 阶段 3: 训练奖励模型 =====
reward_model = train_reward_model(preference_pairs)
# RM 输入: (prompt, response)
# RM 输出: scalar reward

# ===== 阶段 4: RL 优化（PPO 流程）=====
policy_model = copy.deepcopy(sft_model)
ref_model = copy.deepcopy(sft_model)  # Reference model 冻结
ref_model.eval()

for epoch in range(num_epochs):
    # 4.1 Rollout: 策略模型生成回答
    queries = sample_prompts()
    responses = policy_model.generate(queries)
    
    # 4.2 计算 reward
    rewards = reward_model.score(queries, responses)
    # 可选：添加规则奖励（长度惩罚、格式奖励等）
    rewards = rewards + rule_based_rewards(queries, responses)
    
    # 4.3 计算 KL 惩罚
    with torch.no_grad():
        ref_log_probs = ref_model.log_prob(queries, responses)
    policy_log_probs = policy_model.log_prob(queries, responses)
    kl_penalty = beta * (policy_log_probs - ref_log_probs).mean()
    
    # 4.4 计算优势函数（需要 value model）
    advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    
    # 4.5 PPO 更新
    loss = compute_ppo_loss(
        policy_log_probs, old_log_probs, advantages,
        ref_log_probs, clip_ratio=0.2, kl_coef=beta
    )
    loss.backward()
    optimizer.step()

# ===== 阶段 4': DPO 流程（无需 RM）=====
policy_model = copy.deepcopy(sft_model)
ref_model = copy.deepcopy(sft_model)

for epoch in range(num_epochs):
    for batch in preference_pairs:
        # 计算 log probs
        policy_chosen_logps = policy_model.log_prob(batch.prompt, batch.chosen)
        policy_rejected_logps = policy_model.log_prob(batch.prompt, batch.rejected)
        
        with torch.no_grad():
            ref_chosen_logps = ref_model.log_prob(batch.prompt, batch.chosen)
            ref_rejected_logps = ref_model.log_prob(batch.prompt, batch.rejected)
        
        # DPO loss
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        loss.backward()
        optimizer.step()
```

## 为什么（2-3 个因果链）

### 1. 为什么 SFT 之后还要做 RLHF，只用 SFT 可以吗？（腾讯三面）

**只用 SFT 的局限性**：
- **现象**：SFT 模型能模仿格式，但在安全性、有用性上容易"翻车"
- **根因 1**：SFT 是模仿学习（Imitation Learning）
  - 只学习"什么是好的回答"
  - 不知道"什么是坏的回答"
  - 无法区分回答的好坏差异，只能拟合专家分布
- **根因 2**：分布偏移问题
  - SFT 数据通常是"理想回答"，与实际使用分布不同
  - 模型没见过"坏回答"，遇到对抗性输入时容易生成危险内容
- **根因 3**：多目标优化困难
  - SFT 无法显式优化"安全性 vs 有用性"的权衡
  - RLHF 可以通过奖励函数显式建模偏好

**RLHF 的增益**：
1. **偏好学习**：通过对比"好 vs 坏"，学习人类偏好的边界
2. **分布对齐**：RM 在真实使用分布上训练，泛化性更强
3. **多目标权衡**：奖励函数可以综合安全性、有用性、诚实性等

**因果链**：
```
SFT 只学"好回答"
  → 不知道"坏回答"长什么样
  → 面对对抗性输入时缺乏防御
  → 安全性和有用性不足

RLHF 引入偏好对比
  → 学习"好 vs 坏"的边界
  → RM 可以泛化到未见过的"坏回答"
  → 安全性和有用性显著提升
```

**实践证据**：
- InstructGPT 论文：SFT 模型在人类偏好评估中显著落后于 RLHF 模型
- Llama 2 Chat：SFT → RM → PPO 的递进式优化，每阶段都有显著提升
- Constitutional AI：SFT 模型容易生成有害内容，RLHF 后安全性大幅提升

### 2. 为什么需要 KL 约束和 reference model？

**没有 KL 约束的后果**：
- **现象 1**：奖励黑客（Reward Hacking）
  - 模型学会"刷分"，生成 RM 给高分但实际无用的回答
  - 例如：重复模板、长度堆砌、迎合 RM 偏见
- **现象 2**：能力退化（Catastrophic Forgetting）
  - 过度优化奖励，忘记预训练阶段的知识和能力
  - 例如：只学"安全回复"，丧失专业能力

**因果链**：
```
RM 只在特定分布上训练
  → 泛化性有限
  → 策略偏离后，RM 的评分不再可靠
  → 模型可能找到"刷分"策略（奖励黑客）

KL 约束限制策略偏离
  → 强制策略靠近 reference model
  → 保持预训练/SFT 阶段的能力
  → 防止能力退化
```

**Reference Model 的角色**：
1. **锚点（Anchor）**：固定不变，作为"能力基线"
2. **约束源**：KL 散度计算需要参考分布
3. **不参与训练**：避免 moving target 问题

### 3. 为什么 RLHF 能学习人类偏好？

**Bradley-Terry 模型**：
$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

**因果链**：
```
人类偏好数据：(prompt, chosen, rejected)
  → 假设偏好概率与奖励差相关
  → RM 学习预测"chosen 比 rejected 好"的概率
  → RM 泛化到未见过的回答
  → PPO/DPO 用 RM 信号优化策略
  → 策略生成更符合偏好的回答
```

**关键洞察**：
- RM 不需要预测"绝对分数"，只需预测"相对排序"
- 排序任务比打分任务更鲁棒，对标注噪声更不敏感
- 偏好对比数据比绝对评分更容易标注（人类更擅长比较）

## 怎么做（可落地步骤）

### 标准做法

#### 阶段 1: SFT（有监督微调）

**目标**：建立基础能力（指令遵循、格式对齐）

**步骤**：
```python
# 1. 准备 SFT 数据
sft_data = [
    {"prompt": "介绍一下北京", "response": "北京是中国的首都..."},
    {"prompt": "如何学习编程", "response": "学习编程建议从..."},
    ...
]

# 2. 训练 SFT 模型
sft_model = AutoModelForCausalLM.from_pretrained(base_model)
trainer = SFTTrainer(
    model=sft_model,
    train_dataset=sft_data,
    args=TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=3,
        batch_size=32,
        ...
    )
)
trainer.train()
```

**关键点**：
- 数据量：通常 10K~100K 条高质量指令数据
- 数据质量 > 数据量：每条数据都应经过人工审核
- 覆盖度：覆盖多种任务类型（问答、对话、推理、创作等）

#### 阶段 2: 采样与偏好标注

**步骤**：
```python
# 1. 用 SFT 模型采样
prompts = sample_prompts_from_distribution()  # 从实际使用分布采样
for prompt in prompts:
    # 每个 prompt 生成 4~8 个回答
    responses = sft_model.generate(
        prompt,
        n=4,
        temperature=0.7,
        top_p=0.9
    )
    
    # 2. 标注偏好（人工或 LLM）
    # 方案 A: 人工标注
    ranked = human_annotate(responses)  # 从好到坏排序
    chosen, rejected = ranked[0], ranked[-1]
    
    # 方案 B: LLM 辅助标注（如 GPT-4）
    ranked = gpt4_rank(responses, criteria=["helpful", "harmless", "honest"])
    chosen, rejected = ranked[0], ranked[-1]
    
    preference_pairs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })
```

**标注策略**：
1. **人工标注**：质量高但成本高，适合核心场景
2. **LLM 辅助**：用 GPT-4 等强模型标注，成本较低
3. **规则生成**：对简单场景用规则生成 rejected（如重复、过短、跑题）
4. **混合策略**：核心数据人工标注，边缘数据 LLM 辅助

**数据量**：
- PPO: 通常需要 50K~200K 偏好对（用于训练 RM）
- DPO: 通常需要 10K~100K 偏好对（直接用于训练）

#### 阶段 3: 训练奖励模型（RM）

**步骤**：
```python
# 1. 初始化 RM（通常用 SFT 模型初始化）
reward_model = AutoModelForSequenceClassification.from_pretrained(
    sft_model_path,
    num_labels=1  # 输出单个标量奖励
)

# 2. 训练 RM
def train_reward_model(rm, preference_pairs):
    optimizer = AdamW(rm.parameters(), lr=1e-5)
    
    for batch in preference_pairs:
        # 计算 chosen 和 rejected 的奖励
        chosen_rewards = rm(batch.prompt, batch.chosen)
        rejected_rewards = rm(batch.prompt, batch.rejected)
        
        # Bradley-Terry loss
        loss = -torch.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return rm
```

**关键配置**：
- 学习率：1e-5 ~ 1e-6（较小，避免过拟合）
- 数据增强：对 rejected 做轻微修改，增加多样性
- 过滤噪声：移除 chosen/rejected 相似度过高的样本

#### 阶段 4: RL 优化（PPO / DPO / GRPO）

**选择策略**：

| 算法 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| PPO | 需要在线探索、精细控制 | 上限高、可探索新策略 | 实现复杂、不稳定、成本高 |
| DPO | 偏好数据充足、预算有限 | 简单、稳定、无需 RM | 无法在线探索、依赖数据质量 |
| GRPO | 超大规模模型对齐 | 比 PPO 稳定、无需价值函数 | 实现较新、生态不成熟 |

**PPO 流程**（详见 [[02-PPO算法]]）：
```python
# 1. 初始化模型
policy_model = copy.deepcopy(sft_model)
ref_model = copy.deepcopy(sft_model)
value_model = init_value_model(sft_model)  # 共享 backbone，单独 value head

# 2. RL 训练循环
for epoch in range(num_epochs):
    # Rollout: 采样
    queries = sample_prompts()
    responses = policy_model.generate(queries)
    
    # 计算 reward
    rewards = reward_model(queries, responses)
    
    # 计算 advantage（需要 value model）
    advantages = compute_gae(rewards, value_model(queries, responses))
    
    # PPO update
    for _ in range(ppo_epochs):
        loss = compute_ppo_loss(...)
        loss.backward()
        optimizer.step()
```

**DPO 流程**（详见 [[03-DPO算法]]）：
```python
# DPO 不需要 RM，直接用偏好数据训练
policy_model = copy.deepcopy(sft_model)
ref_model = copy.deepcopy(sft_model)

for batch in preference_pairs:
    # 计算 log probs
    policy_chosen_logps = policy_model.log_prob(batch.prompt, batch.chosen)
    policy_rejected_logps = policy_model.log_prob(batch.prompt, batch.rejected)
    
    with torch.no_grad():
        ref_chosen_logps = ref_model.log_prob(batch.prompt, batch.chosen)
        ref_rejected_logps = ref_model.log_prob(batch.prompt, batch.rejected)
    
    # DPO loss
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    loss.backward()
    optimizer.step()
```

### 关键配置/参数

#### SFT 阶段
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 2e-5 | 较小，避免破坏预训练能力 |
| 数据量 | 10K ~ 100K | 质量 > 数量 |
| Epochs | 3 ~ 5 | 过多容易过拟合 |
| Batch size | 32 ~ 128 | 视显存而定 |

#### RM 训练阶段
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-6 ~ 1e-5 | 更小，防止过拟合 |
| 数据量 | 50K ~ 200K | 需要足够多样性 |
| 模型大小 | 与 policy 相当或稍小 | 平衡精度和推理成本 |

#### PPO 阶段
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| KL 系数（β） | 0.05 ~ 0.1 | 控制策略偏离程度 |
| Clip ratio（ε） | 0.1 ~ 0.2 | PPO 裁剪参数 |
| Learning rate | 1e-6 ~ 5e-6 | 更小，稳定训练 |
| PPO epochs | 3 ~ 4 | 每批数据更新轮数 |

#### DPO 阶段
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| β | 0.1 ~ 0.5 | 0.1 通用起点，质量高可增大 |
| Learning rate | 1e-6 ~ 5e-7 | 比 PPO 更小 |
| 数据量 | 10K ~ 100K | 数据质量敏感 |

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **SFT → PPO** | 上限高，可在线探索新策略 | 实现复杂，训练不稳定，需要 RM + Value Model | 有充足算力和标注资源，追求最优效果 |
| **SFT → DPO** | 简单稳定，无需 RM，成本低 | 无法在线探索，依赖数据质量 | 偏好数据质量高，预算有限 |
| **SFT → GRPO** | 比 PPO 稳定，无需价值函数 | 实现较新，生态不成熟 | 超大规模模型（如 DeepSeek） |
| **只用 SFT** | 简单快速，成本最低 | 无法学习偏好边界，安全性和有用性受限 | 数据量小，只需格式对齐 |

### SFT vs DPO vs PPO 详细对比

| 维度 | SFT | DPO | PPO |
|------|-----|-----|-----|
| **目标** | 拟合参考答案（模仿） | 学习偏好排序（对比） | 最大化奖励（探索） |
| **数据** | (prompt, response) | (prompt, chosen, rejected) | 在线采样 + RM 打分 |
| **优化方式** | Cross-entropy loss | Preference loss | Policy gradient + KL |
| **探索能力** | 无（只学已有答案） | 无（只学已有偏好） | 有（可探索新策略） |
| **实现复杂度** | 低 | 中 | 高 |
| **训练稳定性** | 高 | 高 | 低 |
| **效果上限** | 中 | 中高 | 高 |
| **计算成本** | 低 | 中 | 高 |

### DPO 和 SFT 的区别（字节一面）

**目标差异**：
- **SFT**：拟合参考答案，学习"什么是好的回答"
- **DPO**：学习偏好排序，学习"好回答 vs 坏回答的边界"

**数据差异**：
- **SFT**：需要 (prompt, response) 对
- **DPO**：需要 (prompt, chosen, rejected) 三元组

**Loss 差异**：
- **SFT**：$\mathcal{L}_{\text{SFT}} = -\mathbb{E}[\log \pi_\theta(y|x)]$（Cross-entropy）
- **DPO**：$\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$

**能力差异**：
- **SFT**：建立基础能力（指令遵循、格式对齐）
- **DPO**：对齐偏好（安全性、有用性、诚实性）

### 先 DPO 后 SFT 可以吗？

**答案：不可以**

**原因**：
1. **SFT 是建立基础能力**：
   - SFT 教会模型"如何回答问题"（格式、风格、指令遵循）
   - 没有 SFT，模型可能连基本对话能力都没有

2. **DPO 是在已有能力上微调偏好**：
   - DPO 假设模型已经有生成合理回答的能力
   - 如果模型基础能力不足，DPO 会放大噪声，效果很差

3. **反序的后果**：
   - **DPO → SFT**：DPO 学到的偏好会被 SFT 覆盖
   - 模型回到 SFT 数据的分布，丢失偏好对齐效果
   - SFT 数据通常是"理想回答"，但不知道"坏回答"长什么样

**正确顺序**：
```
Base Model → SFT（建立能力） → DPO/PPO（对齐偏好）
```

**因果链**：
```
SFT 学习格式和指令遵循
  → 模型能生成合理回答
  → DPO 在此基础上学习偏好
  → 模型既能回答问题，又符合偏好

如果先 DPO：
  → 模型基础能力不足
  → DPO 难以学到有意义的偏好
  → 再做 SFT 会覆盖 DPO 的效果
  → 最终效果不如 SFT → DPO
```

## 高频追问（至少 5 个）

### 1. Q: 讲一下 RLHF 的流程，写一下 PPO 和 DPO 的 Loss 表达式（阿里通义）

**A**: 

**RLHF 完整流程**：
1. **SFT 阶段**：用有监督数据训练模型，建立基础能力
2. **采样与标注**：用 SFT 模型生成多个回答，人工/LLM 标注偏好
3. **训练 RM**：用偏好数据训练奖励模型
4. **RL 优化**：用 PPO/DPO/GRPO 优化策略

**PPO Loss**：
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：策略比率
- $\hat{A}_t$：优势函数（通过 RM 计算）
- $\epsilon$：clip 参数
- $\beta$：KL 约束系数

**DPO Loss**：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right]$$

其中：
- $y_w$：chosen（偏好回答）
- $y_l$：rejected（非偏好回答）
- $\beta$：KL 约束系数

**核心差异**：
- **PPO**：需要 RM + 在线采样 + value model
- **DPO**：直接用偏好数据，无需 RM

### 2. Q: 知道哪些强化学习算法，除了 PPO 和 DPO 这些呢？（腾讯三面）

**A**: 

**RLHF 相关算法**：

1. **PPO（Proximal Policy Optimization）**
   - OpenAI 主力算法，InstructGPT/Llama 2 使用
   - 通过 clip 约束策略更新，稳定性较好
   - 需要 RM + value model

2. **DPO（Direct Preference Optimization）**
   - 绕过 RM，直接用偏好数据优化
   - 实现简单，训练稳定
   - 无在线探索能力

3. **GRPO（Group Relative Policy Optimization）**
   - DeepSeek 使用，无需价值函数
   - 通过组内对比计算相对优势
   - 比 PPO 更稳定

4. **RRHF（Rank Responses to Align Human Feedback）**
   - 用排序损失替代 PPO 的复杂训练
   - 更简单，但效果略逊于 PPO

5. **IPO（Identity Policy Optimization）**
   - DPO 的变体，移除 sigmoid 假设
   - 对奖励函数分布更鲁棒

6. **KTO（Kahneman-Tversky Optimization）**
   - 基于前景理论，不需要成对偏好数据
   - 可以用二元反馈（好/坏）

**传统 RL 算法**：
- **TRPO（Trust Region Policy Optimization）**：PPO 的前身，二阶优化
- **REINFORCE**：最简单的策略梯度，方差大
- **A3C/A2C**：异步优势 Actor-Critic
- **SAC（Soft Actor-Critic）**：最大熵 RL

**LLM 特有算法**：
- **Expert Iteration**：迭代式自我改进（AlphaGo 风格）
- **STaR（Self-Taught Reasoner）**：用推理链自我训练

### 3. Q: DPO 和 SFT 的区别？先 DPO 后 SFT 可以吗？（字节一面）

**A**: 详见"权衡分析"部分的详细对比。

**简答**：
- **区别**：SFT 是模仿学习（拟合答案），DPO 是偏好学习（对比好坏）
- **顺序**：不可以先 DPO 后 SFT
  - SFT 建立基础能力（格式、指令遵循）
  - DPO 在已有能力上微调偏好
  - 反序会导致 DPO 效果被 SFT 覆盖

### 4. Q: 为什么需要 reference model？冻结还是更新？

**A**: 

**Reference Model 的作用**：
1. **KL 约束的锚点**：计算 KL 散度需要参考分布
2. **防止能力退化**：限制策略偏离预训练/SFT 能力
3. **稳定训练**：避免奖励黑客

**是否更新**：
- **标准做法：冻结**
  - 如果 reference model 也更新，会形成 moving target
  - KL 约束会失效，训练不稳定
  - 参考 DPO 论文：冻结 reference model 是最佳实践

- **特殊情况：定期更新**
  - Iterative DPO：训练 N 步后，用当前策略更新 reference
  - 但这不是标准做法，效果存疑

**代码示例**：
```python
# 正确做法：冻结
ref_model = copy.deepcopy(sft_model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# 错误做法：参与训练
# for param in ref_model.parameters():
#     param.requires_grad = True  # 会导致训练崩溃
```

### 5. Q: KL 约束系数 β 怎么选？

**A**: 

**β 的作用**：
- 控制策略偏离 reference model 的程度
- β 大：约束强，模型变化小，偏好学习弱
- β 小：约束弱，模型变化大，可能遗忘或奖励黑客

**选择策略**：
1. **通用场景**：β = 0.1
2. **数据质量高**：β = 0.3 ~ 0.5（允许更大偏离）
3. **数据质量低**：β = 0.05 ~ 0.1（强约束防过拟合）

**调试方法**：
```python
# 监控 KL 散度
kl_div = (policy_log_probs - ref_log_probs).mean()

# 健康范围：KL 在 [0.5, 5.0] nat
# KL > 10: β 太小，模型偏离太远，可能遗忘
# KL < 0.1: β 太大，模型几乎不变，偏好学习不足
```

**自适应调整**（PPO 常用）：
```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.1, target_kl=6.0):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
    
    def update(self, current_kl):
        if current_kl < self.target_kl / 1.5:
            self.kl_coef /= 2  # KL 太小，减弱约束
        elif current_kl > self.target_kl * 1.5:
            self.kl_coef *= 2  # KL 太大，加强约束
        return self.kl_coef
```

### 6. Q: 偏好数据如何构造？数据量要多少？

**A**: 

**数据构造策略**：

**1. 人工标注**：
- 质量：高
- 成本：高（$1~$5 每对）
- 适用：核心场景、高风险任务

**2. LLM 辅助标注**：
```python
# 用 GPT-4 标注偏好
responses = model.generate(prompt, n=4)
ranked = gpt4.rank(responses, criteria=["helpful", "harmless", "honest"])
chosen, rejected = ranked[0], ranked[-1]
```
- 质量：中高
- 成本：中（$0.01~$0.1 每对）
- 适用：大规模数据

**3. 规则生成 rejected**：
```python
chosen = high_quality_response
rejected = apply_rules(chosen, rules=[
    "truncate_to_half",  # 截断
    "add_repetition",    # 重复
    "inject_factual_error"  # 注入错误
])
```
- 质量：低
- 成本：低
- 适用：数据增强、辅助数据

**数据量建议**：

| 算法 | RM 训练数据 | DPO 训练数据 |
|------|-------------|--------------|
| PPO | 50K ~ 200K | N/A |
| DPO | N/A | 10K ~ 100K |
| GRPO | 50K ~ 200K | N/A |

**关键点**：
- 数据质量 > 数据量
- 覆盖度：覆盖多种任务、难度、错误类型
- 平衡性：chosen/rejected 差距要适中（太简单无用，太难学不会）

### 7. Q: RM 训练有哪些坑？

**A**: 

**1. 过拟合训练数据**：
- **现象**：RM 在训练集准确率高，但泛化性差
- **解决**：增加数据多样性、正则化、early stopping

**2. 奖励黑客**：
- **现象**：模型学会"刷分"，生成 RM 给高分但无用的回答
- **解决**：
  - RM 训练数据多样化
  - 加入 hard negative
  - 多个 RM 投票
  - 规则覆盖

**3. 分布偏移**：
- **现象**：RM 在训练分布上表现好，但实际使用时分布不同
- **解决**：
  - 从实际使用分布采样训练数据
  - 对抗训练（adversarial training）
  - 在线学习（online learning）

**4. 偏见放大**：
- **现象**：RM 学习标注者的偏见（如长度偏好、风格偏好）
- **解决**：
  - 标注指南明确标准
  - 多个标注者投票
  - 偏见检测和修正

## 常见错误（至少 3 个）

### 1. 错误：Reference model 参与训练

**现象**：训练初期 loss 快速下降，但模型能力快速退化

**代码错误**：
```python
# 错误：reference model 也更新
ref_model = AutoModelForCausalLM.from_pretrained(sft_model)
# 忘记冻结参数
optimizer = AdamW(list(policy_model.parameters()) + list(ref_model.parameters()))
```

**正确做法**：
```python
ref_model = AutoModelForCausalLM.from_pretrained(sft_model)
ref_model.eval()  # 设置为评估模式
for param in ref_model.parameters():
    param.requires_grad = False  # 冻结参数
```

**原因**：Reference model 冻结是 KL 约束的前提，如果它也更新，KL 约束会失效

### 2. 错误：直接用 SFT 数据做 RLHF，跳过 RM 训练

**现象**：模型效果不如 SFT，甚至退化

**错误流程**：
```
SFT → 直接 DPO（用 SFT 数据的 chosen/rejected）
```

**正确流程**：
```
SFT → 采样新数据 → 标注偏好 → 训练 RM/DPO
```

**原因**：
- SFT 数据是"理想回答"，不知道"坏回答"长什么样
- RLHF 需要偏好对比数据，学习好坏边界
- 跳过采样和标注，模型学不到偏好

### 3. 错误：KL 系数固定，不自适应调整

**现象**：训练不稳定，KL 爆炸或模型不更新

**错误做法**：
```python
beta = 0.1  # 固定不变
for epoch in range(num_epochs):
    loss = policy_loss + beta * kl_loss
    loss.backward()
    optimizer.step()
```

**正确做法**：
```python
# 自适应 KL 控制
kl_controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0)

for epoch in range(num_epochs):
    loss = policy_loss + kl_controller.kl_coef * kl_loss
    loss.backward()
    optimizer.step()
    
    # 根据 KL 调整系数
    kl_controller.update(current_kl=kl_div)
```

**原因**：
- 训练过程中，策略会逐渐偏离 reference model
- 固定 β 无法适应 KL 的动态变化
- KL 太大要增大 β，KL 太小要减小 β

### 4. 错误：偏好数据质量差，不做过滤

**现象**：训练 loss 下降，但模型效果不提升

**错误做法**：
```python
# 直接使用所有偏好对
preference_pairs = load_all_data()  # 包含大量噪声
train_dpo(preference_pairs)
```

**正确做法**：
```python
# 过滤相似度过高的样本
def filter_preference_pairs(pairs, threshold=0.85):
    filtered = []
    for pair in pairs:
        # 计算 chosen 和 rejected 的相似度
        sim = compute_similarity(pair.chosen, pair.rejected)
        # 保留差异足够大的样本
        if sim < threshold:
            filtered.append(pair)
    return filtered

preference_pairs = filter_preference_pairs(load_all_data())
```

**原因**：
- chosen/rejected 过于相似，模型难以区分
- 噪声数据会干扰学习
- 需要清洗和过滤

## 反问面试官的问题

### 技术深度类
1. "团队在实际项目中，PPO、DPO、GRPO 是如何选择的？有尝过混合方案吗？"
2. "RM 训练数据是如何平衡多样性和质量的？有没有遇到过奖励黑客问题？"
3. "团队对偏好数据的标注流程是怎样的？人工标注和 LLM 辅助的比例是多少？"
4. "RLHF 训练过程中，KL 系数是固定还是自适应？遇到过 KL 爆炸的情况吗？"

### 业务场景类
1. "团队对齐任务的主要目标是什么？安全性、有用性还是诚实性？如何权衡？"
2. "RLHF 的迭代周期大概多久？数据收集到模型上线的完整流程是怎样的？"
3. "有没有遇到过 DPO/PPO 训练后效果不如 SFT 的情况？如何排查和解决？"

## 自测题

### 口述（能流畅讲清楚）
1. RLHF 完整流程（SFT → RM → PPO/DPO）的每个阶段目标是什么？
2. 为什么 SFT 之后还要做 RLHF？只用 SFT 的局限性是什么？
3. Reference model 的作用是什么？为什么要冻结？
4. KL 约束的意义是什么？β 如何选择？
5. PPO 和 DPO 的核心差异是什么？如何选择？

### 手写（5 分钟能写出）
1. **PPO Loss 公式**：
   - 写出完整的 PPO loss（包含 clip、KL、value loss）
   - 解释每个参数的含义

2. **DPO Loss 公式**：
   - 写出 DPO loss
   - 解释为什么是 sigmoid 形式

3. **KL 散度计算**：
   ```python
   # 给定 policy_log_probs 和 ref_log_probs
   # 写出 KL 散度的计算代码
   kl_div = ???
   ```

4. **Preference Pair 过滤逻辑**：
   ```python
   # 输入：原始偏好数据列表
   # 输出：过滤后的数据（移除 chosen/rejected 相似度过高的样本）
   def filter_preference_pairs(data, threshold=0.85):
       # 实现过滤逻辑
       pass
   ```

## 标签
#RLHF #训练 #PPO #DPO #GRPO #对齐 #SFT #奖励模型 #偏好学习 #字节 #阿里 #腾讯 #美团 #百度

## 相关文档
- [[02-PPO算法]]
- [[03-DPO算法]]
- [[04-GRPO算法]]
- [[05-偏好数据设计]]
- [[../03-SFT与微调/01-SFT基础]]