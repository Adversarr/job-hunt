# DPO 算法

## 一句话结论
DPO（Direct Preference Optimization）通过重构奖励函数，**绕过奖励模型训练**，直接用偏好数据优化策略，Loss 本质是**最大化 chosen 与 rejected 的 logit 差距**，是 RLHF 的简化版替代方案。

## 核心定义/公式

### DPO Loss 公式
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

其中：
- $y_w$（chosen）：偏好回答
- $y_l$（rejected）：非偏好回答
- $\pi_\theta$：当前策略模型
- $\pi_{\text{ref}}$：参考模型（通常是 SFT 后的模型）
- $\beta$：KL 约束系数（通常 0.1-0.5）
- $\sigma$：sigmoid 函数

### PyTorch 实现
```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logps_chosen, policy_logps_rejected, 
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    Args:
        policy_logps_chosen: [batch] 当前模型对 chosen 的 log prob
        policy_logps_rejected: [batch] 当前模型对 rejected 的 log prob
        ref_logps_chosen: [batch] 参考模型对 chosen 的 log prob
        ref_logps_rejected: [batch] 参考模型对 rejected 的 log prob
        beta: KL 约束系数
    """
    # log π(y|x) - log π_ref(y|x) = log (π/π_ref)
    chosen_logratios = policy_logps_chosen - ref_logps_chosen
    rejected_logratios = policy_logps_rejected - ref_logps_rejected
    
    # logits = β * (log_ratio_chosen - log_ratio_rejected)
    logits = beta * (chosen_logratios - rejected_logratios)
    
    # Loss = -log(sigmoid(logits))
    loss = -F.logsigmoid(logits).mean()
    
    return loss

# 实际使用：需要在 forward 时计算 log prob
def get_logprobs(model, input_ids, attention_mask, labels):
    """计算每个 token 的 log probability，只对 label 位置求和"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # shift: logits[:, :-1] 对应 labels[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # CrossEntropy: input is [N, C], target is [N]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    # Flatten: [batch * (seq_len-1), vocab_size]
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # reshape back and sum over sequence
    loss = loss.view(shift_labels.size(0), shift_labels.size(1))
    
    # 只对非 -100 的 token 求和（-100 是 ignore index）
    valid_mask = shift_labels != -100
    log_probs = -loss * valid_mask  # 负的交叉熵 = log prob
    seq_log_probs = log_probs.sum(dim=1)
    
    return seq_log_probs
```

## 为什么（2-3 个因果链）

### 1. 为什么 DPO 能绕过奖励模型？
- **RLHF 传统路径**：训练奖励模型 $r_\phi(x, y)$ → 用 PPO 最大化 $\mathbb{E}[r_\phi]$ 并加 KL 约束
- **DPO 的关键洞察**：最优奖励函数可以表示为策略比值的函数：
  $$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$$
- **因果链**：奖励模型本质是学习偏好排序 → Bradley-Terry 模型假设偏好概率与奖励差正相关 → 将奖励表示为策略比值 → 直接优化策略比值 → 无需显式奖励模型

### 2. 为什么 DPO loss 是 sigmoid 形式？
- **Bradley-Terry 模型**：chosen 优于 rejected 的概率为：
  $$P(y_w \succ y_l) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$
- **代入奖励公式**：
  $$P(y_w \succ y_l) = \sigma\left( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$
- **最大化似然**：取 log 得到 DPO loss

### 3. 为什么需要 reference model？
- **防止策略偏离太远**：如果只优化 chosen 的概率，模型可能过拟合偏好数据，丧失通用能力
- **KL 正则化的作用**：
  - $\beta$ 大：约束强，模型变化小，偏好学习弱
  - $\beta$ 小：约束弱，模型变化大，可能过拟合或灾难性遗忘
- **实践观察**：没有 reference model 的 DPO 会快速降低 loss，但模型能力退化严重

## 怎么做（可落地步骤）

### 标准做法
1. **数据准备**
   - SFT 模型作为初始化
   - 收集偏好数据：(prompt, chosen, rejected)
   - 过滤低质量对：chosen/rejected 相似度过高的样本
   - 建议：每个 prompt 准备 1-2 对偏好对，避免单一 prompt 多对（可能导致过拟合）

2. **训练流程**
   ```python
   # Step 1: 加载模型
   model = AutoModelForCausalLM.from_pretrained(sft_model_path)
   ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
   ref_model.eval()  # 冻结，不训练
   
   # Step 2: 计算 log probabilities
   with torch.no_grad():
       ref_chosen_logps = get_logprobs(ref_model, chosen_ids, chosen_mask, chosen_labels)
       ref_rejected_logps = get_logprobs(ref_model, rejected_ids, rejected_mask, rejected_labels)
   
   policy_chosen_logps = get_logprobs(model, chosen_ids, chosen_mask, chosen_labels)
   policy_rejected_logps = get_logprobs(model, rejected_ids, rejected_mask, rejected_labels)
   
   # Step 3: 计算 loss
   loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                   ref_chosen_logps, ref_rejected_logps, beta=0.1)
   
   # Step 4: 反向传播
   loss.backward()
   optimizer.step()
   ```

3. **训练技巧**
   - **学习率**：DPO 通常用较小学习率（1e-6 到 5e-7），比 SFT 小 10x
   - **Batch size**：偏好对数量少时，可以用 gradient accumulation
   - **Early stopping**：监控 validation set 的 chosen/rejected reward gap，gap 稳定后停止

### 关键配置/参数
| 参数 | 推荐值 | 原因 |
|------|--------|------|
| $\beta$（KL 系数） | 0.1-0.5 | 0.1 为通用起点，数据质量高可调至 0.3-0.5 |
| 学习率 | 1e-6 | DPO 损失曲面较平坦，需要稳定优化 |
| reference model 更新 | 冻结 | 避免 moving target 问题 |
| batch size | 32-128 | 视显存而定，但每个 batch 要有足够多样性 |
| 数据量 | 10K-100K 对 | 数据质量 > 数据量 |

### 多轮对话 DPO 数据构造（字节真题）

**问题**：如何将 DPO 应用于"第一轮对话"？

**策略**：
```python
# 方案 1: 单轮独立处理
data = {
    "prompt": "请介绍一下北京",
    "chosen": "北京是中国的首都，位于华北平原...",  # 事实准确、信息丰富
    "rejected": "北京是个很大的城市。"  # 信息量少、模糊
}

# 方案 2: 多轮采样（重点：每轮独立构造偏好对）
conversation = [
    {"role": "user", "content": "介绍一下北京"},
    {"role": "assistant", "content": "北京是中国的首都..."},
    {"role": "user", "content": "有哪些著名景点？"},
    {"role": "assistant", "content": "故宫、长城、颐和园..."},
]

# 采样策略：
# 1. 按轮次采样：早轮/中轮/末轮权重 = [0.3, 0.4, 0.3]
# 2. 按错误类型采样：事实错误/指令偏离/安全违规 = [0.4, 0.4, 0.2]
# 3. 单轮和多轮配比：单轮 60%, 多轮 40%（单轮更可控，多轮考验上下文理解）

# 第一轮对话特殊处理：
# - 保留 system prompt（定义角色和约束）
# - chosen: 遵循 system 指令的回答
# - rejected: 忽略 system 指令的回答
```

**关键点**：
- 第一轮对话要特别关注 system prompt 遵循度
- 多轮对话要控制 exposure bias（避免只采样最后一轮）

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **DPO** | 无需奖励模型，实现简单，训练稳定 | 数据质量要求高，无法在线探索新策略 | 偏好数据质量高，预算有限 |
| **PPO** | 在线学习，可探索更优策略，上限更高 | 需要奖励模型，训练不稳定，实现复杂 | 有充足算力和标注资源 |
| **SFT only** | 最简单，收敛快 | 无法对齐偏好，上限受限 | 数据量小，只需格式对齐 |
| **DPO → SFT** | 先对齐偏好，再强化格式 | 可能破坏偏好对齐效果 | 不推荐，顺序应相反 |
| **SFT → DPO** | 先建立基础能力，再对齐偏好 | 需要两阶段训练 | **推荐标准流程** |

### DPO 的核心缺点（美团真题）
1. **数据质量敏感**：偏好数据噪声直接影响效果
   - 应对：严格的数据清洗，过滤相似度过高的 chosen/rejected
2. **无法在线探索**：只能学习已有偏好数据中的模式
   - 应对：定期收集新数据迭代，或结合 PPO 做 online learning
3. **分布偏移问题**：训练数据分布与实际使用分布可能不匹配
   - 应对：多样化数据收集，覆盖不同场景和难度
4. **奖励模型缺失的副作用**：无法显式建模奖励，难以调试
   - 应对：训练后用奖励模型评估，监控 chosen/rejected reward gap

## 高频追问（≥5 个）

### 1. Q: DPO 和 SFT 的区别？先 DPO 后 SFT 可以吗？（字节一面）
**A**: 
- **目标差异**：SFT 拟合参考答案（模仿学习），DPO 学习偏好排序（相对比较）
- **数据差异**：SFT 需要 (prompt, response) 对，DPO 需要 (prompt, chosen, rejected) 三元组
- **Loss 差异**：SFT 是 cross-entropy loss，DPO 是 preference loss
- **顺序问题**：**不能先 DPO 后 SFT**
  - SFT 是建立基础能力（格式、风格、指令遵循）
  - DPO 是在已有能力基础上微调偏好
  - 反序会导致：DPO 学到的偏好被 SFT 覆盖，模型回到参考答案分布

### 2. Q: DPO 是 on-policy 还是 off-policy？（美团）
**A**: 
- **理论口径**：DPO 是 **off-policy**
  - 数据来源：可以是历史数据、其他模型生成、人工标注（非当前策略采样）
  - 更新方式：用固定数据集更新，不在线采样新数据
- **实现现实**：实践中可以 **online DPO**
  - 每 N 步更新 reference model
  - 用当前策略生成新的 chosen/rejected
  - 本质是 iterative DPO，但更接近 on-policy
- **对比**：
  - **On-policy**: PPO（在线采样，实时更新）
  - **Off-policy**: DPO（离线数据，批量更新）
  - **Online**: PPO、online DPO（实时交互）
  - **Offline**: 标准 DPO（静态数据集）

### 3. Q: DPO 训练时 chosen/rejected reward 都下降，如何解释和应对？（美团）
**A**: 

**现象解释**：
```
训练过程中观察到：
- chosen_reward 从 -1.2 降到 -1.5
- rejected_reward 从 -2.5 降到 -3.0
- reward_gap 保持或略增（0.3 → 1.5）
```

**原因分析**：
1. **Reference 模型偏置**：
   - 如果 reference model 本身对 chosen/rejected 的 log prob 都很低
   - 训练时模型会整体降低 log prob（因为 loss 只关注差值）
   - **排查**：检查 reference model 的 log prob 分布

2. **Logit margin 问题**：
   - DPO loss 只优化 $\pi_\theta(y_w) - \pi_\theta(y_l)$ 的差值
   - 不约束单独的概率值 → 两者可以同时下降
   - **根本原因**：sigmoid(100) ≈ sigmoid(1000)，差值足够大后 loss 饱和

3. **数据噪声**：
   - chosen/rejected 质量接近 → 模型难以区分
   - 过拟合噪声 → 同时降低两者的概率作为"保守策略"
   - **排查**：计算 chosen/rejected 的相似度（BLEU/ROUGE/embedding cos）

**应对策略**：
1. **监控指标**：
   ```python
   # 不仅监控 loss，还要监控：
   chosen_reward = (policy_chosen_logps - ref_chosen_logps).mean()
   rejected_reward = (policy_rejected_logps - ref_rejected_logps).mean()
   reward_gap = chosen_reward - rejected_reward
   
   # 健康状态：
   # - chosen_reward 应略降或持平
   # - rejected_reward 应显著降低
   # - reward_gap 应增大
   ```

2. **调整策略**：
   - **增加 β**：加强 KL 约束，防止整体 log prob 下降
   - **数据清洗**：过滤相似度 > 0.8 的偏好对
   - **Early stopping**：当 reward_gap 达到阈值（如 2.0）就停止
   - **Learning rate warmup**：避免初期梯度爆炸

3. **代码级修复**：
   ```python
   # 方案 1: 添加 absolute reward 约束
   loss = dpo_loss + 0.1 * (-chosen_logratios).mean()  # 惩罚 chosen 过低
   
   # 方案 2: 使用 average log likelihood 而非 sum
   # 避免长文本的 log prob 累积效应
   seq_log_probs = log_probs.mean(dim=1)  # 平均而非求和
   ```

### 4. Q: DPO 数据量不够怎么构造？质量不好怎么改善？（百度）
**A**: 
**数据扩充策略**：
1. **LLM 自举**：
   ```python
   # 用 GPT-4 等强模型生成偏好对
   prompt = "写一段关于AI的介绍"
   responses = model.generate(prompt, n=5, temperature=0.8)
   # 用 GPT-4 标注偏好排序
   ranked = gpt4.rank(responses, criteria=["准确性", "流畅性", "信息量"])
   chosen, rejected = ranked[0], ranked[-1]
   ```

2. **对抗生成**：
   ```python
   # 用现有模型生成"好的"回答
   chosen = sft_model.generate(prompt, temperature=0.7)
   # 故意注入错误，生成"坏的"回答
   rejected = inject_errors(chosen, error_type="factual_hallucination")
   ```

3. **规则化 rejection**：
   ```python
   # Chosen: 高质量回答
   # Rejected: 应用规则生成（重复、过短、跑题、格式错误等）
   rejected_templates = [
       "我不知道。",
       chosen[:len(chosen)//3],  # 截断
       "这个问题很难回答。" * 5,  # 重复
   ]
   ```

**质量改善（不使用人工）**：
1. **Self-consistency 过滤**：
   ```python
   # 多次生成，只保留一致性高的样本
   responses = [model.generate(prompt) for _ in range(5)]
   if pairwise_similarity(responses) > 0.8:  # 高一致性
       chosen = majority_vote(responses)
   ```

2. **Difficulty-based sampling**：
   ```python
   # 用奖励模型打分，保留中等难度的偏好对
   scores = [reward_model(prompt, r) for r in responses]
   # 保留 chosen_score - rejected_score 在 [1.0, 3.0] 的样本
   # 过简单（差距大）：模型已学会
   # 过难（差距小）：数据噪声或质量问题
   ```

3. **Iterative DPO**：
   ```python
   # Round 1: 用初始数据训练 DPO
   model_v1 = train_dpo(data_v0)
   # Round 2: 用 model_v1 生成新数据
   data_v1 = generate_preference_data(model_v1)
   model_v2 = train_dpo(data_v1)
   ```

### 5. Q: 为什么 DPO 不能像 PPO 那样探索新策略？
**A**: 
- **PPO 的探索机制**：
  - 策略模型生成多种回答
  - 奖励模型对这些回答打分
  - 策略根据奖励信号调整 → 可以发现训练数据中没有的高奖励策略
  
- **DPO 的限制**：
  - 只学习已有偏好对中的 chosen vs rejected
  - 无法知道"比 chosen 更好的回答"长什么样
  - **例子**：训练数据中 chosen 是"中等长度"，rejected 是"过短"
    - DPO 只学到"不要太短"
    - 但不知道"详细展开"可能是更优策略

- **解决方案**：
  - Iterative DPO：定期收集新数据
  - 结合 PPO：DPO 快速对齐基础偏好，PPO 精细调优
  - 数据多样性：确保偏好数据覆盖多种"好"的回答风格

### 6. Q: 写一下 DPO 和 PPO 的 Loss 表达式，对比差异（阿里通义）
**A**: 

**DPO Loss**:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right]$$

**PPO Loss**:
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right] - \beta \mathbb{E}\left[\text{KL}(\pi_\theta || \pi_{\text{ref}})\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$，$\hat{A}_t$ 是优势函数（通过奖励模型计算）

**关键差异**：
| 维度 | DPO | PPO |
|------|-----|-----|
| 奖励来源 | 隐式（策略比值差） | 显式奖励模型 |
| 数据需求 | 静态偏好对 | 在线采样 + 奖励标注 |
| 优化目标 | 最大化偏好概率 | 最大化奖励 + KL 约束 |
| 探索能力 | 无（只学习已有数据） | 有（可以生成新策略） |
| 实现复杂度 | 低（单模型训练） | 高（需要奖励模型 + 策略模型 + 价值模型） |
| 训练稳定性 | 高 | 低（需要调 PPO 超参） |

### 7. Q: 如何选择 β（KL 系数）？
**A**: 
- **β 的作用**：控制策略偏离 reference model 的程度
- **选择策略**：
  - **数据质量高**（chosen 明显优于 rejected）：β = 0.3-0.5，允许更大偏离
  - **数据质量中**：β = 0.1，保守起点
  - **数据质量低**（噪声多）：β = 0.05-0.1，强约束防止过拟合
  
- **调试方法**：
  ```python
  # 监控 KL 散度
  kl_chosen = (policy_logps_chosen - ref_logps_chosen).mean()
  kl_rejected = (policy_logps_rejected - ref_logps_rejected).mean()
  
  # 健康范围：KL 在 [0.5, 5.0] nat
  # KL 过大（>10）：β 太小，模型偏离太远，可能遗忘
  # KL 过小（<0.1）：β 太大，模型几乎不变，偏好学习不足
  ```

- **经验值**：
  - 通用场景：β = 0.1
  - 对齐强度要求高：β = 0.05
  - 数据质量极高：β = 0.3

## 常见错误（≥3 个）

### 1. **错误：使用平均 log prob 而非求和**
**现象**：对长文本效果变差
```python
# 错误做法
seq_log_probs = log_probs.mean(dim=1)  # 平均

# 正确做法
seq_log_probs = log_probs.sum(dim=1)  # 求和（整个序列的 log prob）
```
**原因**：DPO 公式基于完整序列的 log prob，平均化会丢失序列长度信息，导致长文本惩罚不足

**特殊情况**：如果 chosen 和 rejected 长度差异极大（>2倍），可以考虑归一化：
```python
seq_log_probs = log_probs.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
```

### 2. **错误：Reference model 参与训练**
**现象**：训练初期 loss 快速下降，但模型能力退化
```python
# 错误做法
for param in ref_model.parameters():
    param.requires_grad = True  # Reference model 也更新

# 正确做法
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
```
**原因**：如果 reference model 也更新，会形成 moving target，训练不稳定

### 3. **错误：忽略 chosen/rejected 相似度过滤**
**现象**：训练集 loss 下降但验证集效果差
```python
# 错误做法：直接使用所有偏好对
train_data = all_preference_pairs  # 包含大量相似样本

# 正确做法：过滤相似度过高的样本
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_similar_pairs(data, threshold=0.85):
    filtered = []
    for item in data:
        emb_chosen = model.encode(item['chosen'])
        emb_rejected = model.encode(item['rejected'])
        similarity = cosine_similarity(emb_chosen, emb_rejected)
        if similarity < threshold:  # 保留差异足够大的样本
            filtered.append(item)
    return filtered
```
**原因**：chosen/rejected 过于相似，模型无法学到有意义的偏好差异，反而学到了噪声

### 4. **错误：学习率过大导致训练崩塌**
**现象**：loss 突然变 NaN 或爆炸
```python
# 错误做法
optimizer = AdamW(model.parameters(), lr=1e-4)  # 过大

# 正确做法
optimizer = AdamW(model.parameters(), lr=1e-6)  # DPO 通常用 1e-6 到 5e-7
```
**原因**：DPO 损失曲面对长文本很敏感，大学习率会导致 log prob 梯度爆炸

## 反问面试官的问题

### 技术深度类
1. "团队在实际项目中，DPO 数据收集是如何平衡成本和质量的？有没有尝试过 LLM 自举或对抗生成？"
2. "对于 chosen/rejected reward 都下降的情况，团队有没有积累过特殊的调试经验或监控指标？"
3. "团队是倾向于纯 DPO，还是 DPO + PPO 的混合方案？在什么场景下会切换？"

### 业务场景类
1. "团队对齐任务中，最常见的偏好类型是什么？是事实准确性、安全性还是风格对齐？"
2. "DPO 迭代周期大概多久？有没有遇到过数据分布偏移导致模型退化的问题？"
3. "团队有没有尝试过 online DPO 或 iterative DPO？效果如何？"

## 自测题

### 口述（能流畅讲清楚）
1. DPO 如何绕过奖励模型，直接优化偏好？
2. 为什么 DPO loss 是 sigmoid 形式？从 Bradley-Terry 模型推导
3. DPO 数据构造时，如何处理多轮对话？单轮和多轮配比原则是什么？
4. Chosen/rejected reward 都下降的原因有哪些？如何排查？

### 手写（5 分钟能写出）
1. **DPO Loss 函数**（PyTorch 实现）
   - 输入：policy log probs, reference log probs
   - 输出：loss 值
   - 关键：log ratio 计算 + sigmoid

2. **监控指标计算**
   ```python
   # 给定 batch 的 log probs，计算：
   # - chosen_reward, rejected_reward
   # - reward_gap
   # - KL divergence
   ```

3. **偏好数据过滤逻辑**
   ```python
   # 输入：原始偏好数据
   # 输出：过滤后的数据
   # 要求：移除 chosen/rejected 相似度过高的样本
   ```

## 标签
#DPO #RLHF #训练 #偏好学习 #对齐 #字节 #美团 #阿里 #腾讯 #百度 #handwrite #derive

## 相关文档
- [[01-RLHF总览]]
- [[02-PPO算法]]
- [[04-GRPO算法]]
- [[05-偏好数据设计]]
- [[06-奖励黑客]]
- [[../03-SFT与LoRA/01-SFT目标与边界]]
