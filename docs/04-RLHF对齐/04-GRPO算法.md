# GRPO算法：为什么不用PPO

## 一句话结论
GRPO（Group Relative Policy Optimization）通过组内相对比较替代独立奖励模型，解决了PPO奖励模型偏差累积、训练不稳定、流程复杂三大痛点，代价是采样开销增加。

## 核心定义/公式

### GRPO 核心思想
对每个prompt生成G个响应，通过组内相对比较计算优势函数，无需独立奖励模型：

```
# GRPO 相对优势计算
对于每个prompt x，采样G个响应 {y_1, ..., y_G}
计算相对优势: A_i = (r_i - mean(r)) / std(r)
其中 r_i 可以是基于规则、模型或人工的评分

优化目标:
L_GRPO = E[Σ log π(y_i|x) · A_i]
```

### PPO vs GRPO 对比

**PPO Loss**:
```python
# PPO需要三步：RM训练 → 策略优化 → KL约束
# 1. 奖励模型损失
L_RM = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]

# 2. PPO策略损失
ratio = π_θ(y|x) / π_ref(y|x)
L_PPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)

# 其中优势函数 A 需要从RM获取奖励，再用GAE计算
```

**GRPO Loss**:
```python
# GRPO一步到位：组内相对比较
def grpo_loss(policy_model, prompts, group_size=4):
    losses = []
    for prompt in prompts:
        # 采样G个响应
        responses = [policy_model.generate(prompt) for _ in range(group_size)]
        
        # 获取评分（可以是模型、规则或混合）
        scores = get_scores(prompt, responses)  # [G]
        
        # 组内归一化计算相对优势
        advantages = (scores - scores.mean()) / (scores.std() + 1e-8)
        
        # 策略梯度
        for i, resp in enumerate(responses):
            log_prob = policy_model.log_prob(prompt, resp)
            losses.append(-log_prob * advantages[i])
    
    return torch.stack(losses).mean()
```

### 关键公式差异

| 组件 | PPO | GRPO |
|------|-----|------|
| 奖励来源 | 独立训练的RM模型 | 组内相对比较（隐式奖励） |
| 优势函数 | RM奖励 + GAE | 组内归一化 (r - μ) / σ |
| KL约束 | 显式KL penalty | 通过组内竞争隐式约束 |
| Reference模型 | 必需 π_ref | 可选（GRPO中reference用于重要性采样稳定性） |

## 为什么（2-3 个因果链）

### 1. **奖励模型偏差累积 → 训练不稳定 → PPO容易崩溃**

**因果链**:
- **现象**: PPO训练中RM偏差被放大
  - RM在训练数据分布上可能有偏（对某些pattern过拟合）
  - Policy model优化时不断利用这些偏差
  - RM和Policy交替训练，偏差累积放大
  
- **根因**: RM与Policy的"对抗失衡"
  - RM是静态的（训练后固定或很少更新）
  - Policy动态探索，容易找到RM的漏洞
  - 没有机制让RM及时纠偏
  
- **结果**: 
  - Policy学会"欺骗"RM（奖励黑客）
  - KL约束过紧则学不动，过松则偏离太远
  - 训练曲线不稳定，需要大量调参

**GRPO解决方案**:
```
GRPO通过组内相对比较，避免了RM偏差问题：
- 每个prompt的G个响应在同组内比较
- 好坏是相对的（相比组内其他响应）
- 不依赖绝对奖励值，只看相对排序
- 天然抗偏：即使评分器有偏，组内比较时偏差被抵消
```

### 2. **PPO实现复杂度高 → 工程落地难 → 多个易错环节**

**因果链**:
- **现象**: PPO需要多阶段训练流水线
  - Stage 1: SFT → Stage 2: 训练RM → Stage 3: PPO优化
  - 每个阶段都有超参数、数据、模型要管理
  
- **根因**: 解耦带来的复杂性
  - RM和Policy是两个独立模型
  - 需要同步数据流、checkpoint、训练状态
  - GAE计算、价值函数估计、KL调度都是易错点
  
- **结果**:
  - 训练pipeline长，调试困难
  - 任何一个环节出问题都影响最终效果
  - 工程成本高，小团队难以维护

**GRPO简化**:
```python
# GRPO将RM隐式化，简化流程
# 传统PPO: SFT → RM → PPO（3步）
# GRPO: SFT → GRPO（2步，无需单独训练RM）

# 工程上：
# - 少一个模型要训练和保存
# - 少一份训练数据标注（RM偏好数据）
# - 少一套超参数调优流程
```

### 3. **采样效率权衡 → GRPO用采样换稳定性**

**因果链**:
- **现象**: GRPO每个prompt需要生成G个响应（通常G=4~8）
  - PPO每步生成1个响应
  - GRPO每步生成G个响应
  
- **根因**: 相对比较需要多样本
  - 组内比较的可靠性依赖于样本多样性
  - G太小：相对优势估计不准
  - G太大：计算成本高
  
- **结果**:
  - 采样成本是PPO的G倍
  - 但换来了无需训练RM的开销
  - 总体权衡：GRPO可能更高效（省去RM训练 + 更稳定）

**权衡分析**:
```
设PPO的RM训练成本 = C_RM
设PPO每步采样成本 = 1, GRPO每步采样成本 = G
设训练步数为N

PPO总成本 ≈ C_RM + N × 1
GRPO总成本 ≈ N × G

当 C_RM > N × (G-1) 时，GRPO更优
即 RM训练成本 > (G-1)倍的训练步数采样成本

实际情况：
- RM训练需要大量偏好数据标注（人工/API成本）
- RM训练本身需要多个epoch
- G通常取4-8，是可控开销
```

## 怎么做（可落地步骤）

### 标准 GRPO 训练流程

#### Step 1: 数据准备
```python
# GRPO数据格式（比PPO简单）
{
    "prompt": "请解释什么是机器学习",
    # 不需要chosen/rejected pairs
    # 只需要prompt，由模型生成多个response
}

# 可选：预定义评分规则
# - 规则评分（长度、格式、关键词）
# - 模型评分（用另一个LLM打分）
# - 混合评分
```

#### Step 2: 配置GRPO训练参数
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# GRPO配置
grpo_config = {
    # 采样相关
    "group_size": 4,  # G: 每个prompt生成几个响应（4-8常用）
    "temperature": 1.0,  # 采样温度，保证多样性
    "top_p": 0.95,
    "max_new_tokens": 512,
    
    # 训练相关
    "learning_rate": 1e-6,  # GRPO通常用较小lr
    "batch_size": 8,  # prompts per batch
    "gradient_accumulation_steps": 4,
    
    # 评分器配置
    "scorer_type": "model",  # "rule" | "model" | "hybrid"
    "scorer_model": "reward-model-v1",  # 如果用模型评分
    
    # 稳定性相关
    "clip_range": 0.2,  # 虽然GRPO主要靠组内归一化，仍可加clip
    "reference_free": False,  # 是否完全不用reference模型
}
```

#### Step 3: 实现GRPO核心训练循环
```python
import torch
import torch.nn.functional as F

class GRPOTrainer:
    def __init__(self, policy_model, ref_model=None, config=None):
        self.policy = policy_model
        self.ref = ref_model if ref_model else policy_model  # 可以不用独立ref
        self.config = config or {}
        
    def compute_advantages(self, scores):
        """组内相对优势计算"""
        # scores: [batch_size, group_size]
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True) + 1e-8
        advantages = (scores - mean) / std
        return advantages
    
    def generate_group_responses(self, prompts, group_size):
        """为每个prompt生成G个响应"""
        batch_size = len(prompts)
        all_responses = []
        all_log_probs = []
        
        for g in range(group_size):
            # 并行生成一个group的响应
            outputs = self.policy.generate(
                prompts,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                temperature=self.config.get("temperature", 1.0),
                top_p=self.config.get("top_p", 0.95),
                do_sample=True,
            )
            all_responses.append(outputs)
            
            # 计算log prob（用于策略梯度）
            log_prob = self.compute_log_prob(prompts, outputs)
            all_log_probs.append(log_prob)
        
        # shape: [batch_size, group_size]
        return all_responses, torch.stack(all_log_probs, dim=1)
    
    def compute_log_prob(self, prompts, responses):
        """计算策略模型对响应的log概率"""
        inputs = self.policy.tokenizer(
            [p + r for p, r in zip(prompts, responses)],
            return_tensors="pt",
            padding=True,
        ).to(self.policy.device)
        
        with torch.no_grad():
            outputs = self.policy(**inputs, labels=inputs["input_ids"])
            # 只计算response部分的log prob
            log_prob = -outputs.loss  # 负的cross entropy
        return log_prob
    
    def score_responses(self, prompts, responses):
        """评分器：可以是规则、模型或混合"""
        scorer_type = self.config.get("scorer_type", "model")
        
        if scorer_type == "rule":
            # 规则评分示例
            scores = []
            for resp in responses:
                score = 0.0
                if len(resp) > 50: score += 1.0  # 长度奖励
                if "例如" in resp: score += 0.5  # 举例奖励
                scores.append(score)
            return torch.tensor(scores)
        
        elif scorer_type == "model":
            # 用RM或其他LLM打分
            scores = self.scorer_model.score(prompts, responses)
            return scores
        
        else:  # hybrid
            rule_scores = self.rule_scorer(responses)
            model_scores = self.model_scorer(prompts, responses)
            return 0.3 * rule_scores + 0.7 * model_scores
    
    def train_step(self, prompts):
        """单步GRPO训练"""
        group_size = self.config.get("group_size", 4)
        
        # 1. 生成G个响应
        responses, log_probs = self.generate_group_responses(prompts, group_size)
        # log_probs: [batch, group_size]
        
        # 2. 评分（这里可以并行化）
        scores = self.score_responses(prompts, responses)
        # scores: [batch, group_size]
        
        # 3. 计算相对优势
        advantages = self.compute_advantages(scores)
        # advantages: [batch, group_size]
        
        # 4. 策略梯度损失
        # GRPO核心：用相对优势加权的log prob
        loss = -(log_probs * advantages).mean()
        
        # 5. 可选：重要性采样比率clip（增强稳定性）
        if self.config.get("use_importance_sampling", False):
            with torch.no_grad():
                ref_log_probs = self.compute_ref_log_prob(prompts, responses)
            ratio = torch.exp(log_probs - ref_log_probs)
            clip_range = self.config.get("clip_range", 0.2)
            ratio_clipped = torch.clamp(ratio, 1-clip_range, 1+clip_range)
            loss = -torch.min(
                ratio * advantages,
                ratio_clipped * advantages
            ).mean()
        
        return loss, {
            "loss": loss.item(),
            "mean_score": scores.mean().item(),
            "score_std": scores.std().item(),
        }
    
    def train(self, train_data, num_epochs=3):
        """完整训练循环"""
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.get("learning_rate", 1e-6)
        )
        
        for epoch in range(num_epochs):
            for batch_prompts in train_data:
                loss, metrics = self.train_step(batch_prompts)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, "
                      f"Mean Score: {metrics['mean_score']:.2f}")

# 使用示例
model = AutoModelForCausalLM.from_pretrained("model_path")
trainer = GRPOTrainer(model, config=grpo_config)
trainer.train(train_prompts, num_epochs=3)
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| group_size (G) | 4-8 | G太小相对优势不准，G太大计算成本高；4是平衡点 |
| temperature | 1.0-1.2 | GRPO依赖组内多样性，温度不能太低 |
| learning_rate | 5e-7 ~ 1e-6 | GRPO相对稳定，可以用较小lr防止过拟合 |
| batch_size | 8-16 | 每个batch有G个响应，实际处理的样本数是batch×G |
| scorer_type | "model" | 规则评分易被hack，模型评分更robust |

### 评分器设计

**选项1: 独立奖励模型**
```python
# 用预训练的RM或另一个LLM
rm = AutoModelForSequenceClassification.from_pretrained("rm_path")
scores = rm(prompt + response)  # [batch, G]
```

**选项2: 规则评分（快速实验）**
```python
def rule_scorer(response):
    score = 0.0
    # 长度合理性
    if 50 < len(response) < 500:
        score += 1.0
    # 格式规范
    if response.endswith("。"):
        score += 0.5
    # 关键词覆盖
    if any(kw in response for kw in ["因此", "所以", "例如"]):
        score += 0.5
    return score
```

**选项3: LLM-as-Judge**
```python
# 用GPT-4等强模型打分
prompt_template = """
请对以下回答打分（0-10分）：
问题：{question}
回答：{answer}
评分标准：准确性、完整性、清晰度
"""
scores = llm_judge.score(prompt_template, prompts, responses)
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **GRPO** | • 无需训练独立RM<br>• 训练更稳定（无RM偏差累积）<br>• 实现简单，易于调试<br>• 抗奖励hack能力强 | • 每步采样成本G倍<br>• 需要可靠的评分器<br>• 组内比较可能引入噪声 | • 有足够算力支持多次采样<br>• 训练稳定性优先<br>• 小团队快速迭代 |
| **PPO** | • 单次采样效率高<br>• 理论基础成熟<br>• 开源实现多 | • 需要训练RM（数据+计算）<br>• 训练不稳定，需要大量调参<br>• RM偏差易被放大<br>• 实现复杂 | • 有高质量偏好数据<br>• 有丰富RL调优经验<br>• 追求SOTA效果 |
| **DPO** | • 最简单，无需采样<br>• 训练快，稳定性好<br>• 无需RM | • 依赖数据质量（需要chosen/rejected pairs）<br>• 难以处理复杂偏好<br>• off-policy局限 | • 有高质量偏好对数据<br>• 简单对齐任务<br>• 快速实验验证 |

### DeepSeek 选择 GRPO 的权衡

**改了什么**:
1. 移除独立奖励模型训练阶段
2. 用组内相对比较替代绝对奖励
3. 简化训练pipeline：SFT → GRPO（2阶段）

**解决了什么**:
1. **RM训练成本**: 省去大规模偏好数据标注和RM训练
2. **训练稳定性**: 无RM偏差累积，训练曲线平滑
3. **工程复杂度**: 少一个模型、少一套流程、少一批超参数
4. **奖励黑客**: 组内相对比较天然抗hack

**代价是什么**:
1. **采样开销**: 每步生成G个响应（4-8倍计算）
2. **评分器依赖**: 需要可靠的评分机制（模型或规则）
3. **理论成熟度**: GRPO较新，开源实现和最佳实践较少

**DeepSeek的实际考量**:
```
DeepSeek-V3技术报告披露：
- 模型规模大（671B MoE），训练稳定性至关重要
- 有充足算力支持多响应采样
- 更关注训练稳定性而非极致采样效率
- 工程简洁性降低团队维护成本
```

## 高频追问（至少 5 个）

### 1. Q: GRPO 的 group_size 设多少合适？为什么？
**A**: 推荐4-8。原因：
- **G太小（如2）**: 组内比较不稳定，方差大，优势估计不准
- **G太大（如16+）**: 采样成本过高，边际收益递减
- **经验值**: DeepSeek用4，其他实践表明4-8性价比最优
- **权衡**: G=4时，采样成本可接受，相对优势估计足够准

### 2. Q: GRPO 的评分器怎么选？用规则会不会被hack？
**A**: 分场景：
- **快速实验**: 规则评分（长度、格式、关键词）—— 易hack但快
- **生产环境**: 模型评分（独立RM或LLM-as-Judge）—— 更robust
- **混合策略**: 0.7×模型分 + 0.3×规则分 —— 平衡效果和成本
- **防hack**: GRPO本身抗hack能力强（组内相对比较，不追求绝对高分）

### 3. Q: GRPO 和 PPO 的采样效率具体差多少？
**A**: 定量对比：
- **PPO**: 每步生成1个响应 × N步 = N次采样
- **GRPO**: 每步生成G个响应 × N步 = G×N次采样
- **但**: GRPO省去了RM训练的采样（RM训练需要M个偏好对，每个对要采样多次）
- **综合**: 当 RM训练成本 > (G-1)×策略训练采样成本 时，GRPO总成本更低
- **实际**: RM训练通常需要数万偏好对 + 多轮训练，成本往往高于GRPO额外采样

### 4. Q: GRPO 相比 PPO 训练稳定性如何体现？
**A**: 具体指标：
- **Loss曲线**: GRPO更平滑，PPO易震荡（RM偏差 + KL调度敏感）
- **KL散度**: GRPO天然控制（组内竞争），PPO需要显式KL penalty调参
- **奖励hacking**: GRPO通过相对比较降低，PPO常见（policy学会骗RM）
- **超参数敏感性**: GRPO对lr、KL系数等更鲁棒，PPO需要精细调参
- **训练成功率**: GRPO几乎不掉，PPO需要多次尝试找到稳定配置

### 5. Q: GRPO 能完全替代 PPO 吗？有什么局限？
**A**: 不能完全替代，各有适用场景：

**GRPO局限**:
1. **需要可靠评分器**: 如果评分器质量差，GRPO效果受限
2. **采样开销大**: 算力受限时不适用
3. **复杂偏好难建模**: 某些场景需要精细的奖励塑形，RM更灵活
4. **理论支撑不足**: GRPO较新，最优实践和理论分析不如PPO成熟

**PPO仍有优势**:
1. **单步采样效率高**: 算力紧张时更优
2. **奖励塑形灵活**: 可以手工设计复杂奖励函数
3. **社区支持完善**: 开源实现、tricks、调参经验丰富

**选择建议**:
- 大模型训练、追求稳定性 → GRPO
- 算力受限、简单任务 → DPO
- 需要精细奖励控制 → PPO

### 6. Q: DeepSeek 具体是怎么实现 GRPO 的？有技术细节吗？
**A**: DeepSeek-V3技术报告要点：
1. **Group size = 4**: 平衡效果和效率
2. **评分器**: 用独立的Reward Model（但不同于PPO的RM训练方式）
   - DeepSeek的RM在GRPO过程中可以动态更新
   - 避免了PPO中RM固定的偏差累积问题
3. **与MoE架构结合**: GRPO的稳定性对MoE训练至关重要
4. **KL处理**: 通过组内竞争隐式控制，而非显式KL penalty
5. **工程优化**: 并行生成G个响应，减少wall-clock时间

### 7. Q: GRPO 中的"相对"是什么意思？为什么有效？
**A**: "相对"的含义：
- **传统方法**: 绝对奖励 r(x, y)，模型学习"这个回答好不好"
- **GRPO相对**: 组内优势 A_i = (r_i - μ) / σ，模型学习"这个回答比组内其他回答好多少"

**为什么有效**:
1. **消除偏置**: 评分器系统性偏差（如偏好长文本）在组内归一化时被抵消
2. **稳定学习**: 不追求绝对高分，只追求相对更好，避免奖励hack
3. **自适应难度**: 组内样本质量自动定义"好"的标准，无需手工设定阈值

**类比**: 就像考试排名比绝对分数更能反映学生水平（试卷难易会影响绝对分，但不影响排名）

### 8. Q: GRPO 适合多轮对话训练吗？怎么处理？
**A**: 适合，但需要特殊处理：

**多轮GRPO挑战**:
- 每轮对话生成都需要G个候选
- 计算成本指数级增长

**解决方案**:
```python
# 方案1: 只对最后一轮做GRPO
# 前面轮次用SFT或直接生成，最后一轮采样G个候选
def multi_turn_grpo(conversation_history, group_size=4):
    # 前N-1轮：正常生成
    for turn in conversation_history[:-1]:
        response = model.generate(turn)
    
    # 最后一轮：生成G个候选
    last_turn = conversation_history[-1]
    candidates = [model.generate(last_turn) for _ in range(group_size)]
    scores = scorer(candidates)
    advantages = normalize(scores)
    # GRPO loss...
```

```python
# 方案2: 分层采样（降低成本）
# 关键轮次用GRPO，次要轮次用单样本
def hierarchical_grpo(conversation, critical_turns=[2, 4]):
    for i, turn in enumerate(conversation):
        if i in critical_turns:
            # GRPO: 生成G个候选
            responses = sample_group(turn, G=4)
        else:
            # 单样本：生成1个
            responses = [model.generate(turn)]
```

**DeepSeek实践**: 多轮对话主要用SFT，GRPO用于关键能力对齐（如数学、代码）

## 常见错误（至少 3 个）

### 1. **错误1: Group size 设置过小或过大**
**描述**: 
- G=2: 组内比较方差大，训练不稳定
- G=16+: 计算成本过高，且边际收益递减

**正确做法**:
```python
# 推荐配置
group_size = 4  # DeepSeek的选择，性价比最优

# 如果算力充足且追求效果
group_size = 8  # 更稳定的优势估计

# 如果算力紧张
group_size = 4  # 不建议更小
batch_size = 4  # 用batch_size补偿
```

### 2. **错误2: 评分器质量差导致GRPO效果不好**
**描述**:
- 用简单规则评分，模型学会hack规则
- 评分器有明显偏置（如偏好长文本）
- 评分器本身不稳定（相同输入不同分数）

**正确做法**:
```python
# 1. 评分器验证：确保打分稳定
for _ in range(5):
    score = scorer(prompt, response)
    # 检查分数方差，应该很小

# 2. 混合评分策略
def hybrid_scorer(prompt, response):
    # 规则分：快速但有偏
    rule_score = rule_based_score(response)
    # 模型分：慢但准确
    model_score = rm_model.score(prompt, response)
    # 混合：平衡速度和准确性
    return 0.3 * rule_score + 0.7 * model_score

# 3. 定期评估评分器
# 用人工标注的golden set检查评分器准确率
```

### 3. **错误3: Temperature设置过低，组内响应太相似**
**描述**:
- temperature=0.3, 组内响应几乎一样
- 相对优势接近0，学不到东西
- GRPO退化为单样本训练

**正确做法**:
```python
# GRPO依赖组内多样性
temperature = 1.0  # 默认值，保证多样性

# 验证多样性
responses = [model.generate(prompt, temp=temperature) for _ in range(4)]
diversity = calculate_diversity(responses)  # e.g., self-BLEU, edit distance
# 如果diversity太低，提高temperature或调整top_p

# 多样性不足的补救
if diversity < threshold:
    temperature = 1.2  # 提高
    # 或者用不同的随机种子、不同的prefix
```

### 4. **错误4: 直接用PPO的超参数训练GRPO**
**描述**:
- PPO的lr=1e-5, GRPO也用这个，导致过拟合
- PPO的batch_size=64, GRPO也用这个，OOM
- 不理解GRPO的采样机制，资源规划错误

**正确做法**:
```python
# GRPO和PPO参数不同
ppo_config = {
    "learning_rate": 1e-5,
    "batch_size": 64,
    "num_rollouts": 1,  # 每步生成1个
}

grpo_config = {
    "learning_rate": 1e-6,  # GRPO更稳定，用更小lr
    "batch_size": 8,  # 每个batch有G个响应，实际处理batch×G
    "group_size": 4,  # 每个prompt生成4个响应
}

# 资源规划
# PPO: batch_size=64 → 处理64个样本
# GRPO: batch_size=8, G=4 → 处理32个样本，但每个prompt有4个候选
# 显存: GRPO需要batch_size×G×seq_len的空间存储所有响应
```

### 5. **错误5: 忽略GRPO的训练曲线监控**
**描述**:
- 只看loss，不看组内分数分布
- 不监控多样性指标
- 不检查相对优势的统计特性

**正确做法**:
```python
def grpo_train_step(prompts):
    responses, log_probs = generate_group(prompts, G=4)
    scores = scorer(responses)
    advantages = normalize(scores)
    loss = -(log_probs * advantages).mean()
    
    # 关键监控指标
    metrics = {
        "loss": loss.item(),
        "mean_score": scores.mean(),  # 组内平均分（应该稳定提升）
        "score_std": scores.std(),  # 组内分数方差（不应该太小）
        "advantage_std": advantages.std(),  # 优势方差（学习信号强度）
        "diversity": calculate_diversity(responses),  # 组内多样性
    }
    
    # 异常检测
    if metrics["score_std"] < 0.1:
        print("Warning: 组内分数差异太小，检查温度或评分器")
    if metrics["diversity"] < 0.3:
        print("Warning: 组内响应太相似，提高温度")
    
    return loss, metrics
```

## 反问面试官的问题

### 技术深度类
1. **您团队在实际训练中，GRPO和PPO的效果差异有多大？是否观察到GRPO的稳定性优势？**
   - 了解实际工程经验，对比理论预期

2. **对于GRPO的评分器，您倾向于用独立RM还是LLM-as-Judge？成本和效果如何权衡？**
   - 了解评分器设计的实践选择

3. **GRPO的group_size在您的场景下如何选择？有没有做过消融实验？**
   - 了解超参数调优经验

### 业务场景类
1. **在您的业务中，对齐训练的主要目标是什么？是通用能力还是垂直领域？GRPO是否适用？**
   - 判断GRPO在该场景的适用性

2. **您团队目前在RLHF流程中遇到的最大痛点是什么？是稳定性、成本还是效果？**
   - 了解是否适合引入GRPO

3. **对于多轮对话对齐，您倾向于用GRPO还是DPO？各自的实践经验如何？**
   - 了解多轮场景的算法选择

## 自测题

### 口述（能流畅讲清楚）
1. **GRPO的核心思想**：用组内相对比较替代独立奖励模型，消除RM偏差累积
2. **GRPO vs PPO三大差异**：
   - 无需独立RM（GRPO隐式奖励）
   - 更稳定（组内相对比较抗偏差）
   - 更简单（少一个模型少一套流程）
3. **GRPO代价**：采样成本G倍，需要可靠评分器

### 手写（5分钟能写出）
1. **GRPO相对优势计算**：
```python
def compute_advantages(scores):
    # scores: [batch, group_size]
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True) + 1e-8
    advantages = (scores - mean) / std
    return advantages
```

2. **GRPO核心loss**：
```python
def grpo_loss(log_probs, advantages):
    # log_probs: [batch, G]
    # advantages: [batch, G]
    loss = -(log_probs * advantages).mean()
    return loss
```

3. **采样成本对比**：
```python
# PPO: 每步采样1个
ppo_cost_per_step = 1

# GRPO: 每步采样G个
grpo_cost_per_step = G

# PPO总成本（含RM训练）
ppo_total = C_RM + N * 1

# GRPO总成本
grpo_total = N * G

# GRPO更优条件
if C_RM > N * (G - 1):
    print("GRPO总成本更低")
```

## 标签
#GRPO #RLHF #PPO #训练 #DeepSeek #对齐 #稳定性 #采样效率 #腾讯 #美团 #阿里

## 相关文档
- [[01-RLHF总览]]
- [[02-PPO算法]]
- [[03-DPO算法]]
- [[05-偏好数据设计]]
- [[06-奖励黑客]]

---

## 参考资源
1. **DeepSeek-V3 Technical Report** - GRPO原始论文和工程实践
2. **PPO论文** - Proximal Policy Optimization Algorithms (Schulman et al., 2017)
3. **DPO论文** - Direct Preference Optimization (Rafailov et al., 2023)
4. **RLHF最佳实践** - OpenAI, Anthropic对齐技术报告

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX  
**适用场景**: LLM面试 - RLHF/对齐/训练  
**难度级别**: ⭐⭐⭐⭐ (高级，需要RL基础)
