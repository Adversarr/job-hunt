# PPO 算法

## 一句话结论
PPO 通过裁剪策略比率（clip）约束更新幅度，结合优势函数估计和 KL 惩罚，在保证训练稳定性的同时优化 LLM 策略，是 RLHF 的主流算法但存在采样成本高、训练不稳定、奖励模型偏差放大等痛点。

## 核心定义/公式

### PPO-Clip 目标函数

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：策略比率（policy ratio）
- $\hat{A}_t$：优势函数估计
- $\epsilon$：裁剪参数，通常取 0.1~0.2

### 优势函数（GAE）

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$：TD 残差
- $\gamma$：折扣因子，通常 0.99
- $\lambda$：GAE 参数，通常 0.95

### RLHF-PPO 完整目标

$$L(\theta) = \mathbb{E}\left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref})$$

或使用 adaptive KL penalty：
$$L_{total} = L^{CLIP} - c_1 L^{VF} + c_2 H(\pi_\theta) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

### 关键组件

| 组件 | 作用 | 典型配置 |
|------|------|----------|
| Policy Ratio | 衡量策略变化幅度 | 监控是否超出 [1-ε, 1+ε] |
| Clip (ε) | 限制单步更新幅度 | 0.1~0.2 |
| Advantage (Â) | 估计动作相对好坏 | GAE(γ=0.99, λ=0.95) |
| KL Penalty (β) | 约束策略不偏离太远 | 初始 0.05，adaptive 调整 |
| Value Loss 权重 (c₁) | 价值函数拟合权重 | 0.5 |
| Entropy Bonus (c₂) | 鼓励探索 | 0.01~0.02 |

### 代码示例

```python
import torch
import torch.nn.functional as F

def compute_ppo_loss(
    log_probs,           # [batch, seq_len] 当前策略的 log probability
    old_log_probs,       # [batch, seq_len] 旧策略的 log probability
    advantages,          # [batch, seq_len] 优势函数
    values,              # [batch, seq_len] 当前价值估计
    returns,             # [batch, seq_len] 回报
    ref_log_probs,       # [batch, seq_len] reference model 的 log probability
    clip_ratio=0.2,      # ε
    value_coef=0.5,      # c_1
    entropy_coef=0.01,   # c_2
    kl_coef=0.1,         # β
):
    """PPO loss 计算"""
    
    # 1. Policy Ratio: r(θ) = π_θ / π_old
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 2. Clipped Policy Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 3. Value Function Loss (MSE)
    value_loss = F.mse_loss(values, returns)
    
    # 4. Entropy Bonus（鼓励探索）
    entropy = -torch.exp(log_probs) * log_probs  # H(π) = -∑π log π
    entropy_loss = -entropy_coef * entropy.mean()
    
    # 5. KL Divergence Penalty
    kl_div = (torch.exp(ref_log_probs) * (ref_log_probs - log_probs)).sum(dim=-1).mean()
    kl_loss = kl_coef * kl_div
    
    # 6. Total Loss
    total_loss = policy_loss + value_coef * value_loss + entropy_loss + kl_loss
    
    return total_loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
        'kl_div': kl_div.item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_max': ratio.max().item(),
    }


def compute_gae(
    rewards,        # [num_steps, batch_size]
    values,         # [num_steps, batch_size]
    dones,          # [num_steps, batch_size]
    gamma=0.99,     # 折扣因子
    lam=0.95,       # GAE λ
):
    """Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + (γλ)δ_{t+1} + ...
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages)
    returns = advantages + values  # Q = A + V
    
    return advantages, returns


class AdaptiveKLController:
    """自适应 KL 惩罚系数"""
    def __init__(self, init_kl_coef=0.2, target_kl=6.0, horizon=10000):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
    
    def update(self, current_kl):
        """根据当前 KL 散度调整系数"""
        if current_kl < self.target_kl / 1.5:
            self.kl_coef /= 2
        elif current_kl > self.target_kl * 1.5:
            self.kl_coef *= 2
        return self.kl_coef
```

## 为什么（2-3 个因果链）

1. **为什么需要 Clip 机制？**
   - **现象**：策略梯度方法（如 REINFORCE）容易因大步长更新导致策略崩溃
   - **根因**：策略空间非凸，大步长可能使性能骤降且难以恢复
   - **结果**：PPO 通过裁剪 policy ratio，将更新约束在信任区域内，保证单调或近似单调改进

2. **为什么 PPO 需要 KL Penalty？**
   - **现象**：LLM 在 RL 阶段容易"忘记"预训练/SFT 阶段的能力
   - **根因**：奖励模型仅在特定分布上训练，策略偏离后 RM 泛化性下降，导致奖励黑客
   - **结果**：KL penalty 强制策略不偏离 reference model 太远，保持能力同时优化目标

3. **为什么用 GAE 而不是简单的 TD(0)？**
   - **现象**：方差大导致训练不稳定，或偏差大导致收敛慢
   - **根因**：TD(0) 方差小但偏差大，Monte Carlo 方差大但无偏
   - **结果**：GAE 通过 λ 参数在偏差和方差间权衡，λ=0 退化为 TD(0)，λ=1 退化为 Monte Carlo

## 怎么做（可落地步骤）

### 标准做法

1. **初始化模型**：
   - Policy Model：待优化的模型（从 SFT 初始化）
   - Value Model：价值函数（通常与 Policy 共享 backbone，单独 value head）
   - Reward Model：奖励模型（提前训练好）
   - Reference Model：参考模型（冻结，用于 KL 计算）

2. **采样阶段（Rollout）**：
   - 用当前 policy 生成响应
   - 计算 reward（RM 打分 + 规则奖励）
   - 存储 trajectories：$(s_t, a_t, r_t, \log\pi_{old})$

3. **优势估计**：
   - 用 Value Model 估计 V(s)
   - 用 GAE 计算 $\hat{A}_t$

4. **多轮更新**：
   - 对同一批数据执行 K 轮（通常 3~4 轮）梯度更新
   - 监控 KL 散度，超出阈值提前停止

5. **评估与调整**：
   - 监控 reward、KL、entropy、value loss
   - 调整 clip_ratio、kl_coef、learning rate

### 关键配置/参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| clip_ratio (ε) | 0.1~0.2 | 太小收敛慢，太大不稳定 |
| gamma (γ) | 0.99 | 折扣因子，LLM 中通常接近 1 |
| lam (λ) | 0.95 | GAE 参数，平衡偏差/方差 |
| kl_coef (β) | 0.05~0.1（adaptive） | 初始值，训练中动态调整 |
| target_kl | 0.1~0.2（adaptive） | KL 阈值，超出则调整 β |
| value_coef | 0.5 | 价值损失权重 |
| entropy_coef | 0.01~0.02 | 熵奖励权重 |
| ppo_epochs | 3~4 | 每批数据更新轮数 |
| learning_rate | 1e-6~5e-6 | Policy 学习率，较小防崩溃 |
| batch_size | 64~256 | PPO mini-batch 大小 |

### 工程实现要点

```python
# TRL (Transformer Reinforcement Learning) 框架配置示例
from trl import PPOConfig

ppo_config = PPOConfig(
    model_name="your-model",
    learning_rate=1e-6,
    adap_kl_ctrl=True,          # 自适应 KL 控制
    init_kl_coef=0.05,          # 初始 KL 系数
    target_kl=0.1,              # 目标 KL
    horizon=10000,              # KL 控制器更新频率
    gamma=0.99,                 # 折扣因子
    lam=0.95,                   # GAE λ
    cliprange=0.2,              # clip ε
    cliprange_value=0.2,        # value clip
    vf_coef=0.1,                # 价值损失权重
    batch_size=128,             # PPO batch size
    gradient_accumulation_steps=1,
    ppo_epochs=4,               # 更新轮数
    max_grad_norm=0.5,          # 梯度裁剪
    seed=42,
)
```

### 训练循环伪代码

```python
for epoch in range(num_epochs):
    # 1. Rollout: 用当前策略采样
    rollout_buffer = collect_rollouts(policy_model, reward_model, env)
    
    # 2. Compute advantages
    with torch.no_grad():
        values = value_model(rollout_buffer.states)
        advantages, returns = compute_gae(
            rollout_buffer.rewards, 
            values, 
            rollout_buffer.dones
        )
        # Normalize advantages（重要！稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 3. PPO update (K epochs)
    for _ in range(ppo_epochs):
        for batch in rollout_buffer.get minibatches(batch_size):
            # Forward pass
            log_probs, new_values, entropy = policy_model.evaluate(batch)
            
            # Compute loss
            loss, stats = compute_ppo_loss(
                log_probs, batch.old_log_probs, batch.advantages,
                new_values, batch.returns, batch.ref_log_probs,
                clip_ratio, value_coef, entropy_coef, kl_coef
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Early stop if KL too large
            if stats['kl_div'] > target_kl * 2:
                break
    
    # 4. Update KL controller
    kl_coef = kl_controller.update(stats['kl_div'])
    
    # 5. Logging
    print(f"Epoch {epoch}: reward={stats['reward_mean']:.3f}, "
          f"KL={stats['kl_div']:.4f}, entropy={stats['entropy']:.3f}")
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| PPO | 稳定性优于 TRPO；实现相对简单；样本效率高于 on-policy 平均水平 | 仍需大量采样（on-policy）；多模型维护成本高；超参敏感 | 需要精细控制策略更新、有足够算力的场景 |
| DPO | 无需训练 RM；无需价值函数；off-policy，样本效率高 | 无在线探索能力；依赖偏好数据质量；无法处理复杂奖励函数 | 偏好数据充足、奖励函数可隐式表达的场景 |
| GRPO | 相比 PPO 更稳定；无需价值函数；Group-based 采样 | 实现较新，生态不如 PPO 成熟；需要多响应采样 | DeepSeek 等超大规模模型对齐 |
| REINFORCE | 最简单，无需价值函数 | 方差大，收敛慢；无稳定性保证 | 快速验证、小规模实验 |

## 高频追问（至少 5 个）

1. **Q: PPO 为什么比 TRPO 更实用？**
   A: TRPO 需要计算二阶导（Fisher 信息矩阵）和共轭梯度，计算开销大且实现复杂。PPO 用一阶优化的 clip 近似 TRPO 的信任区域约束，实现简单且效果接近。

2. **Q: Policy Ratio 为什么要取 exp(log_prob - old_log_prob)？**
   A: 数值稳定性。直接计算概率比 $\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$ 在高维空间容易数值不稳定，而在 log 空间做减法后再 exp 更稳定。同时，log_prob 可以通过 `log_softmax` 直接获得，避免先 softmax 再 log 的精度损失。

3. **Q: PPO 中多个模型（policy/value/reward/reference）如何节省显存？**
   A: 
   - Policy 和 Value 共享 backbone，仅 value head 独立
   - Reference 模型可量化和/或 offload 到 CPU
   - Reward Model 可推理时量化或用较小的蒸馏版本
   - 使用 Gradient Checkpointing 减少 activation 内存

4. **Q: 为什么 PPO 要对同一批数据更新 K 轮？**
   A: 提高样本效率。on-policy 算法采样后数据即"过期"，单轮更新浪费。但 K 不能太大，否则策略偏离 rollout 时的策略太远，重要性采样的方差会爆炸。实践中 K=3~4，配合 early stopping（KL 超阈值）。

5. **Q: PPO 的奖励黑客问题怎么解决？**
   A: 
   - **数据层面**：RM 训练数据多样化、加入 hard negative、对抗样本
   - **模型层面**：KL penalty 约束、entitlement bonus
   - **工程层面**：长度惩罚/归一化、多 RM 投票、规则覆盖
   - **监控层面**：held-out evaluation、人工抽样检查

6. **Q: 为什么 GRPO 比 PPO 更稳定？**
   A: GRPO 通过 Group-based 采样（同一 prompt 生成多响应，组内对比计算相对优势）消除了对价值函数的依赖。价值函数估计不准是 PPO 不稳定的主要来源之一。GRPO 直接用组内 reward 排序估计优势，避免了 value fitting 的偏差。

7. **Q: PPO 训练时 loss 震荡怎么排查？**
   A: 
   - 检查 reward 尺度（是否过大/过小，需要 normalize）
   - 检查 KL 是否爆炸（降低 learning rate 或增大 kl_coef）
   - 检查 value loss 是否不收敛（value head 是否够大、是否需要预训练）
   - 检查优势函数是否 normalize
   - 检查 batch size 和 sample efficiency

## 常见错误（至少 3 个）

1. **忘记 normalize 优势函数**
   - **错误**：直接用 GAE 输出的优势值，不同 batch 尺度差异大
   - **正确做法**：`advantages = (advantages - mean) / (std + 1e-8)`，归一化后训练更稳定

2. **clip_ratio 设置过大或过小**
   - **错误**：ε=0.5 导致策略更新过度；ε=0.01 导致收敛极慢
   - **正确做法**：从 0.2 开始，根据 KL 和 reward 曲线微调，观察 ratio 的统计量

3. **KL penalty 系数固定不调整**
   - **错误**：固定 β 导致 KL 可能爆炸或策略过于保守
   - **正确做法**：使用 adaptive KL controller，根据 target_kl 动态调整 β

4. **价值和策略模型共享全部参数**
   - **错误**：policy loss 和 value loss 直接相加，相互干扰
   - **正确做法**：共享 backbone，但 value head 独立；或使用 multi-head 设计；注意 value_coef 的权重

5. **忽略 entropy bonus**
   - **错误**：策略过早收敛到次优解
   - **正确做法**：加入 entropy bonus 鼓励探索，特别是 LLM 这种高维动作空间

## 反问面试官的问题

1. **技术深度类**：
   - "团队在实际 PPO 训练中遇到过哪些稳定性问题？是主要靠调参还是有更系统的解决方案？"
   - "对于 GRPO 这种新方法，团队是否有尝试？相比 PPO 的收益和风险如何评估？"

2. **业务场景类**：
   - "我们的 RM 主要覆盖哪些维度？如何平衡奖励黑客风险和对齐效果？"
   - "PPO 训练的算力成本如何？是否有 off-policy 或蒸馏等降低成本的方案？"

## 自测题

- **口述**：
  - 解释 PPO 的四个核心组件（ratio、clip、advantage、KL）及其作用
  - 说明 GAE 中 λ 和 γ 参数如何影响偏差-方差权衡
  - 描述 PPO 训练的完整 pipeline（从采样到更新）

- **手写**：
  - 5 分钟内写出 PPO-Clip 的损失函数公式
  - 写出 GAE 的递推公式
  - 写出 KL 散度的计算公式（从 log_prob 形式）

## 标签
#PPO #RLHF #RL #训练 #策略梯度 #GAE #advantage_function #handwrite #derive #阿里 #腾讯 #美团

## 相关文档
- [[01-RLHF总览]]
- [[03-DPO算法]]
- [[04-GRPO算法]]
- [[05-偏好数据设计]]
- [[06-奖励黑客]]
