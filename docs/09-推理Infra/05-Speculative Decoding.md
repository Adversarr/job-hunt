# Speculative Decoding

## 一句话结论
Speculative Decoding 用小模型（draft）快速推测多个 token，大模型（target）批量验证，接受率约 60-80% 时能将 decode 延迟降低 2-3 倍，但接受率低时反而更慢，适合长序列生成、batch 较小、target 与 draft 对齐良好的场景。

## 核心定义/公式

### Speculative Decoding 流程

```
Draft Model（小模型）快速推测 γ 个 token
         ↓
    [t1, t2, t3, ..., tγ]
         ↓
Target Model（大模型）批量验证
         ↓
    接受前 k 个 token，拒绝第 k+1 个
         ↓
    从拒绝位置重新采样
```

### 接受率（Acceptance Rate）公式

对于推测的第 $i$ 个 token $t_i$，接受概率为：

$$p(\text{accept } t_i) = \min\left(1, \frac{p_{\text{target}}(t_i | x, t_{<i})}{p_{\text{draft}}(t_i | x, t_{<i})}\right)$$

**整体接受率**：
$$\text{Acceptance Rate} = \frac{\text{接受的 token 数}}{\text{推测的 token 总数}}$$

**期望加速比**：
$$\text{Speedup} = \frac{\mathbb{E}[\text{接受的 token 数}] + 1}{\text{draft 推测成本} + \text{target 验证成本}}$$

### 理论最大加速比

当接受率为 $\alpha$ 时，推测 $\gamma$ 个 token 的期望生成 token 数：

$$\mathbb{E}[\text{tokens}] = \sum_{i=1}^{\gamma} \alpha^i + 1 \approx \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

其中最后一项 $+1$ 是拒绝后的重新采样保证。

### 代码示例（PyTorch 实现核心逻辑）

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional

class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        gamma: int = 4,  # 推测 token 数
        temperature: float = 1.0
    ):
        self.draft = draft_model
        self.target = target_model
        self.gamma = gamma
        self.temperature = temperature
        
    @torch.no_grad()
    def speculate(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft model 推测 γ 个 token"""
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone() if attention_mask is not None else None
        
        for _ in range(self.gamma):
            # Draft forward
            outputs = self.draft(
                current_ids,
                attention_mask=current_mask,
                use_cache=True
            )
            
            # 采样下一个 token
            logits = outputs.logits[:, -1, :] / self.temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs)
            
            # 更新输入
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask, 
                    torch.ones_like(next_token)
                ], dim=-1)
        
        return torch.cat(draft_tokens, dim=-1), torch.stack(draft_probs, dim=1)
    
    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:
        """Target model 批量验证推测的 token"""
        batch_size = input_ids.size(0)
        gamma = draft_tokens.size(1)
        
        # 拼接输入：input_ids + draft_tokens
        full_ids = torch.cat([input_ids, draft_tokens], dim=-1)
        
        # Target model 一次 forward 评估所有位置
        # 只需要计算 input_ids 长度之后的位置
        outputs = self.target(
            full_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        # 提取 target model 在每个推测位置的 logits
        # logits[0] 对应第一个 draft token 的预测
        start_pos = input_ids.size(1) - 1
        target_logits = outputs.logits[:, start_pos:start_pos+gamma+1, :]
        target_probs = torch.softmax(target_logits / self.temperature, dim=-1)
        
        # 逐个验证
        accepted_tokens = []
        n_accepted = 0
        
        for i in range(gamma):
            draft_token = draft_tokens[:, i]
            draft_prob = draft_probs[:, i]
            target_prob = target_probs[:, i]
            
            # 获取 target 对 draft token 的概率
            p_target = target_prob.gather(dim=-1, index=draft_token.unsqueeze(-1)).squeeze(-1)
            p_draft = draft_prob.gather(dim=-1, index=draft_token.unsqueeze(-1)).squeeze(-1)
            
            # 接受概率
            accept_prob = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))
            
            # 随机判断是否接受
            rand = torch.rand(batch_size, device=input_ids.device)
            accepted = (rand < accept_prob) & (p_draft > 1e-10)
            
            if accepted.all():
                accepted_tokens.append(draft_token)
                n_accepted += 1
            else:
                # 拒绝：从 target 分布重新采样
                # 使用 adjusted distribution: max(0, p_target - p_draft) 归一化
                adjusted_probs = torch.clamp(target_prob - draft_prob, min=0)
                adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                
                # 对拒绝的样本重新采样
                new_token = torch.multinomial(adjusted_probs, num_samples=1).squeeze(-1)
                new_token = torch.where(accepted, draft_token, new_token)
                accepted_tokens.append(new_token)
                break
        
        # 如果全部接受，从最后一个 target 分布采样一个 bonus token
        if n_accepted == gamma:
            bonus_probs = target_probs[:, -1]
            bonus_token = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
            accepted_tokens.append(bonus_token)
        
        return torch.stack(accepted_tokens, dim=1), n_accepted
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """完整生成流程"""
        generated = input_ids.clone()
        current_mask = attention_mask.clone() if attention_mask is not None else None
        
        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # 1. Draft 推测
            draft_tokens, draft_probs = self.speculate(generated, current_mask)
            
            # 2. Target 验证
            accepted_tokens, n_accepted = self.verify(
                generated, draft_tokens, draft_probs, current_mask
            )
            
            # 3. 更新生成结果
            generated = torch.cat([generated, accepted_tokens], dim=-1)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(accepted_tokens.shape, dtype=current_mask.dtype, device=current_mask.device)
                ], dim=-1)
            
            tokens_generated += accepted_tokens.size(1)
        
        return generated
```

### vLLM 配置示例

```python
from vllm import LLM, SamplingParams

# Speculative decoding 配置
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",  # Target model
    speculative_model="meta-llama/Llama-2-7b-hf",  # Draft model
    num_speculative_tokens=4,  # γ
    tensor_parallel_size=4
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

outputs = llm.generate(prompts, sampling_params)
```

## 为什么（2-3 个因果链）

### 1. 为什么 Speculative Decoding 能加速推理？

**因果链**：
```
Decode 阶段每个 token 都需要一次完整 forward
  → 串行生成，延迟随 token 数线性增长
  → GPU 计算能力利用率低（batch=1 时 compute-bound）
  
Draft model 参数小、forward 快
  → 快速推测 γ 个 token（几乎无额外延迟）
  → Target model 一次 forward 批量验证 γ 个位置
  → 如果接受率高（60-80%），平均每步生成 2-4 个 token
  → 延迟降低 2-3 倍
```

**关键洞察**：
- **瓶颈转移**：从 "多次小 forward" 到 "一次大 forward"
- **内存带宽优化**：Target model 的参数加载次数减少（从 N 次到 N/γ 次）
- **计算密度提升**：批量验证时矩阵乘法效率更高

### 2. 为什么有时反而更慢？

**因果链**：
```
接受率低（< 50%）
  → 大量推测 token 被拒绝
  → Draft 的推测成本 + Target 的验证成本 > 直接生成的成本
  
Draft model 太慢
  → 推测延迟接近 Target 单步延迟
  → 即使接受率高，加速比也有限
  
Draft/Target 分布不匹配
  → Draft 偏好与 Target 差异大
  → 接受率持续低
  → 频繁回退到单步生成
```

**典型场景**：
- **短序列生成**（max_tokens < 20）：推测开销占比高
- **Draft 与 Target 未对齐**：接受率可能 < 40%
- **大 batch 场景**：Target 本身计算密度已高，加速空间有限

### 3. 为什么接受率是关键指标？

**因果链**：
```
接受率 α 决定期望生成 token 数
  → E[tokens] = α + α² + ... + α^γ + 1 ≈ 1/(1-α)（当 γ 足够大）
  → α=0.7 时，期望每步生成 3.3 个 token
  → α=0.4 时，期望每步生成 1.7 个 token
  
加速比 = (期望 token 数 × Target 单步延迟) / (Draft 推测成本 + Target 验证成本)
  → α 高：分子大，分母固定，加速明显
  → α 低：分子小，推测开销浪费，加速比 < 1
```

## 怎么做（可落地步骤）

### 标准做法

#### 步骤 1: 选择 Draft Model

**策略**：
1. **同系列小模型**：
   - Target: Llama-2-70B → Draft: Llama-2-7B
   - Target: Qwen-72B → Draft: Qwen-7B
   - **优势**：对齐度高，接受率 70-85%

2. **蒸馏小模型**：
   - 用 Target 数据蒸馏小模型
   - **优势**：接受率更高（可达 80-90%）
   - **代价**：需要额外训练

3. **自推测（Self-Speculative）**：
   - 跳过部分层做 draft
   - **优势**：无需额外模型
   - **适用**：层数多的大模型（如 96 层用前 8 层 draft）

#### 步骤 2: 确定推测长度 γ

**选择策略**：
```python
# 经验公式：根据接受率选择 γ
# α > 0.8: γ = 4-8
# α = 0.6-0.8: γ = 3-5
# α < 0.6: γ = 2-3（或放弃 speculative decoding）

gamma_values = {
    "high_acceptance": 6,   # α > 0.8
    "medium_acceptance": 4, # 0.6 < α < 0.8
    "low_acceptance": 2     # α < 0.6
}
```

**调整方法**：
1. **离线测试**：在验证集上测试不同 γ 的加速比
2. **动态调整**：根据实时接受率调整 γ
   ```python
   if avg_acceptance_rate > 0.8:
       gamma = min(gamma + 1, max_gamma)
   elif avg_acceptance_rate < 0.5:
       gamma = max(gamma - 1, min_gamma)
   ```

#### 步骤 3: 部署与监控

**部署配置**：
```python
# vLLM 配置
speculative_config = {
    "speculative_model": "meta-llama/Llama-2-7b-hf",
    "num_speculative_tokens": 4,
    "speculative_max_model_len": 2048,  # Draft 最大长度
    
    # 接受率监控
    "acceptance_rate_window": 100,  # 滑动窗口大小
    "min_acceptance_rate": 0.5,     # 低于此值动态关闭
}
```

**监控指标**：
```python
# 关键指标
metrics = {
    "acceptance_rate": "平均接受率（目标 > 60%）",
    "avg_tokens_per_step": "每步平均生成 token 数（目标 > 2）",
    "speedup": "实际加速比（目标 > 1.5x）",
    "draft_latency": "Draft 推测延迟",
    "target_latency": "Target 验证延迟",
    "total_latency": "总延迟",
}

# 报警规则
alerts = {
    "acceptance_rate < 0.4": "接受率过低，考虑关闭或切换 draft",
    "speedup < 1.0": "负收益，立即关闭 speculative decoding",
    "draft_latency > target_latency * 0.5": "Draft 太慢，考虑更小模型",
}
```

#### 步骤 4: 与动态 Batching 配合

**策略**：
```python
# Continuous Batching + Speculative Decoding
# 挑战：不同请求的接受率不同，生成速度差异大

class AdaptiveSpeculativeScheduler:
    def __init__(self, min_acceptance_rate=0.5):
        self.min_acceptance = min_acceptance_rate
        self.request_stats = {}  # 跟踪每个请求的接受率
        
    def should_speculate(self, request_id: str) -> bool:
        """动态决定是否启用 speculative decoding"""
        if request_id not in self.request_stats:
            return True  # 初始启用
        
        stats = self.request_stats[request_id]
        
        # 接受率低于阈值，关闭 speculative
        if stats["acceptance_rate"] < self.min_acceptance:
            return False
        
        # 短序列不启用
        if stats["remaining_tokens"] < 20:
            return False
        
        return True
    
    def update_stats(self, request_id: str, n_accepted: int, gamma: int):
        """更新请求统计"""
        if request_id not in self.request_stats:
            self.request_stats[request_id] = {
                "total_speculated": 0,
                "total_accepted": 0,
                "acceptance_rate": 0.0
            }
        
        stats = self.request_stats[request_id]
        stats["total_speculated"] += gamma
        stats["total_accepted"] += n_accepted
        stats["acceptance_rate"] = stats["total_accepted"] / stats["total_speculated"]
```

### 关键配置/参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `gamma`（推测长度） | 4（初始值） | 根据接受率动态调整，范围 2-8 |
| `draft_model_size` | Target 的 1/10 ~ 1/5 | 平衡推测速度和对齐度 |
| `temperature` | Target 和 Draft 保持一致 | 不一致会降低接受率 |
| `min_acceptance_rate` | 0.5 | 低于此值动态关闭 |
| `max_speculative_len` | min(draft_max_len, target_max_len) | 受两个模型上下文限制 |

### 接受率优化技巧

```python
# 1. Draft 和 Target 使用相同的 tokenizer
draft_tokenizer = target_tokenizer  # 必须

# 2. Temperature 和 sampling 参数一致
draft_sampling = target_sampling

# 3. Top-p / Top-k 过滤一致性
def aligned_sampling(logits, temperature, top_p, top_k):
    # Top-k 过滤
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p 过滤
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    # Temperature
    probs = torch.softmax(logits / temperature, dim=-1)
    return probs

# 4. 对齐训练（可选）
def distill_draft_from_target(draft_model, target_model, data):
    """用 target 数据蒸馏 draft，提高对齐度"""
    for batch in data:
        with torch.no_grad():
            target_logits = target_model(batch).logits
        
        draft_logits = draft_model(batch).logits
        loss = kl_divergence(draft_logits, target_logits)
        
        loss.backward()
        optimizer.step()
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **Speculative Decoding（高接受率）** | 延迟降低 2-3x，吞吐提升 1.5-2x | 额外显存（Draft 模型），实现复杂 | 长序列生成、小 batch、延迟敏感场景 |
| **Speculative Decoding（低接受率）** | 无收益或负收益 | 浪费计算资源，延迟反而增加 | 不适用（应关闭） |
| **Self-Speculative（跳层）** | 无需额外模型，显存开销小 | 只适用于深模型，精度损失 | 层数 ≥ 48 的模型 |
| **常规 Decoding** | 实现简单，稳定 | 延迟较高，GPU 利用率低 | 短序列、大 batch、吞吐优先 |

### 场景收益对比

| 场景 | 预期加速比 | 接受率 | 是否推荐 |
|------|-----------|--------|----------|
| 长序列生成（max_tokens > 100） | 2-3x | 70-85% | **强烈推荐** |
| 短序列生成（max_tokens < 20） | 0.8-1.2x | 60-70% | 不推荐 |
| 大 batch（batch_size > 32） | 1.0-1.3x | 60-75% | 视情况 |
| 小 batch（batch_size ≤ 4） | 2-3x | 70-85% | **强烈推荐** |
| 流式输出（逐 token） | 2-3x | 70-85% | **推荐** |
| Draft/Target 未对齐 | 0.6-0.9x | 30-50% | **强烈不推荐** |

## 高频追问（至少 5 个）

### 1. Q: Speculative decoding 是啥？（字节/阿里/腾讯）

**A**: 
Speculative decoding 是一种推理加速技术，用小模型（draft model）快速推测多个 token，大模型（target model）批量验证。核心思想是：
- **Draft**：小模型快速生成 γ 个候选 token（如 4 个）
- **Verify**：大模型一次 forward 验证所有候选，接受匹配的 token
- **Accept/Reject**：根据概率比例决定接受或重新采样

**关键公式**：
$$p(\text{accept } t_i) = \min\left(1, \frac{p_{\text{target}}(t_i)}{p_{\text{draft}}(t_i)}\right)$$

**收益**：接受率 70% 时，延迟降低 2-3 倍。

### 2. Q: Draft/target 怎么配？

**A**:
**三种主流策略**：

1. **同系列小模型**：
   - Llama-2-70B + Llama-2-7B（接受率 70-80%）
   - Qwen-72B + Qwen-7B
   - **关键**：tokenizer 必须一致

2. **蒸馏小模型**：
   ```python
   # 用 Target 数据蒸馏 Draft
   loss = KL_divergence(draft_logits, target_logits)
   ```
   - 接受率可达 80-90%
   - 需要额外训练成本

3. **Self-Speculative（跳层）**：
   - 96 层模型用前 8 层做 draft
   - 无需额外模型，显存节省
   - 适用深模型（≥ 48 层）

**配置原则**：
- Draft 大小：Target 的 1/10 ~ 1/5
- Tokenizer：必须一致
- Temperature：必须一致
- 对齐度：越高越好

### 3. Q: 接受率低会怎样？

**A**:
**后果**：
1. **负收益**：推测 + 验证成本 > 直接生成成本
   - 例如：γ=4，接受率 30%，平均生成 1.4 token/step
   - 加速比 = 1.4 / (draft_成本 + target_成本) < 1.0

2. **延迟增加**：每次拒绝都需要重新采样
   - Draft 浪费的计算无法回收

3. **吞吐下降**：GPU 资源浪费在无效推测上

**原因分析**：
- Draft/Target 分布不匹配（最常见）
- Sampling 参数不一致
- 任务太难（如推理任务）

**解决方案**：
```python
# 1. 动态关闭
if acceptance_rate < 0.4:
    disable_speculative_decoding()

# 2. 减小 gamma
if acceptance_rate < 0.5:
    gamma = max(2, gamma - 1)

# 3. 对齐训练
distill_draft_from_target()
```

### 4. Q: 什么场景收益最大？

**A**:
**高收益场景**：
1. **长序列生成**（max_tokens > 100）
   - 推测开销摊薄，加速比 2-3x

2. **小 batch 推理**（batch_size ≤ 4）
   - GPU 利用率低，speculative 提升明显

3. **流式输出**
   - 用户感知延迟降低明显

4. **延迟敏感场景**
   - 在线对话、实时翻译

**低收益/负收益场景**：
1. **短序列**（max_tokens < 20）：推测开销占比高
2. **大 batch**（batch_size > 32）：Target 本身计算密度高
3. **接受率低**（< 50%）：负收益
4. **吞吐优先**：不追求单请求延迟

**经验公式**：
$$\text{Expected Speedup} \approx \frac{\mathbb{E}[\text{tokens}] \times T_{\text{target}}}{T_{\text{draft}} \times \gamma + T_{\text{target}}}$$

### 5. Q: 为什么有时反而更慢？

**A**:
**三大原因**：

1. **接受率过低**（< 50%）：
   ```
   平均生成 token 数 ≈ 1/(1-α) - 1
   α=0.3: 生成 1.4 token/step，推测 4 个，浪费 2.6 个
   总成本 = draft_4 + target_4 > target_1
   ```

2. **Draft 模型太慢**：
   ```
   如果 draft_延迟 > target_延迟 * 0.5
   即使接受率 80%，加速比也有限
   ```

3. **序列太短**：
   ```
   max_tokens = 10
   推测开销：draft_4 + target_4 ≈ target_1.5
   实际只生成 10 token，无法摊薄
   ```

**排查方法**：
```python
# 监控关键指标
if speedup < 1.0:
    print(f"Acceptance rate: {acceptance_rate}")
    print(f"Draft latency: {draft_latency}")
    print(f"Target latency: {target_latency}")
    print(f"Avg tokens/step: {avg_tokens_per_step}")
    
    # 建议
    if acceptance_rate < 0.5:
        print("建议：关闭 speculative decoding")
    elif draft_latency > target_latency * 0.3:
        print("建议：使用更小的 draft model")
```

### 6. Q: 和动态 batching 怎么配合？

**A**:
**挑战**：
- Continuous batching 中不同请求生成速度不同
- Speculative decoding 让速度差异更大
- 可能导致调度混乱

**解决方案**：

1. **动态启用/禁用**：
   ```python
   def should_speculate(request):
       # 短序列禁用
       if request.remaining_tokens < 20:
           return False
       # 低接受率禁用
       if request.acceptance_rate < 0.5:
           return False
       return True
   ```

2. **分组调度**：
   ```python
   # 将启用和禁用 speculative 的请求分组
   speculative_batch = [r for r in batch if should_speculate(r)]
   normal_batch = [r for r in batch if not should_speculate(r)]
   ```

3. **自适应 γ**：
   ```python
   # 根据实时接受率调整
   if avg_acceptance_rate > 0.8:
       gamma = min(gamma + 1, max_gamma)
   elif avg_acceptance_rate < 0.6:
       gamma = max(gamma - 1, min_gamma)
   ```

4. **资源隔离**：
   ```python
   # Draft 和 Target 分配不同 GPU
   draft_device = "cuda:0"
   target_device = "cuda:1"
   ```

### 7. Q: 线上怎么评估值不值得上？

**A**:
**评估流程**：

1. **离线测试**：
   ```python
   # 在验证集上测试
   baseline_latency = measure_latency(target_only)
   speculative_latency = measure_latency(draft + target)
   speedup = baseline_latency / speculative_latency
   
   metrics = {
       "speedup": speedup,
       "acceptance_rate": avg_acceptance,
       "p50_latency": p50,
       "p99_latency": p99,
       "throughput": tokens_per_second
   }
   ```

2. **A/B 测试**：
   ```python
   # 线上灰度
   ab_test_config = {
       "baseline_group": {"speculative": False},
       "experimental_group": {"speculative": True},
       "traffic_split": 0.1,  # 10% 流量
       "duration": "7d"
   }
   
   # 监控指标
   watch_metrics = [
       "latency_p50", "latency_p99",
       "acceptance_rate", "throughput",
       "error_rate", "user_satisfaction"
   ]
   ```

3. **ROI 计算**：
   ```
   收益：
   - 延迟降低 → 用户满意度提升
   - 吞吐提升 → 单请求成本降低
   
   成本：
   - 额外 GPU 显存（Draft 模型）
   - 实现和维护成本
   
   ROI = (收益 - 成本) / 成本
   如果 ROI > 1.5，值得上线
   ```

4. **决策标准**：
   ```
   必要条件：
   - 接受率 > 60%
   - 加速比 > 1.5x
   - 无精度损失
   
   充分条件：
   - 加速比 > 2x
   - 显存开销可接受
   - 延迟 P99 改善 > 30%
   ```

## 常见错误（至少 3 个）

### 1. **错误：Draft 和 Target 使用不同 tokenizer**

**现象**：接受率极低（< 30%），大量 token 被拒绝

**错误代码**：
```python
draft_tokenizer = AutoTokenizer.from_pretrained("gpt2")
target_tokenizer = AutoTokenizer.from_pretrained("llama-2")
# 错误：token id 不对应，无法正确验证
```

**正确做法**：
```python
# 必须使用相同的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("llama-2")
draft_model = AutoModel.from_pretrained("llama-2-7b")
target_model = AutoModel.from_pretrained("llama-2-70b")
```

**原因**：Speculative decoding 假设 draft 和 target 对同一个 token id 有相同理解，tokenizer 不同会导致语义错位。

### 2. **错误：Temperature 或 sampling 参数不一致**

**现象**：接受率低，即使 draft 和 target 对齐良好

**错误代码**：
```python
# Draft 使用 temperature=0.8
draft_probs = draft_model.forward(tokens, temperature=0.8)

# Target 使用 temperature=1.0
target_probs = target_model.forward(tokens, temperature=1.0)
# 错误：概率分布不匹配
```

**正确做法**：
```python
temperature = 0.8
draft_probs = draft_model.forward(tokens, temperature=temperature)
target_probs = target_model.forward(tokens, temperature=temperature)
# 确保参数一致
```

**原因**：接受概率计算依赖 $p_{\text{target}}(t) / p_{\text{draft}}(t)$，temperature 不同会导致概率分布差异，降低接受率。

### 3. **错误：短序列强制启用 speculative decoding**

**现象**：max_tokens < 20 时，延迟反而增加

**错误做法**：
```python
# 不加判断，所有请求都启用
llm = LLM(
    model=target_model,
    speculative_model=draft_model,
    num_speculative_tokens=4
)
```

**正确做法**：
```python
def generate(prompt, max_tokens):
    if max_tokens < 20:
        # 短序列禁用 speculative
        return target_model.generate(prompt, max_tokens)
    else:
        # 长序列启用
        return speculative_generate(prompt, max_tokens)
```

**原因**：短序列时，推测开销无法摊薄，总延迟 = draft_延迟 + target_延迟 > target_延迟。

### 4. **错误：固定 γ，不自适应调整**

**现象**：接受率波动时，性能不稳定

**错误做法**：
```python
gamma = 4  # 固定不变
for step in range(max_steps):
    draft_tokens = speculate(gamma)
    verify(draft_tokens)
```

**正确做法**：
```python
gamma = 4
acceptance_history = []

for step in range(max_steps):
    draft_tokens = speculate(gamma)
    n_accepted, tokens = verify(draft_tokens)
    acceptance_history.append(n_accepted / gamma)
    
    # 动态调整 γ
    if len(acceptance_history) > 10:
        avg_acceptance = np.mean(acceptance_history[-10:])
        if avg_acceptance > 0.8:
            gamma = min(gamma + 1, 8)
        elif avg_acceptance < 0.5:
            gamma = max(gamma - 1, 2)
```

**原因**：不同 prompt、不同生成阶段接受率差异大，固定 γ 无法适应动态变化。

### 5. **错误：忽略验证时的数值稳定性**

**现象**：出现 NaN 或接受概率 > 1

**错误代码**：
```python
# 直接计算比例
accept_prob = p_target / p_draft  # p_draft 可能为 0
```

**正确做法**：
```python
# 加数值稳定保护
epsilon = 1e-10
accept_prob = torch.min(
    torch.ones_like(p_target),
    p_target / (p_draft + epsilon)
)

# 处理极端情况
accept_prob = torch.where(
    p_draft < epsilon,
    torch.ones_like(p_target),  # draft 概率极小时直接接受
    accept_prob
)
```

**原因**：浮点数精度问题，p_draft 接近 0 时会导致除零或数值不稳定。

## 反问面试官的问题

### 技术深度类

1. **"在你们的生产环境中，speculative decoding 的接受率大概是多少？是通过蒸馏提升对齐度，还是直接用同系列小模型？"**
   - 目的：了解团队的实践经验和技术路线

2. **"你们遇到过 speculative decoding 负收益的情况吗？如何排查和解决的？"**
   - 目的：了解实际落地的坑和解决方案

3. **"在 continuous batching 场景下，如何处理不同请求启用/禁用 speculative decoding 的调度问题？"**
   - 目的：了解系统级工程能力

4. **"有没有考虑过 self-speculative decoding（跳层）？相比额外的 draft model，权衡如何？"**
   - 目的：展示技术广度

### 业务场景类

1. **"你们线上推理的延迟和吞吐目标是什么？speculative decoding 的收益是否满足业务需求？"**
   - 目的：了解业务约束

2. **"在多轮对话场景中，speculative decoding 的接受率会变化吗？如何处理上下文变化？"**
   - 目的：了解长尾场景

3. **"你们是如何评估 speculative decoding 是否值得上的？有做过 A/B 测试吗？"**
   - 目的：了解评估方法和决策流程

## 自测题

### 口述（能流畅讲清楚的知识点）

1. **Speculative decoding 的完整流程**：
   - Draft 推测 → Target 验证 → 接受/拒绝 → 重新采样
   - 每个阶段做什么，为什么能加速

2. **接受率公式推导**：
   - 为什么是 $\min(1, p_{\text{target}}/p_{\text{draft}})$
   - 这个公式如何保证输出分布不变

3. **Draft/Target 配对策略**：
   - 同系列小模型、蒸馏小模型、Self-speculative 的优劣
   - 各自适用场景

4. **为什么有时更慢**：
   - 三大原因：接受率低、Draft 太慢、序列太短
   - 如何排查和解决

5. **与动态 batching 配合**：
   - 挑战：生成速度差异大
   - 解决方案：动态启用、分组调度、自适应 γ

### 手写（5 分钟能写出的代码/公式）

1. **接受概率计算**：
   ```python
   def compute_accept_prob(p_target, p_draft):
       """计算接受概率"""
       # 实现数值稳定的版本
       pass
   ```

2. **期望生成 token 数**：
   ```
   给定接受率 α=0.7，推测长度 γ=4
   计算期望每步生成的 token 数
   ```

3. **验证流程伪代码**：
   ```python
   def verify(draft_tokens, draft_probs, target_probs):
       """批量验证 draft token"""
       accepted = []
       for i, token in enumerate(draft_tokens):
           # 判断是否接受
           # 如果拒绝，从 target 分布重新采样
           pass
       return accepted
   ```

4. **加速比估算**：
   ```python
   def estimate_speedup(acceptance_rate, gamma, draft_latency_ratio):
       """
       acceptance_rate: 接受率
       gamma: 推测 token 数
       draft_latency_ratio: draft 延迟 / target 延迟
       
       返回：加速比
       """
       pass
   ```

## 标签

#推理 #spec_decode #工程 #延迟优化 #KV_cache #batching #vLLM #字节 #阿里 #腾讯 #美团 #百度

## 相关文档

- [[01-Prefill与Decode]] - 理解 decode 阶段的特性和瓶颈
- [[04-Batching与调度]] - 动态 batching 与 speculative decoding 的协同
- [[../01-Transformer基础/02-Attention机制]] - Attention 机制与 KV cache 基础
- [[03-Paged-Attention]] - KV cache 管理与显存优化