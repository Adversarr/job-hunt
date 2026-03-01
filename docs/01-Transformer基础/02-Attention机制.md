# Attention 机制全家桶：Scaled Dot-Product Attention、MHA、MQA、GQA

## 一句话结论
Attention 本质是通过 Query-Key 相似度加权聚合 Value；MHA 通过多头并行捕捉不同子空间的特征模式，不提升计算效率但提升表达能力；MQA/GQA 在推理侧通过共享 KV cache 大幅降低显存占用和访存带宽，代价是精度轻微下降。

## 核心定义/公式

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$，$K, V \in \mathbb{R}^{m \times d_k}$
- $\sqrt{d_k}$ 为缩放因子，防止点积过大导致 softmax 梯度消失

**Softmax 稳定性（max-trick）**：
```python
def stable_softmax(x, dim=-1):
    # 数值稳定性：减去最大值
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

**Causal Mask 实现**：
```python
def apply_causal_mask(scores, mask_value=-1e9):
    # scores: [batch, heads, seq_len, seq_len]
    seq_len = scores.size(-1)
    # 上三角为 -inf（不含对角线）
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask.bool(), mask_value)
    return scores
```

### Multi-Head Attention (MHA)

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

参数量对比：
- 单头：$d_{model} \times d_{model} = d_{model}^2$
- 多头：$h \times (d_{model} \times d_k) \times 3 + d_{model} \times d_{model} = d_{model}^2 \times 4$（QKV + O 投影）

**维度推导**：
```python
# 输入: [batch, seq_len, d_model]
# 多头切分: d_model = h * d_k
Q = x @ W_Q  # [batch, seq_len, d_model]
Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
# Q: [batch, num_heads, seq_len, head_dim]
```

### Multi-Query Attention (MQA)

所有头共享同一组 K、V：
```python
# K, V 不分头，只有一份
K = x @ W_K  # [batch, seq_len, head_dim]
V = x @ W_V  # [batch, seq_len, head_dim]
# Q 仍然分头
Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
# Attention: Q @ K^T / sqrt(d_k) -> [batch, num_heads, seq_len, seq_len]
```

### Grouped-Query Attention (GQA)

将头分组，每组共享一组 K、V：
```python
# num_groups = num_heads // group_size
# 例如：32 头，8 组，每组 4 头共享 KV
K = K.view(batch, seq_len, num_groups, head_dim).transpose(1, 2)
V = V.view(batch, seq_len, num_groups, head_dim).transpose(1, 2)
# 然后对每组内的头进行广播
```

## 为什么（2-3 个因果链）

### 1. 为什么需要 Scaled Dot-Product

**因果链**：点积幅度随 $d_k$ 增大 → softmax 输入变大 → 输出趋近 one-hot → 梯度趋近 0 → 训练不稳定

```python
# 不加缩放时，d_k=512 时点积方差约 512
# 加 1/sqrt(d_k) 后方差归一化为 1
scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
```

### 2. 为什么需要多头（而不是单头）

**因果链**：单头注意力只能学习一种全局权重模式 → 表达能力受限 → 多头让不同头关注不同子空间（如：头 1 关注局部、头 2 关注全局、头 3 关注语义相似）→ 提升模型表达能力

**关键洞察**：多头不是为了效率，而是为了表达多样性。每个头维度变小 $d_k = d_{model}/h$，但并行计算 $h$ 个独立子空间。

### 3. 为什么 MQA/GQA 能加速推理

**因果链**：推理时 KV cache 存储历史 K/V → 显存占用随 seq_len 线性增长 → MHA 存 $h$ 组 KV → MQA/GQA 共享 KV → 显存降低 $h$ 倍（MQA）或 $g$ 倍（GQA）→ 访存带宽压力降低 → decode 阶段加速显著

```python
# MHA KV cache 显存（单层）：
# batch × seq_len × 2 × num_heads × head_dim × sizeof(dtype)
# MQA：batch × seq_len × 2 × 1 × head_dim × sizeof(dtype)
# 节省：num_heads 倍显存
```

## 怎么做（可落地步骤）

### 标准做法

#### 1. MHA 实现（PyTorch）

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性投影
        Q = self.W_Q(x)  # [batch, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 2. 多头切分
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]
        
        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 6. 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 7. 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        # 8. 输出投影
        output = self.W_O(context)
        
        return output, attn_weights
```

#### 2. MQA/GQA 实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        assert num_heads % num_groups == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_groups
        
        # Q 每个头独立
        self.W_Q = nn.Linear(d_model, d_model)
        # K/V 按组共享
        self.W_K = nn.Linear(d_model, num_groups * self.head_dim)
        self.W_V = nn.Linear(d_model, num_groups * self.head_dim)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Q 分头
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K/V 分组
        K = self.W_K(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        
        # 扩展 K/V 到每个头
        K = K.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)
        K = K.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # 后续同 MHA
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)
        
        return output, attn_weights
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `num_heads` | 32-64（大模型） | 平衡表达能力和计算开销，通常 $d_{model}/\text{num\_heads} \in [64, 128]$ |
| `head_dim` | 64-128 | 太小限制表达能力，太大参数冗余 |
| `num_groups`（GQA） | 8 | LLaMA2/3 使用 32 头 8 组，平衡精度和显存 |
| `mask_value` | -1e9 | 足够小避免数值问题，但不溢出 |

### 代码示例：只使用 NumPy 实现 MHA

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention_numpy(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    """
    x: [batch, seq_len, d_model]
    W_Q, W_K, W_V: [d_model, d_model]
    W_O: [d_model, d_model]
    mask: [seq_len, seq_len] 或 None
    """
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    
    # 投影
    Q = x @ W_Q  # [batch, seq_len, d_model]
    K = x @ W_K
    V = x @ W_V
    
    # 重塑为多头
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    # [batch, num_heads, seq_len, head_dim]
    
    # 注意力分数
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    
    # 应用 mask
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # softmax
    attn_weights = softmax(scores, axis=-1)
    
    # 加权求和
    context = attn_weights @ V
    
    # 合并多头
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 输出投影
    output = context @ W_O
    
    return output, attn_weights
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **MHA** | 表达能力最强，每个头独立学习 | 推理显存占用大（KV cache × num_heads） | 训练阶段、对精度要求极高场景 |
| **MQA** | 显存节省 h 倍，decode 加速明显 | 精度轻微下降（2-5%），表达能力受限 | 推理加速优先、长序列场景 |
| **GQA** | 平衡精度与显存，显存节省 g 倍 | 比 MHA 精度略低，比 MQA 实现复杂 | 推理加速与精度平衡（LLaMA2/3 默认） |
| **单头 Attention** | 计算简单，参数少 | 表达能力严重受限，无法捕捉多模式 | 极小模型或特定任务 |

**GQA vs MQA 精度对比**（以 LLaMA 为例）：
- MHA baseline：100%
- GQA (8 groups)：98-99%
- MQA：95-98%

## 高频追问（至少 5 个）

### 1. Q: 多头注意力能提高计算效率吗？

**A**: 不能。MHA 的计算量与单头相同（假设总维度 $d_{model}$ 不变），但并行性好。严格来说：
- FLOPs：$\mathcal{O}(n^2 \cdot d_{model})$，与头数无关
- 实际加速：来自并行硬件（GPU）对不同头的并行计算，而非算法本身的复杂度降低
- 表达能力提升：核心收益，非效率提升

### 2. Q: 头数减少会怎样？

**A**: 
- **精度下降**：表达能力受限，无法捕捉足够的子空间模式
- **但非线性下降**：有些头可能冗余，减少到一定程度影响不大
- **显存收益**：训练阶段收益小，推理 KV cache 节省明显
- **经验**：$d_{model}/\text{num\_heads} \in [64, 128]$ 较合理

### 3. Q: MQA/GQA 省的是什么？代价是什么？

**A**:
- **省显存**：KV cache 显存降低 $h$ 倍（MQA）或 $g$ 倍（GQA）
- **省带宽**：decode 阶段访存压力大幅降低（主要瓶颈）
- **代价**：
  1. 精度损失：MQA 约 2-5%，GQA 约 1-2%
  2. 表达能力：多个头共享 KV，子空间多样性降低
  3. 训练不稳定：需要特殊初始化和调整

### 4. Q: 为什么推理侧 KV cache 没有 Q cache？

**A**:
- **因果性**：decode 阶段每步只看之前 token，Q 只需当前步
- **不复用**：每步的 Q 是新 query，历史 Q 无复用价值
- **K/V 复用性**：历史 K/V 在所有后续步都参与计算，存储后避免重复计算

```python
# decode 第 t 步
Q_t = embed(token_t) @ W_Q  # 只有当前 token
K_1_to_t = [K_1, K_2, ..., K_t]  # 历史都参与
V_1_to_t = [V_1, V_2, ..., V_t]
```

### 5. Q: MHA 的主流优化方向有哪些？

**A**:
1. **MQA/GQA**：推理显存优化（LLaMA2/3, PaLM）
2. **FlashAttention**：IO 感知优化，减少 HBM 访问（训练加速）
3. **PagedAttention**：KV cache 分页管理（vLLM）
4. **Sliding Window Attention**：限制注意力范围（长文本）
5. **Linear Attention**：线性复杂度（Performer, Linear Transformer）
6. **Sparse Attention**：稀疏模式（BigBird, Longformer）

### 6. Q: Softmax 数值稳定性怎么保证？

**A**:
1. **Max-trick**：减去最大值防止 overflow
   ```python
   x_stable = x - x.max(dim=-1, keepdim=True)
   ```
2. **避免 FP16 overflow**：在 softmax 之前用 FP32 累加
3. **检查 NaN/Inf**：监控 score 分布，异常样本排查

### 7. Q: MQA/GQA 与 KV cache 的关系？

**A**:
- **MHA**：每层存储 `batch × seq_len × 2 × num_heads × head_dim`
- **MQA**：存储 `batch × seq_len × 2 × 1 × head_dim`，节省 `num_heads` 倍
- **GQA**：存储 `batch × seq_len × 2 × num_groups × head_dim`
- **长序列场景**：seq_len=8K, num_heads=32 时，MQA 可节省 GB 级显存

## 常见错误（至少 3 个）

### 1. **错误：多头能提升计算效率**

**正确理解**：
- 多头不改变 FLOPs（$d_{model}$ 固定时）
- 加速来自硬件并行，非算法优化
- 真正效率提升来自 MQA/GQA 的显存/带宽优化

### 2. **错误：Mask 直接置 0**

**正确做法**：
```python
# 错误：score 置 0 后 softmax 仍会分配概率
scores[mask == 0] = 0

# 正确：置 -inf 或极小值，softmax 后趋近 0
scores = scores.masked_fill(mask == 0, float('-inf'))
# 或
scores = scores.masked_fill(mask == 0, -1e9)
```

### 3. **错误：忘记 contiguous 和 reshape 顺序**

**正确做法**：
```python
# 错误顺序：直接 reshape
context = context.reshape(batch, seq_len, d_model)  # 维度错乱

# 正确：先 transpose 再 contiguous 再 reshape
context = context.transpose(1, 2).contiguous()
context = context.view(batch, seq_len, d_model)
```

### 4. **错误：MQA 训练与 MHA 完全一样**

**正确做法**：
- MQA 需要特殊初始化（K/V 投影参数共享）
- 学习率可能需要调整
- 训练稳定性监控更关键

## 反问面试官的问题

### 1. 技术深度类
- "在长序列场景（如 128K context），你们团队更关注 attention 的显存优化还是计算优化？有哪些实践经验？"
- "GQA 在你们的生产环境中的精度损失是否可接受？是否做过 A/B 测试对比？"

### 2. 业务场景类
- "推理侧 KV cache 的显存占用在你们的模型中占比大概多少？是否考虑过 PagedAttention 或其他分页方案？"
- "在多轮对话场景中，position id 和 KV cache 的管理有哪些坑？如何处理滑动窗口？"

## 自测题

### 口述（能流畅讲清楚的知识点）
1. **MHA 为什么需要多头**：从单头局限性、子空间表达、并行计算三个角度解释
2. **MQA/GQA 的收益与代价**：显存、带宽、精度、适用场景
3. **Softmax 数值稳定性**：max-trick 原理、为什么需要、实现细节

### 手写（5 分钟能写出的代码/公式）
1. **Scaled Dot-Product Attention 公式**：写出完整公式并解释每项含义
2. **MHA 的 forward 流程**：从输入到输出的维度变化（含 reshape/transpose）
3. **Causal Mask 实现**：用 PyTorch 实现上三角 mask
4. **KV cache 显存估算公式**：给定 batch_size、seq_len、num_layers、num_heads、head_dim，计算显存占用

```python
# KV cache 显存估算（单精度 FP16）
def estimate_kv_cache_memory(batch_size, seq_len, num_layers, num_heads, head_dim, dtype='fp16'):
    bytes_per_element = 2 if dtype == 'fp16' else 4
    # 每层：K + V，每个元素 head_dim 维
    memory_per_layer = batch_size * seq_len * 2 * num_heads * head_dim * bytes_per_element
    total_memory = num_layers * memory_per_layer
    return total_memory / (1024**3)  # 返回 GB
```

## 标签

#Transformer #MHA #MQA #GQA #Attention #KV_cache #handwrite #derive #阿里 #百度 #字节 #Infra

## 相关文档

- [[01-Transformer总览]] - Transformer 整体架构与 forward 流程
- [[03-FFN与Normalization]] - FFN 与 RMSNorm/LayerNorm 细节
- [[../09-推理与Infra/01-KV-Cache]] - KV cache 详细分析与显存估算
- [[../09-推理与Infra/02-Paged-Attention]] - vLLM 的 PagedAttention 机制
- [[../10-FlashAttention/01-FlashAttention原理]] - FlashAttention 的 IO 优化