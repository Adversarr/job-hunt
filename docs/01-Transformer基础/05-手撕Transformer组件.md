# 手撕 Transformer 组件

## 一句话结论
MHA 手写关键：QKV 投影 → 分头 → 缩放点积（加 mask）→ softmax（max-trick 稳定）→ 加权求和 → 合并头；Decoder layer 关键：MHA（causal mask）→ Add & Norm → FFN → Add & Norm，确保因果性。

## 核心定义/公式

### Multi-Head Attention (MHA) 核心公式
```
Attention(Q, K, V) = softmax(QK^T / √d_k + M) V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 形状推导
- Input: `[batch, seq_len, d_model]`
- Q/K/V 投影: `[batch, seq_len, d_model] → [batch, seq_len, d_k]`
- 分头: `[batch, seq_len, d_k] → [batch, num_heads, seq_len, head_dim]`
- Attention scores: `[batch, num_heads, seq_len, seq_len]`
- Output: `[batch, seq_len, d_model]`

### Causal Mask
```python
# 上三角为 -inf，下三角为 0
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
```

### Softmax 数值稳定性（Max Trick）
```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 防止 exp 溢出
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### Cross-Entropy Loss
```python
def cross_entropy_loss(logits, labels):
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    """
    # Softmax
    probs = softmax(logits, axis=-1)
    # 取对应 token 的概率
    batch_size, seq_len, vocab_size = logits.shape
    probs_flat = probs.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    # 避免数值溢出
    probs_clipped = np.clip(probs_flat, 1e-10, 1.0)
    # 计算负对数似然
    loss = -np.log(probs_clipped[np.arange(batch_size * seq_len), labels_flat])
    return loss.mean()
```

## 为什么（2-3 个因果链）

### 1. 为什么 MHA 需要多头而不是单头？
- **现象**：单头注意力只能学习一种注意力模式
- **根因**：不同头可以关注不同的语义子空间（句法、语义、位置等）
- **结果**：多头允许模型同时捕捉多种依赖关系，提升表达能力

### 2. 为什么 softmax 前要除以 √d_k？
- **现象**：点积结果随维度增大而增大，导致 softmax 梯度消失
- **根因**：假设 Q/K 元素独立且均值为 0 方差为 1，点积方差为 d_k
- **结果**：除以 √d_k 使方差归一化到 1，保证 softmax 输出分布稳定

### 3. 为什么 causal mask 要用 -inf 而不是 0？
- **现象**：未来位置需要被完全屏蔽
- **根因**：softmax 公式 exp(score_i) / Σexp(score_j)，若 mask=0 则未来位置仍有贡献
- **结果**：使用 -inf 使得 exp(-inf) = 0，未来位置在 softmax 中权重为 0，实现严格因果性

## 怎么做（可落地步骤）

### 手写 MHA（NumPy 实现）

```python
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, mask=None):
    """
    Args:
        X: [batch, seq_len, d_model] 输入
        W_Q, W_K, W_V: [d_model, d_k] 投影矩阵
        W_O: [d_model, d_model] 输出投影
        mask: [seq_len, seq_len] causal mask (可选)
    
    Returns:
        output: [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = X.shape
    num_heads = 8
    head_dim = d_model // num_heads
    
    # 1. 线性投影
    Q = X @ W_Q  # [batch, seq_len, d_k]
    K = X @ W_K
    V = X @ W_V
    
    # 2. 分头：[batch, seq_len, d_k] -> [batch, num_heads, seq_len, head_dim]
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # 3. 计算注意力分数
    d_k = head_dim
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # [batch, num_heads, seq_len, seq_len]
    
    # 4. 应用 mask（关键！）
    if mask is not None:
        scores = scores + mask  # broadcast mask 到所有 batch 和 head
    
    # 5. Softmax（带 max-trick 数值稳定性）
    attn_weights = softmax(scores, axis=-1)
    
    # 6. 加权求和
    context = attn_weights @ V  # [batch, num_heads, seq_len, head_dim]
    
    # 7. 合并头
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 8. 输出投影
    output = context @ W_O
    
    return output, attn_weights

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 手写 Transformer Decoder Layer

```python
def decoder_layer(X, params, mask=None):
    """
    Args:
        X: [batch, seq_len, d_model]
        params: 包含所有权重和 norm 参数的字典
        mask: causal mask [seq_len, seq_len]
    
    Returns:
        output: [batch, seq_len, d_model]
    """
    d_model = X.shape[-1]
    
    # 1. Masked Multi-Head Self-Attention
    attn_out, _ = multi_head_attention(
        X, params['W_Q'], params['W_K'], params['W_V'], 
        params['W_O'], mask
    )
    
    # 2. Add & Norm (残差连接 + LayerNorm)
    X = layer_norm(X + attn_out, params['ln1_gamma'], params['ln1_beta'])
    
    # 3. FFN (Feed-Forward Network)
    ffn_out = X @ params['W_ff1'] + params['b_ff1']  # [batch, seq_len, d_ff]
    ffn_out = gelu(ffn_out)  # 激活函数
    ffn_out = ffn_out @ params['W_ff2'] + params['b_ff2']  # [batch, seq_len, d_model]
    
    # 4. Add & Norm
    output = layer_norm(X + ffn_out, params['ln2_gamma'], params['ln2_beta'])
    
    return output

def layer_norm(X, gamma, beta, eps=1e-6):
    """Layer Normalization"""
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return gamma * X_norm + beta

def gelu(x):
    """GELU 激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

### Causal Mask 创建

```python
def create_causal_mask(seq_len):
    """
    创建因果 mask：上三角为 -inf，下三角为 0
    确保每个位置只能看到自己和之前的 token
    
    Returns:
        mask: [seq_len, seq_len]
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    return mask

# 示例
# seq_len = 4 时
# [[0, -inf, -inf, -inf],
#  [0,    0, -inf, -inf],
#  [0,    0,    0, -inf],
#  [0,    0,    0,    0]]
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| num_heads | 8-32 | 头数需整除 d_model，过多头会增加计算和内存开销 |
| head_dim | 64-128 | 通常 d_model / num_heads，影响注意力细粒度 |
| d_ff | 4 × d_model | FFN 中间层维度，标准配置（如 LLaMA 用 2.67×） |
| mask value | -1e9 或 -inf | 足够小使得 exp(mask) ≈ 0，避免梯度问题 |
| softmax eps | 1e-10 | 防止 log(0) 导致 NaN |

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **多头 vs 单头** | 多种注意力模式，表达能力强 | 计算量增加，头间需要同步 | 大模型标配，小模型可减少头数 |
| **GQA/MQA vs MHA** | 减少 KV cache，推理加速 | 表达能力略降，训练需要专门设计 | 推理密集场景（LLaMA-2, Qwen 等） |
| **Pre-LN vs Post-LN** | 训练更稳定，梯度流畅 | 可能轻微影响性能 | 现代 Transformer 默认选择（GPT, LLaMA） |
| **GELU vs ReLU** | 更平滑的非线性，性能略好 | 计算稍慢 | 主流大模型标配 |

## 高频追问（至少 5 个）

### Q1: MHA 为什么不能提高计算效率？
A: MHA 将 d_model 分成多个头，每个头计算量减小，但总计算量不变。头数增加不改变整体 FLOPs，只是为了增加表达能力。效率提升来自并行化（所有头并行计算）和缓存优化（GQA/MQA 减少重复计算）。

### Q2: 如何理解"多头注意力没有提高计算效率"？
A: 从公式看，单头 QK^T 维度 [seq_len, d_model]；多头 8 个头，每头 [seq_len, head_dim] 其中 head_dim = d_model / 8。矩阵乘法 FLOPs 相同：seq_len × d_model × seq_len。差异在于：
- **并行性**：多头可并行，硬件利用率更高
- **语义多样性**：不同头学不同模式
- **缓存优化**：GQA/MQA 在推理时减少 KV cache

### Q3: 为什么 Decoder 的 mask 是上三角而不是下三角？
A: 取决于 mask 加法方向。常见写法：`scores = scores + mask`，此时 mask 上三角为 -inf，下三角为 0，结果就是未来位置被屏蔽。如果用乘法 `scores = scores * mask`，则下三角为 1，上三角为 0。本质一样，只是实现方式不同。

### Q4: Softmax 的 max-trick 为什么能保证数值稳定性？
A: Softmax 公式 exp(x_i) / Σ exp(x_j)。若 x_i 很大，exp(x_i) 会溢出为 inf。max-trick 将每个元素减去最大值：
- exp(x_i - max) ≤ exp(0) = 1，不会上溢
- 依然保证概率和为 1（分子分母同时除以 exp(max)）

### Q5: FFN 为什么用两个线性层而不是一个？
A: 一个线性层只是线性变换，无法增加表达能力。两个线性层：
1. 第一层 d_model → d_ff（通常 4×），**升维**增加非线性表达
2. 第二层 d_ff → d_model，**降维**恢复维度
中间有激活函数，整体等价于一个两层的 MLP，提供更强的非线性映射能力。

### Q6: 手写时 batch 维度如何处理？
A: NumPy 中矩阵乘法天然支持 batch：
- `[batch, seq, d_model] @ [d_model, d_k] → [batch, seq, d_k]`（广播）
- 多头需要在分头后显式 transpose：`[batch, seq, num_heads, head_dim] → [batch, num_heads, seq, head_dim]`
- 注意维度顺序：PyTorch 默认 `(batch, num_heads, seq, head_dim)`

### Q7: MQA 和 GQA 的手撕差异是什么？
A: 
- **MHA**：每个头独立 Q、K、V 投影矩阵
- **MQA**：所有头共享 **同一个 K 和 V** 投影矩阵，只有 Q 分头
- **GQA**：K、V 被分组，每组内的头共享一个 K 和 V（介于 MHA 和 MQA 之间）

```python
# MQA 示例
# K, V 不分头，只有 Q 分头
K = X @ W_K  # [batch, seq_len, head_dim]，所有头共享
V = X @ W_V
Q = X @ W_Q  # [batch, seq_len, d_model]
Q = Q.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)

# scores 计算
scores = Q @ K.unsqueeze(1).transpose(-1, -2) / sqrt(d_k)
# K 需要 broadcast 到 [batch, num_heads, seq, head_dim]
```

## 常见错误（至少 3 个）

### 错误 1: Softmax 前忘记除以 √d_k
**描述**：直接计算 `Q @ K.T` 后 softmax，导致数值过大。
**正确做法**：`scores = QK^T / np.sqrt(d_k)`，使方差归一化到 1。

### 错误 2: Mask 使用 0 而不是 -inf
**描述**：`mask[mask_position] = 0`，导致 softmax 后未来位置仍有概率。
**正确做法**：`mask = np.triu(...) * (-1e9)`，使得 `exp(-inf) = 0`。

### 错误 3: 维度 reshape/transpose 顺序错误
**描述**：分头时直接 reshape 不 transpose，导致注意力计算错误。
**正确做法**：
```python
# 错误
Q.reshape(batch, num_heads, seq, head_dim)  # 维度对但语义错

# 正确
Q.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
```

### 错误 4: 忘记 LayerNorm 的 epsilon
**描述**：`X_norm = (X - mean) / np.sqrt(var)`，当 var=0 时除零错误。
**正确做法**：`X_norm = (X - mean) / np.sqrt(var + eps)`，eps=1e-6 或 1e-5。

### 错误 5: Cross-Entropy 没做 shift（标签对齐）
**描述**：训练时 logits 是预测下一个 token，标签需要左移。
**正确做法**：
```python
# logits: [batch, seq_len, vocab_size] 预测 next token
# labels: [batch, seq_len] 是当前 token
# 需要对齐：logits[i] 预测 labels[i+1]
logits = logits[:, :-1, :]  # 去掉最后一个
labels = labels[:, 1:]       # 去掉第一个
```

## 反问面试官的问题

### 技术深度类
1. 在实际项目中，你们如何选择多头注意力的头数？是基于经验还是通过实验调优？
2. GQA/MQA 在训练和推理中的权衡如何考虑？什么时候值得牺牲表达能力换速度？
3. LayerNorm 放在 attention 前后（Pre-LN vs Post-LN）的选择依据是什么？

### 业务场景类
1. 你们团队在长上下文场景中，如何优化 causal mask 的显存占用？
2. 实际部署时，会采用哪些手写算子优化（如 FlashAttention）？收益如何？

## 自测题

### 口述（能流畅讲清楚的知识点）
- MHA 的分头和合并过程，维度变化
- Causal mask 的作用和实现方式
- Softmax 数值稳定性的原理
- FFN 为什么需要两个线性层
- Pre-LN 和 Post-LN 的区别

### 手写（5 分钟能写出的代码/公式）
- 只用 NumPy 实现 MHA（含 mask 和 softmax）
- 创建 causal mask 的函数
- LayerNorm 实现
- Cross-Entropy Loss 实现
- Decoder Layer 的前向传播（伪代码）

## 标签
#handwrite #MHA #Transformer #字节 #腾讯 #阿里

## 相关文档
- [[01-Transformer总览]]：整体架构与各组件关系
- [[02-Attention机制]]：MHA/MQA/GQA 原理对比
- [[03-FFN与归一化]]：FFN 结构与 RMSNorm/LayerNorm
- [[../13-手撕算法题/06-工程手撕题型]]：其他手撕题目汇总