# FFN 与归一化

## 一句话结论
FFN 通过两次线性变换+激活函数提供非线性表达能力，SwiGLU 是门控版 FFN，RMSNorm 比 LayerNorm 少了均值计算更高效且在现代大模型中表现更好。

## 核心定义/公式

### 标准 FFN
```
FFN(x) = Linear2(GELU(Linear1(x)))
```
- **维度**：`Linear1: [d_model, d_ff]`, `Linear2: [d_ff, d_model]`
- **典型比例**：`d_ff = 4 * d_model`（原始 Transformer）
- **参数量**：`2 * d_model * d_ff + d_ff + d_model`（含 bias）

### SwiGLU FFN
```
SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)
         = (xW₁ ⊙ sigmoid(βxW₁)) ⊙ (xW₂)
```
- **结构**：两个投影矩阵 `W₁, W₂`，门控机制
- **维度**：`W₁: [d_model, d_ff]`, `W₂: [d_model, d_ff]`
- **参数量**：`2 * d_model * d_ff`（比标准 FFN 多 50% 参数）
- **典型配置**：`d_ff = 8/3 * d_model`（缩减以平衡参数量）

### RMSNorm
```
RMSNorm(x) = (x / √(mean(x²) + ε)) * γ
```
- **计算**：仅计算平方均值，不求均值
- **参数**：只有缩放参数 `γ`，无偏置
- **复杂度**：比 LayerNorm 少 1 次 mean 操作

### LayerNorm
```
LayerNorm(x) = ((x - μ) / √(σ² + ε)) * γ + β
```
- **计算**：计算均值 `μ` 和方差 `σ²`
- **参数**：缩放参数 `γ` 和偏置 `β`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 标准 FFN
class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# SwiGLU FFN
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or int(8/3 * d_model)  # LLaMA 默认
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # output
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        gate = F.silu(self.w1(x))  # SiLU = Swish(β=1)
        value = self.w2(x)
        x = gate * value  # element-wise gating
        x = self.dropout(x)
        x = self.w3(x)
        return x

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma

# LayerNorm (reference)
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
```

## 为什么（2-3 个因果链）

### 1. FFN 为什么需要？Attention 为什么不够？
- **现象**：Transformer 中每个 Block = Attention + FFN
- **根因**：Attention 是线性操作（加权求和），只能做信息聚合，无法引入新的非线性变换
- **结果**：FFN 提供逐位置（position-wise）的非线性映射，是模型容量的主要来源
- **定量**：FFN 参数量占 Transformer 总参数的 2/3（`2 * d_model * d_ff` vs `4 * d_model * d_model`）

### 2. SwiGLU 为什么比 GELU 好？
- **现象**：LLaMA/PaLM 等现代大模型从 GELU 切换到 SwiGLU
- **根因**：
  - 门控机制允许模型自适应地过滤信息（类似 LSTM 的 gate）
  - Swish 激活函数在负值区有轻微的非零输出，避免梯度完全消失
  - GLU 提供更灵活的特征组合方式
- **结果**：相同参数量下 SwiGLU 性能优于 GELU（Papers with Code 显示 ~2% 提升）

### 3. RMSNorm 为什么比 LayerNorm 好？
- **现象**：LLaMA/GPT-NeoX/Qwen 等现代模型都用 RMSNorm
- **根因**：
  - LayerNorm 计算均值和方差，均值中心化对 NLP 任务收益有限
  - RMSNorm 仅缩放特征尺度，保留相对方向信息
  - 少 1 次 mean 计算，在长序列/大 batch 下更高效
- **结果**：计算更快（~5-10%），性能相当或略好，显存占用更小

## 怎么做（可落地步骤）

### 标准做法

#### 1. FFN 维度选择
```python
# 原始 Transformer (Vaswani et al., 2017)
d_model = 512
d_ff = 4 * d_model  # 2048

# LLaMA-7B (d_model=4096)
d_model = 4096
d_ff = int(8/3 * d_model)  # 11008 for SwiGLU

# Qwen-7B (d_model=4096)
d_model = 4096
d_ff = int(8/3 * d_model)  # 类似 LLaMA
```

#### 2. 归一化位置选择
```python
# Post-LN (原始 Transformer)
x = x + FFN(LayerNorm(x))  # LN 在残差后

# Pre-LN (现代主流，LLaMA/Qwen)
x = x + FFN(x)
x = RMSNorm(x)  # LN 在残差前，训练更稳定
```

#### 3. 激活函数选择
```python
# 标准 FFN: GELU (BERT, GPT-2)
activation = F.gelu

# SwiGLU FFN: SiLU/Swish (LLaMA, PaLM)
activation = F.silu  # torch 1.7+ 
```

### 关键配置/参数

| 配置项 | 推荐值 | 原因 |
|--------|--------|------|
| `d_ff` (标准 FFN) | `4 * d_model` | 平衡容量与计算 |
| `d_ff` (SwiGLU) | `8/3 * d_model` | 平衡参数量（SwiGLU 参数更多） |
| `eps` (RMSNorm) | `1e-6` 或 `1e-5` | 避免除零，影响数值稳定性 |
| `dropout` | `0.1` (训练) / `0.0` (推理) | 正则化强度 |
| bias | `False` (SwiGLU) | 减少参数，提升效率 |

### 代码示例

#### 完整的 Transformer Block (Pre-LN + RMSNorm + SwiGLU)
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or int(8/3 * d_model)
        
        # Attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        
        # FFN (SwiGLU)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # Pre-LN Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(x)
        
        # Pre-LN FFN (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        x = residual + self.dropout(x)
        
        return x
```

#### 手写 FFN 要点（面试常考）
```python
def handwrite_ffn_demo():
    """面试手写 FFN 的关键点"""
    
    # 1. 维度定义
    batch_size, seq_len, d_model, d_ff = 2, 10, 64, 256
    
    # 2. 创建参数
    W1 = torch.randn(d_model, d_ff)  # [64, 256]
    b1 = torch.zeros(d_ff)
    W2 = torch.randn(d_ff, d_model)  # [256, 64]
    b2 = torch.zeros(d_model)
    
    # 3. 输入
    x = torch.randn(batch_size, seq_len, d_model)  # [2, 10, 64]
    
    # 4. FFN 前向传播
    h = x @ W1 + b1              # [2, 10, 256]
    h = F.gelu(h)                # 激活函数
    out = h @ W2 + b2            # [2, 10, 64]
    
    # 5. 维度检查
    assert out.shape == x.shape  # 输出维度必须回到 d_model
    
    return out
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **标准 FFN (GELU)** | 简单稳定、生态成熟、易于调试 | 容量相对较低、无门控机制 | 中小模型（<7B）、快速原型、研究实验 |
| **SwiGLU FFN** | 性能更好（+2%）、门控机制、自适应过滤 | 参数多 50%、计算量增加、实现复杂 | 大模型（≥7B）、追求极致性能、LLaMA 系列 |
| **RMSNorm** | 计算快 5-10%、显存省、无均值偏置 | 少了中心化、对分布变化稍敏感 | 现代大模型、长序列、大规模训练 |
| **LayerNorm** | 理论更完备、对分布敏感、预训练模型兼容 | 计算慢、显存多 | 传统模型（BERT）、迁移学习、调试阶段 |
| **Pre-LN** | 训练稳定、梯度流动好 | 可能轻微性能下降 | 深层网络（>12层）、大模型训练 |
| **Post-LN** | 理论优雅、原始 Transformer | 训练不稳定、需要 warmup | 浅层网络、研究对比 |

## 高频追问（至少 5 个）

### 1. Q: FFN 在 Transformer 中占多少参数量？
A: 占总参数量的约 2/3。具体计算：
- 每个 FFN block：`2 * d_model * d_ff`
- 每个 Attention block：`4 * d_model^2`（Q/K/V/O 四个投影）
- 以 LLaMA-7B 为例：FFN 参数 ≈ 24B / 7B ≈ 3.4B，约 50%（因 d_ff 调小）

### 2. Q: 为什么 SwiGLU 参数更多但性能更好？
A: 参数增加 50%，但门控机制带来：
- 更灵活的特征选择（类似 LSTM gate）
- Swish 激活在负值区有梯度（避免 ReLU 死亡）
- 实验表明 GLU 类激活普遍优于 GELU（GLU Variants Paper, 2020）

### 3. Q: RMSNorm 和 LayerNorm 的数值稳定性有差异吗？
A: 有差异但不大：
- LayerNorm：均值中心化后方差归一化，理论上更稳定
- RMSNorm：仅缩放尺度，保留方向信息
- 实践中 RMSNorm 通过 `eps` 参数（通常 `1e-6`）保证数值稳定
- 关键是现代框架（PyTorch）实现都用了数值稳定的算法

### 4. Q: Pre-LN 和 Post-LN 怎么选？
A: 
- **Pre-LN**（推荐）：训练稳定、不需要 warmup、梯度流动好，缺点是理论性能可能略差
- **Post-LN**：需要 learning rate warmup、深层网络训练困难，适合浅层网络或对比实验
- **经验法则**：12 层以上必须用 Pre-LN，现代大模型（LLaMA/Qwen）全用 Pre-LN

### 5. Q: 手写 FFN 时维度怎么对？
A: 
```python
# 输入: [batch, seq_len, d_model]
# W1: [d_model, d_ff]
# 中间: [batch, seq_len, d_ff]
# W2: [d_ff, d_model]
# 输出: [batch, seq_len, d_model]
```
关键是输入输出维度必须一致（`d_model`），中间维度 `d_ff` 可以灵活选择（通常 `4 * d_model` 或 `8/3 * d_model`）

### 6. Q: 梯度消失和梯度爆炸怎么解决？
A: 
- **梯度消失**：
  - 残差连接（ResNet）：梯度直通，`∂L/∂x ≈ 1 + ∂L/∂F(x)`
  - LayerNorm/RMSNorm：归一化后梯度稳定
  - Pre-LN：归一化在残差前，梯度流更好
  - 门控激活（SwiGLU）：避免 ReLU 死亡
  
- **梯度爆炸**：
  - 梯度裁剪：`grad = grad * min(1, max_norm / ||grad||)`
  - LayerNorm/RMSNorm：归一化限制数值范围
  - 合理的学习率：配合 warmup 和 decay
  - 权重初始化：Xavier/He 初始化

### 7. Q: FFN 为什么叫 position-wise？
A: 
- FFN 对每个位置独立应用相同的变换
- 没有跨位置的信息交互（不像 Attention）
- 数学上：`FFN(x_i)` 只依赖于 `x_i`，不依赖于其他 `x_j`
- 实现上：用 1x1 卷积或 `nn.Linear`（等价）

### 8. Q: SwiGLU 的 SiLU 和 GELU 有什么区别？
A: 
- **GELU**：`x * Φ(x)`，其中 `Φ` 是标准正态分布 CDF
- **SiLU/Swish**：`x * sigmoid(βx)`，β=1 时叫 SiLU
- **差异**：
  - SiLU 计算更快（sigmoid 比误差函数快）
  - SiLU 在负值区梯度更大（Swish 论文显示）
  - GELU 在 BERT 中效果略好，但 SiLU+GLU 组合更优

## 常见错误（至少 3 个）

### 1. **错误：FFN 维度设置不当**
**错误做法**：
```python
# 错误：d_ff 太小（容量不足）
d_ff = d_model  # 应该是 4 * d_model

# 错误：d_ff 太大（显存爆炸）
d_ff = 16 * d_model  # 显存占用 4 倍
```

**正确做法**：
```python
# 标准 FFN
d_ff = 4 * d_model

# SwiGLU FFN（参数平衡）
d_ff = int(8/3 * d_model)  # 或向上取整到 2 的倍数
```

### 2. **错误：手写 FFN 忘记维度对齐**
**错误做法**：
```python
# 错误：输出维度不对
def ffn_error(x):
    h = Linear1(x)      # [batch, seq, d_ff]
    h = gelu(h)
    out = Linear2(h)     # [batch, seq, d_ff] ❌
    return out  # 维度错误！应该是 [batch, seq, d_model]
```

**正确做法**：
```python
def ffn_correct(x):
    # x: [batch, seq_len, d_model]
    h = Linear1(x)       # [batch, seq_len, d_ff]
    h = gelu(h)
    out = Linear2(h)     # [batch, seq_len, d_model]
    return out  # 维度回到 d_model
```

### 3. **错误：Pre-LN/Post-LN 顺序搞反**
**错误做法**：
```python
# 错误：想用 Pre-LN 但写成 Post-LN
x = x + Attention(x)      # 残差
x = LayerNorm(x)          # 归一化在后面 → Post-LN ❌

# 或者 Pre-LN 位置错误
x = LayerNorm(x + Attention(x))  # ❌ 应该是 x + Attention(LN(x))
```

**正确做法**：
```python
# Pre-LN（现代主流）
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-LN（原始 Transformer）
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### 4. **错误：RMSNorm 实现 bug**
**错误做法**：
```python
# 错误 1：忘记 eps
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
return x / rms * gamma  # 数值不稳定

# 错误 2：维度错误
rms = torch.sqrt(torch.mean(x ** 2, dim=-1))  # 缺少 keepdim=True
return x / rms * gamma  # 广播错误

# 错误 3：gamma 初始化错误
self.gamma = nn.Parameter(torch.zeros(d_model))  # ❌ 应该初始化为 1
```

**正确做法**：
```python
def forward(self, x):
    # x: [batch, seq_len, d_model]
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
    return x / rms * self.gamma  # gamma 初始化为 1
```

### 5. **错误：SwiGLU 的三个矩阵理解错误**
**错误理解**：
- 以为 `w1` 和 `w2` 是一个矩阵的分解
- 以为 `w3` 是输出层的 bias

**正确理解**：
```python
# SwiGLU 有 3 个矩阵：
# w1: gate projection [d_model, d_ff]
# w2: value projection [d_model, d_ff]
# w3: output projection [d_ff, d_model]

# 前向传播：
gate = silu(w1(x))   # 门控信号
value = w2(x)        # 值信号
output = w3(gate * value)  # 门控后投影回 d_model
```

## 反问面试官的问题

### 技术深度类
1. "您团队在训练大模型时，Pre-LN 和 Post-LN 有对比过吗？深层网络稳定性差异明显吗？"
2. "SwiGLU 相比 GELU 在您的场景中性能提升有多少？是否值得参数量增加的代价？"
3. "RMSNorm 在长序列训练时数值稳定性如何？是否遇到过 NaN 问题？"

### 业务场景类
1. "推理部署时，FFN 是计算密集型还是访存密集型？有没有做过量化压缩？"
2. "在您的业务中，模型容量（FFN 维度）和训练成本的权衡点在哪里？"
3. "有没有遇到过 FFN 输出异常（NaN/Inf）的情况？通常怎么排查？"

## 自测题

### 口述（能流畅讲清楚的知识点）
1. FFN 在 Transformer 中的作用是什么？为什么 Attention 不够？
2. SwiGLU 和 GELU 的区别是什么？为什么现代大模型倾向 SwiGLU？
3. RMSNorm 和 LayerNorm 的计算差异是什么？为什么 RMSNorm 更快？
4. Pre-LN 和 Post-LN 的区别是什么？什么时候用哪个？
5. 手写 FFN 的维度推导流程（从输入到输出）

### 手写（5 分钟能写出的代码/公式）
1. 手写标准 FFN（含维度注释）
2. 手写 SwiGLU FFN（三个矩阵）
3. 手写 RMSNorm（含 eps 处理）
4. 手写完整的 Transformer Block（Pre-LN + RMSNorm + SwiGLU）
5. 计算给定模型的 FFN 参数量（`d_model=4096, d_ff=11008`）

## 标签
#Transformer #FFN #handwrite #字节 #阿里 #腾讯

## 相关文档
- [[01-Transformer总览]]：Transformer 整体架构与 forward 流程
- [[02-Attention机制]]：Attention 层的结构与实现
- [[04-Tokenizer与Embedding]]：Tokenizer 与 Embedding 层配置