# Transformer 总览：架构对比与 Forward 链路

## 一句话结论
Decoder-only 是当前 LLM 主流架构，结构简化（单向 attention + causal mask），训练目标统一（next token prediction），适合生成任务；Encoder-Decoder 保留双向编码，在理解任务和短序列翻译上有优势。Forward 链路为：Embedding → 位置编码 → 多层（Attention → FFN → Residual + Norm）→ logits。

## 核心定义/公式

### Decoder-only 架构
```
输入: x ∈ R^(batch×seq_len×d_model)
输出: logits ∈ R^(batch×seq_len×vocab_size)

Forward 链路:
x_embed = Embedding(token_ids)              # [b, s, d]
x_pos = x_embed + RoPE(x_embed)             # 位置编码
for layer in layers:
    x = x + Attention(LayerNorm(x))         # Pre-Norm 结构
    x = x + FFN(LayerNorm(x))
logits = Linear(x)                          # 输出层
```

### Decoder-only vs Encoder-Decoder 对比

| 组件 | Decoder-only | Encoder-Decoder |
|------|--------------|-----------------|
| Encoder | 无 | 双向 attention，无 mask |
| Decoder | 单向 attention + causal mask | 单向 attention + cross-attention |
| 位置 | 全部在 decoder | encoder + decoder 都有 |
| 训练目标 | Next token prediction | Seq2seq (翻译/摘要) |
| 参数共享 | 单流结构 | Encoder/Decoder 独立参数 |

### Causal Mask 实现
```python
# PyTorch 实现
def create_causal_mask(seq_len):
    """生成上三角 mask，避免看到未来 token"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # [seq_len, seq_len]

# 应用到 attention score
attn_scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
attn_scores = attn_scores + mask  # 加 mask 后 softmax
attn_weights = F.softmax(attn_scores, dim=-1)
```

## 为什么（3 个因果链）

### 1. 为什么主流 LLM 选择 Decoder-only？

**现象**：GPT、LLaMA、Qwen、DeepSeek 等主流模型都是 Decoder-only，而非 Encoder-Decoder。

**根因**：
- **训练效率**：单流结构参数利用率高，所有参数都参与生成任务训练；Encoder-Decoder 的 encoder 参数在生成时不更新。
- **推理效率**：Decoder-only 的 KV cache 只需缓存 decoder 的 K/V；Encoder-Decoder 需额外缓存 encoder 输出（cross-attention 用）。
- **任务泛化**：生成任务是 LLM 核心场景（对话、代码、创作），Decoder-only 的 next token prediction 天然适配。

**结果**：Decoder-only 在生成任务上训练更快、推理更省、工程实现更简单，成为工业界主流。

### 2. 为什么 Encoder-Decoder 在翻译任务仍有优势？

**现象**：翻译模型（如 T5、mT5）仍用 Encoder-Decoder，而非 Decoder-only。

**根因**：
- **双向编码**：Encoder 可双向看上下文，对源语言理解更深（特别是短句子）。
- **分离式注意力**：Cross-attention 让 decoder 专注于目标语言生成，encoder 专注于源语言理解。
- **训练目标匹配**：Seq2seq 任务天然适配 Encoder-Decoder 结构。

**结果**：Encoder-Decoder 在翻译、摘要等短序列理解任务上效果更好，但在长文本生成上劣势明显（encoder 开销大）。

### 3. 为什么 Forward 链路中 Attention 和 FFN 交替出现？

**现象**：Transformer 每层都是 Attention → FFN（或 FFN → Attention）的交替结构，而非堆叠多层相同组件。

**根因**：
- **Attention**：建模序列内的全局依赖关系（token 间交互），参数少（QKV 投影），计算复杂度 O(n²)。
- **FFN**：提供非线性变换，参数多（2 个线性层），计算复杂度 O(n)，增加模型容量。
- **互补性**：Attention 捕获"关系"，FFN 提供"变换"，交替堆叠让模型同时具备关系建模和特征提取能力。

**结果**：交替结构平衡了参数效率和计算效率，单一组件无法同时实现全局依赖和非线性变换。

## 怎么做（可落地步骤）

### 标准 Forward 流程（Decoder-only）

#### 步骤 1：Embedding 层
```python
# 输入: token_ids [batch, seq_len]
# 输出: embeddings [batch, seq_len, d_model]

self.embedding = nn.Embedding(vocab_size, d_model)
x = self.embedding(token_ids)  # 查表得到 embedding
```

**关键点**：
- Embedding 大小：vocab_size × d_model（如 LLaMA-7B: 32000 × 4096）
- 是否与输出层共享权重（tie embedding）：LLaMA/GPT 共享，减少参数

#### 步骤 2：位置编码
```python
# RoPE 实现（旋转位置编码）
def apply_rotary_pos_emb(x, cos, sin):
    """应用旋转位置编码到 Q/K"""
    # x: [batch, seq_len, num_heads, head_dim]
    # cos/sin: [seq_len, head_dim]
    
    x1 = x[..., ::2]  # 偶数维度
    x2 = x[..., ::2]  # 奇数维度
    
    # 旋转操作
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    return x_rotated
```

**关键点**：
- RoPE 注入相对位置信息，外推性好
- 不需要可学习参数，训练和推理一致

#### 步骤 3：Multi-Head Attention
```python
def multi_head_attention(x, mask=None):
    """
    x: [batch, seq_len, d_model]
    mask: [seq_len, seq_len] causal mask
    """
    batch_size, seq_len, d_model = x.shape
    
    # 线性投影
    Q = self.q_proj(x)  # [b, s, num_heads * head_dim]
    K = self.k_proj(x)
    V = self.v_proj(x)
    
    # Reshape 为多头
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention 计算
    attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    
    # 应用 causal mask
    if mask is not None:
        attn_scores = attn_scores + mask
    
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = attn_weights @ V
    
    # 合并多头
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_model)
    
    # 输出投影
    output = self.o_proj(attn_output)
    return output
```

**关键配置**：
- `num_heads`：LLaMA-7B 为 32，GPT-3 为 96
- `head_dim`：通常 d_model / num_heads（如 4096 / 32 = 128）
- GQA：Qwen-72B 用 GQA（num_kv_heads < num_heads），减少 KV cache

#### 步骤 4：FFN（SwiGLU 变体）
```python
class SwiGLU_FFEN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return self.dropout(output)
```

**关键配置**：
- `d_ff`：通常为 d_model × 4（LLaMA 用 ×8/3 ≈ 2.67×）
- SwiGLU 比标准 FFN 多一个线性层，但效果更好
- 参数量：2 × d_model × d_ff（标准 FFN）→ 3 × d_model × d_ff（SwiGLU）

#### 步骤 5：Residual + LayerNorm
```python
# Pre-Norm 结构（LLaMA/GPT 使用）
def transformer_layer(x):
    # 先 Norm 再 Attention
    x = x + self.attention(self.layer_norm1(x))
    # 先 Norm 再 FFN
    x = x + self.ffn(self.layer_norm2(x))
    return x

# Post-Norm 结构（原始 Transformer）
def transformer_layer_post_norm(x):
    x = self.layer_norm1(x + self.attention(x))
    x = self.layer_norm2(x + self.ffn(x))
    return x
```

**关键差异**：
- **Pre-Norm**：梯度流更稳定，训练更深网络更容易（LLaMA 用）
- **Post-Norm**：原始 Transformer 用，需要 warm-up，易梯度消失

#### 步骤 6：输出层
```python
# 输出 logits
if self.tie_embedding:
    # 共享 embedding 权重
    logits = F.linear(x, self.embedding.weight)
else:
    logits = self.lm_head(x)  # 独立输出层

# logits: [batch, seq_len, vocab_size]
```

### 关键配置参数表

| 参数 | LLaMA-7B | LLaMA-70B | GPT-3 | Qwen-72B |
|------|----------|-----------|-------|----------|
| d_model | 4096 | 8192 | 12288 | 8192 |
| num_layers | 32 | 80 | 96 | 80 |
| num_heads | 32 | 64 | 96 | 64 |
| num_kv_heads | 32 | 8 (GQA) | 96 | 8 (GQA) |
| d_ff | 11008 | 28672 | 49152 | 21248 |
| vocab_size | 32000 | 32000 | 50257 | 151936 |
| 参数量 | 7B | 70B | 175B | 72B |

### 代码示例：完整 Transformer Block
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU_FFEN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm + Residual
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        # Embedding
        x = self.embedding(input_ids)
        
        # 生成 causal mask
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(x.device)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

## 权衡分析

### Decoder-only vs Encoder-Decoder

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| Decoder-only | 参数利用率高，推理 KV cache 小，工程实现简单 | 单向编码，理解任务需堆叠更多层 | 生成任务（对话、代码、创作） |
| Encoder-Decoder | 双向编码，短序列理解强，分离式注意力 | 参数利用率低，encoder 输出需缓存，推理慢 | 翻译、摘要等 Seq2seq 任务 |
| Encoder-only (BERT) | 双向编码，理解任务强 | 无法生成，需 fine-tune | 分类、抽取、检索 |

### Pre-Norm vs Post-Norm

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| Pre-Norm | 梯度稳定，可训深层网络，无需 warm-up | 理论表达力略弱 | 深层网络（>12 层） |
| Post-Norm | 表达力更强（原始设计） | 训练不稳定，需 warm-up，易梯度消失 | 浅层网络或特定任务 |

### Tie Embedding vs Untied

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| Tie Embedding | 参数量减少 vocab_size × d_model | 可能限制模型能力 | 大词表模型（LLaMA/GPT） |
| Untied | 表达能力更强 | 参数量增加 | 小词表或特定任务 |

## 高频追问（7 个）

### 1. Q: Transformer 的非线性由什么提供？

**A**: 非线性来源有三个：
1. **Attention 层的 softmax**：归一化操作引入非线性（QK^T 后的 softmax）
2. **FFN 的激活函数**：ReLU/GELU/SiGLU 提供主要非线性变换
3. **LayerNorm**：归一化操作引入非线性（均值和方差的计算）

没有这些非线性，多层线性变换可合并为单层，模型退化为线性模型。

### 2. Q: 为什么用 LayerNorm 而非 BatchNorm？

**A**: 三个原因：
1. **序列长度可变**：BatchNorm 对 batch 维度归一化，不同序列长度难以统一；LayerNorm 对 feature 维度归一化，不受序列长度影响。
2. **小 batch size 训练**：LLM 训练常用小 batch（显存限制），BatchNorm 统计量不准；LayerNorm 不依赖 batch 统计。
3. **推理一致性**：BatchNorm 训练/推理行为不同（需要 running mean/var）；LayerNorm 训练/推理一致。

### 3. Q: Decoder-only 的 causal mask 为什么不影响性能？

**A**: 
- **训练时**：每个 token 只能看历史 token，看似限制信息流，但模型通过多层堆叠仍能捕获全局信息（间接依赖）。
- **推理时**：与训练一致，避免信息泄漏，保证自回归生成的正确性。
- **经验证据**：GPT/LLaMA 等模型已证明 Decoder-only 在长文本任务上效果不差于 Encoder-Decoder。

### 4. Q: Attention 和 FFN 哪个参数更多？哪个计算量更大？

**A**: 以 LLaMA-7B 为例（d_model=4096, num_heads=32, d_ff=11008）：

**参数量**：
- Attention：4 × d_model²（QKV+O 投影）= 4 × 4096² ≈ 67M
- FFN（SwiGLU）：3 × d_model × d_ff = 3 × 4096 × 11008 ≈ 135M
- **FFN 参数约为 Attention 的 2 倍**

**计算量**（seq_len=2048）：
- Attention：O(seq_len² × d_model) ≈ 2048² × 4096 ≈ 17B FLOPs
- FFN：O(seq_len × d_model × d_ff) ≈ 2048 × 4096 × 11008 ≈ 92B FLOPs
- **FFN 计算量约为 Attention 的 5 倍（长序列时）**

**结论**：FFN 提供了模型的主要容量和计算量。

### 5. Q: 为什么现在主流模型都用 SwiGLU 而非标准 FFN？

**A**: SwiGLU 的优势：
1. **门控机制**：门控分支（gate_proj）提供自适应的特征选择，类似 LSTM 的门控思想。
2. **平滑激活**：SiLU（Swish）比 ReLU 更平滑，梯度流更好，训练更稳定。
3. **经验效果**：LLaMA 论文显示 SwiGLU 在相同参数下比标准 FFN 效果好 1-2%。

**代价**：参数量增加 50%（2×d_model×d_ff → 3×d_model×d_ff）。

### 6. Q: 如果让你从头训练一个 7B 模型，如何选择架构？

**A**: 
- **架构**：Decoder-only + Pre-Norm + SwiGLU + RoPE（复刻 LLaMA）
- **Attention**：如果推理场景多，用 GQA（num_kv_heads=8）减少 KV cache
- **词表**：中文场景用更大词表（100k+），英文用 32k 即可
- **d_ff**：用 8/3 × d_model 而非 4×（LLaMA 经验）
- **训练目标**：Next token prediction + 数据配比（代码 30% + 文本 70%）

### 7. Q: Transformer 如何解决梯度消失和梯度爆炸？

**A**: 
1. **Residual Connection**：梯度通过残差路径直接传递，避免多层连乘导致的消失。
2. **LayerNorm**：归一化激活值，稳定梯度尺度，防止梯度爆炸/消失。
3. **Pre-Norm**：Pre-Norm 结构下梯度可直接流过 LN，比 Post-Norm 更稳定。
4. **梯度裁剪**：训练时对梯度范数裁剪（如 max_norm=1.0），防止爆炸。

## 常见错误（5 个）

### 1. 错误：认为 Decoder-only 只能用于生成任务

**错误理解**：Decoder-only 有 causal mask，只能看历史 token，理解能力不如 Encoder。

**正确做法**：
- Decoder-only 通过多层堆叠仍能捕获全局信息（间接依赖）。
- GPT-4、Claude 等模型在理解任务上表现优异，证明 Decoder-only 架构足够强。
- 关键是训练数据和任务设计，而非架构本身。

### 2. 错误：混淆 tie embedding 的实现方式

**错误代码**：
```python
# 错误：tie embedding 时直接共享权重对象
self.lm_head = self.embedding  # 错误！
```

**正确做法**：
```python
# 正确：forward 时复用 embedding 权重
def forward(self, x):
    x = self.embedding(x)
    # ... transformer layers ...
    logits = F.linear(x, self.embedding.weight)  # 复用权重矩阵
```

### 3. 错误：认为 Attention 复杂度是 O(n × d)

**错误理解**：Attention 是线性复杂度。

**正确做法**：
- Attention 计算是 QK^T（矩阵乘法），复杂度为 O(seq_len² × d_model)。
- FFN 才是 O(seq_len × d_model²) 线性复杂度。
- 长序列（>8k）时 Attention 成为瓶颈，需用 FlashAttention 或稀疏注意力优化。

### 4. 错误：忽略 mask 的数据类型

**错误代码**：
```python
# 错误：mask 用 int 类型
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).int()
```

**正确做法**：
```python
# 正确：mask 需要浮点型 -inf，与 attention score 相加
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf')).float()
```

### 5. 错误：误解 Pre-Norm 和 Post-Norm 的计算顺序

**错误代码**：
```python
# 错误：Pre-Norm 写成 Post-Norm
x = self.ln1(x + self.attention(x))  # 这是 Post-Norm！
```

**正确做法**：
```python
# 正确：Pre-Norm 是先 LayerNorm 再 Attention
x = x + self.attention(self.ln1(x))
```

## 反问面试官的问题

### 1. 技术深度类
- "贵团队的模型架构在长上下文场景下做了哪些优化？是用 RoPE scaling、滑动窗口还是其他方案？"
- "推理侧 KV cache 的显存占用是主要瓶颈吗？是否考虑过 GQA 或 MLA 来压缩？"

### 2. 业务场景类
- "如果业务主要是短文本对话（<1k tokens），是否考虑过 Encoder-Decoder 架构，还是统一用 Decoder-only？"
- "模型部署时是否遇到过长序列推理的瓶颈？主要卡在计算（Attention）还是访存（KV cache）？"

## 自测题

### 口述（能流畅讲清楚）
1. **Decoder-only vs Encoder-Decoder** 的核心差异和适用场景
2. **Transformer Forward 链路**的完整流程（从 input_ids 到 logits）
3. **为什么 FFN 参数量比 Attention 大**，但计算量不一定更大
4. **Causal mask 的作用和实现方式**
5. **Pre-Norm vs Post-Norm** 的差异和选择依据

### 手写（5 分钟能写出）
1. **Causal mask 生成函数**：
```python
def create_causal_mask(seq_len):
    # 要求：生成上三角 -inf mask
    pass
```

2. **单层 Transformer Block（Pre-Norm）**：
```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask):
        # 要求：Pre-Norm + Residual 结构
        pass
```

3. **SwiGLU FFN**：
```python
def swiglu_ffn(x, w_gate, w_up, w_down):
    # 要求：实现 down(silu(gate(x)) * up(x))
    pass
```

## 标签
#Transformer #架构 #Decoder-only #Encoder-Decoder #forward链路 #Attention #FFN #RMSNorm #RoPE #Pre-Norm #Post-Norm #阿里 #字节

## 相关文档
- [[02-Attention机制详解]]：MHA/MQA/GQA 原理与实现
- [[03-FFN与SwiGLU]]：FFN 变体与参数设计
- [[04-位置编码与RoPE]]：RoPE 原理与外推
- [[06-模型架构对比]]：LLaMA/Qwen/DeepSeek 架构差异
- [[09-推理与KV Cache]]：推理链路与 cache 优化
