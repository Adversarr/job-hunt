# KV Cache 核心

## 一句话结论
KV cache 存储每层所有历史 token 的 Key 和 Value 向量，避免 decode 阶段重复计算；显存占用随 batch_size × seq_len × num_layers × num_heads × head_dim 线性增长，是推理 OOM 的首要原因。

## 核心定义/公式

### KV Cache 存储内容
- **存储对象**：每层 Transformer 的 Key 和 Value 矩阵
- **切分粒度**：按 attention head 切分（每个 head 独立缓存）
- **数据类型**：通常为 FP16 或 BF16（可量化到 INT8/INT4）

### 形状定义
```python
# 单层 KV cache 形状
K_cache: [batch_size, num_heads, seq_len, head_dim]
V_cache: [batch_size, num_heads, seq_len, head_dim]

# 其中：
# - batch_size: 批次大小
# - num_heads: 注意力头数（MHA=GQA的头数，MQA=1，GQA=分组头数）
# - seq_len: 已生成的序列长度（prompt + generated tokens）
# - head_dim: 每个头的维度（通常为 d_model / num_heads）
```

### 显存估算公式
```python
# 总显存占用（字节）
kv_cache_memory = (
    2  # K + V
    * batch_size
    * seq_len
    * num_layers
    * num_heads
    * head_dim
    * dtype_size  # FP16=2, BF16=2, INT8=1, INT4=0.5
)

# 示例：LLaMA-7B (FP16)
# batch_size=32, seq_len=2048, num_layers=32, num_heads=32, head_dim=128
# kv_cache_memory = 2 × 32 × 2048 × 32 × 32 × 128 × 2 bytes
#                 = 34.6 GB
```

### GQA/MQA 的显存节省
```python
# MHA: 每个 head 独立 KV cache
kv_cache_mha = 2 × batch × seq × layers × heads × head_dim

# MQA: 所有 head 共享一组 KV
kv_cache_mqa = 2 × batch × seq × layers × 1 × head_dim

# GQA: 分组共享 KV
kv_cache_gqa = 2 × batch × seq × layers × (heads // group_size) × head_dim

# 节省比例
# MQA: 节省 (num_heads - 1) / num_heads ≈ 97% (LLaMA-7B: 32 heads)
# GQA: 节省 (num_heads - num_groups) / num_heads (如 group_size=8, 节省 87.5%)
```

## 为什么（2-3 个因果链）

### 1. 为什么需要 KV Cache？
**因果链**：Decoder 自回归生成 → 每步需要所有历史 token → 注意力计算需要 Q·K^T → **根因**：K 和 V 只依赖历史，不依赖未来 → **结果**：缓存历史 K/V 避免重复计算

**推导过程**：
```python
# 第 t 步生成
Q_t = W_q @ x_t  # 只需要当前 token
K_t = W_k @ x_t  # 当前 token 的 K
V_t = W_v @ x_t  # 当前 token 的 V

# 注意力计算需要所有历史
Attention(Q_t, [K_1, K_2, ..., K_t], [V_1, V_2, ..., V_t])

# 无 cache：每步重新计算 K_1...K_{t-1}
# 有 cache：直接读取 cache 中 K_1...K_{t-1}，只计算 K_t
```

### 2. 为什么没有 Q Cache？
**因果链**：Q 只与当前 token 相关 → **根因**：自回归生成中，Q 每步都变化 → **结果**：缓存的 Q 无法复用

**详细解释**：
- Q 是 query，代表"我要查询什么"
- 第 t 步的 Q_t 只由 x_t 决定
- 第 t+1 步的 Q_{t+1} 由 x_{t+1} 决定
- 前后步的 Q 没有关联，缓存无意义

**对比**：
```python
# K 和 V：可以复用
K_i = W_k @ x_i  # 只依赖第 i 个输入，第 t 步生成时仍需要
V_i = W_v @ x_i  # 同上

# Q：无法复用
Q_t = W_q @ x_t  # 第 t 步的 Q 只用于第 t 步
# 第 t+1 步会生成新的 Q_{t+1}，旧的 Q_t 不再需要
```

### 3. 为什么 Batch 一大就 OOM？
**因果链**：KV cache 显存与 batch 线性相关 → **根因**：推理时显存 = 模型权重 + KV cache + 激活 → **结果**：batch 增大 → KV cache 指数级增长（batch × seq_len 同时增长）→ 超过 GPU 显存上限

**显存占用分解**：
```python
# 推理显存占用
total_memory = (
    model_weights        # 固定
    + kv_cache          # 随 batch × seq_len 增长
    + activations       # 随 batch 增长
    + cuda_kernels      # 固定
)

# 关键问题：seq_len 也在增长
# decode 阶段：seq_len = prompt_len + generated_len
# batch=1, seq=4096 时可能 OK
# batch=32, seq=4096 时 KV cache 占用 = 32× 显存
```

## 怎么做（可落地步骤）

### 标准做法

#### 1. KV Cache 初始化（Prefill 阶段）
```python
def prefill_kv_cache(model, input_ids):
    """Prefill 阶段初始化 KV cache"""
    batch_size, seq_len = input_ids.shape
    
    # 初始化空 cache（按层）
    kv_cache = {
        layer_idx: {
            'key': torch.zeros(batch_size, num_heads, seq_len, head_dim),
            'value': torch.zeros(batch_size, num_heads, seq_len, head_dim)
        }
        for layer_idx in range(num_layers)
    }
    
    # Forward 计算并填充 cache
    hidden_states = model.embed_tokens(input_ids)
    
    for layer_idx, layer in enumerate(model.layers):
        # 计算 Q, K, V
        Q = layer.q_proj(hidden_states)
        K = layer.k_proj(hidden_states)
        V = layer.v_proj(hidden_states)
        
        # 填充 cache
        kv_cache[layer_idx]['key'] = K.view(batch_size, num_heads, seq_len, head_dim)
        kv_cache[layer_idx]['value'] = V.view(batch_size, num_heads, seq_len, head_dim)
        
        # 计算 attention（需要 causal mask）
        hidden_states = layer.attention(Q, K, V)
        hidden_states = layer.ffn(hidden_states)
    
    return kv_cache, hidden_states
```

#### 2. Decode 阶段增量更新
```python
def decode_step(model, input_token, kv_cache, position_id):
    """Decode 阶段：单步生成 + 增量更新 cache"""
    batch_size = input_token.shape[0]
    
    # Embedding
    hidden_states = model.embed_tokens(input_token)
    
    for layer_idx, layer in enumerate(model.layers):
        # 只计算当前 token 的 Q, K, V
        Q = layer.q_proj(hidden_states)  # [batch, 1, num_heads, head_dim]
        K = layer.k_proj(hidden_states)
        V = layer.v_proj(hidden_states)
        
        # 增量更新 cache（append 到末尾）
        kv_cache[layer_idx]['key'] = torch.cat([
            kv_cache[layer_idx]['key'],
            K.view(batch_size, num_heads, 1, head_dim)
        ], dim=2)  # seq_len 维度
        
        kv_cache[layer_idx]['value'] = torch.cat([
            kv_cache[layer_idx]['value'],
            V.view(batch_size, num_heads, 1, head_dim)
        ], dim=2)
        
        # Attention：当前 Q × 所有历史 K/V
        K_all = kv_cache[layer_idx]['key']
        V_all = kv_cache[layer_idx]['value']
        
        hidden_states = layer.attention(Q, K_all, V_all)
        hidden_states = layer.ffn(hidden_states)
    
    return kv_cache, hidden_states
```

### 关键配置/参数

#### 显存优化策略
| 参数 | 推荐值 | 原因 |
|------|--------|------|
| dtype | BF16 | 动态范围大，训练稳定，显存减半 |
| GQA group_size | 8 | 平衡质量与显存，节省 87.5% cache |
| cache 量化 | INT8 | 长上下文场景，需校准精度损失 |
| max_seq_len | 业务需求 × 1.2 | 预留 buffer，避免边界溢出 |

#### vLLM 配置示例
```python
from vllm import LLM, SamplingParams

# 启用 PagedAttention（KV cache 动态分配）
llm = LLM(
    model="Qwen/Qwen2-7B",
    tensor_parallel_size=2,      # TP 并行
    gpu_memory_utilization=0.9,  # GPU 显存利用率
    max_model_len=8192,          # 最大序列长度
    enforce_eager=False,         # 使用 CUDA graph
)

sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.8,
)

outputs = llm.generate(prompts, sampling_params)
```

### 代码示例：显存监控
```python
import torch

def monitor_kv_cache_memory(model, batch_size, seq_len):
    """估算并监控 KV cache 显存占用"""
    # 模型配置
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    dtype_size = 2  # FP16
    
    # 计算显存
    kv_memory_bytes = (
        2  # K + V
        * batch_size
        * seq_len
        * num_layers
        * num_heads
        * head_dim
        * dtype_size
    )
    
    kv_memory_gb = kv_memory_bytes / (1024 ** 3)
    
    print(f"KV Cache 显存估算:")
    print(f"  Batch: {batch_size}")
    print(f"  Seq Len: {seq_len}")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Head Dim: {head_dim}")
    print(f"  KV Cache: {kv_memory_gb:.2f} GB")
    
    # 检查 GPU 显存
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    available = total_memory - allocated
    
    print(f"  GPU 总显存: {total_memory:.2f} GB")
    print(f"  已占用: {allocated:.2f} GB")
    print(f"  可用: {available:.2f} GB")
    
    if kv_memory_gb > available:
        print(f"  ⚠️  警告：KV cache 超出可用显存！")
    
    return kv_memory_gb
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **标准 KV Cache (FP16)** | 推理速度提升 2-5×，无精度损失 | 显存占用大（长文本易 OOM） | 短文本（<4K）、小 batch |
| **GQA/MQA** | KV cache 减少 80-97% | 质量轻微下降（0.5-2%） | 大 batch、长文本、推理优先 |
| **KV Cache 量化 (INT8)** | 显存再降 50% | 精度损失（长文本更明显） | 超长文本（>32K）、资源受限 |
| **PagedAttention** | 动态分配，无碎片，支持超长文本 | 实现复杂，需要框架支持 | 生产环境、多用户并发 |
| **Prefix Cache** | 相同前缀场景节省 90%+ 计算 | 命中率依赖场景 | System prompt、多轮对话 |
| **不使用 KV Cache** | 显存占用最小 | 推理慢 10-50×，不可接受 | 仅 Debug/测试 |

### 不同序列长度的显存对比（LLaMA-7B, FP16, batch=1）
| 序列长度 | KV Cache 大小 | 占比（相对 80GB A100） |
|----------|----------------|----------------------|
| 512 | 0.5 GB | 0.6% |
| 2K | 2.2 GB | 2.8% |
| 4K | 8.6 GB | 10.8% |
| 8K | 17.2 GB | 21.5% |
| 16K | 34.4 GB | 43.0% |
| 32K | 68.8 GB | **86.0%** ⚠️ |

## 高频追问（至少 5 个）

### 1. Q: KV cache 影响的是算力还是带宽？
**A**: 主要影响**内存带宽**。Decode 阶段是 memory-bound（访存密集），每生成一个 token 需要读取整个 KV cache（大小 ∝ seq_len），而计算量固定。随着序列变长，KV cache 访存成为瓶颈，这也是为什么 decode 比 prefill 更慢。

### 2. Q: KV cache 会影响模型效果吗？
**A**: **理论上不影响**。KV cache 是计算优化，数学上等价于不使用 cache 的增量计算。但实际可能因以下原因影响效果：
- **数值精度**：FP16 累积误差（通常可忽略）
- **量化损失**：INT8/INT4 量化可能导致质量下降（长文本更敏感）
- **实现 bug**：position id 错误、cache 拼接错误、序列重排未同步

### 3. Q: KV cache 和 tokenizer 有关系吗？
**A**: **间接相关**。Tokenizer 决定了：
1. **序列长度**：不同 tokenizer 的压缩率不同（如中文：LLaMA tokenizer 比中文 tokenizer 长 2-3×）→ 直接影响 KV cache 大小
2. **Position id**：Tokenization 后的 token 数决定 position embedding
3. **Cache 管理**：多轮对话中，tokenizer 的特殊 token（如 `<|im_end|>`）影响 cache 边界

**示例**：
```python
# 中文文本 tokenizer 对比
text = "你好，世界"

# LLaMA tokenizer (英文为主)
tokens_llama = [29871, 30919, 31076, 30210, 29945]  # 5 tokens
kv_cache_size = 5 × head_dim × ...

# Qwen tokenizer (中文优化)
tokens_qwen = [9240, 9239, 9255]  # 3 tokens
kv_cache_size = 3 × head_dim × ...  # 节省 40% cache
```

### 4. Q: 多卡推理（TP）下 KV cache 怎么切分？
**A**: 按 **attention head 切分**。
```python
# TP=2, num_heads=32
# GPU 0: cache for heads 0-15
# GPU 1: cache for heads 16-31

# 每卡显存
kv_cache_per_gpu = 2 × batch × seq × layers × (heads / tp_size) × head_dim

# 优点：
# 1. 无需跨卡同步（每个 head 独立计算）
# 2. 计算和存储都切分，负载均衡

# 注意：
# - 需要 all-reduce 聚合各 head 输出
# - Decode 阶段通信开销可能成为瓶颈（小 batch 时）
```

### 5. Q: 为什么 GQA/MQA 能省显存但不省算力？
**A**: 
- **省显存**：KV cache 只存储 num_groups 组（MQA=1, GQA=8），减少存储
- **不省算力**：
  - 每个 head 仍需要完整的 Q·K^T 计算
  - K/V 虽然共享，但仍需广播到每个 head 参与计算
  - 计算量 = batch × heads × seq × head_dim（不变）

```python
# MHA
for each head h:
    K_h, V_h = cache[h]  # 独立 cache
    attention(Q_h, K_h, V_h)

# GQA (group_size=8)
for each head h:
    g = h // group_size
    K_g, V_g = cache[g]  # 共享 cache
    attention(Q_h, K_g, V_g)  # 仍需完整计算
```

### 6. Q: 长上下文（128K+）场景，KV cache 怎么优化？
**A**: 
1. **GQA/MQA**：从架构层面减少 80%+ cache
2. **量化**：INT8/INT4 量化 cache，减半显存
3. **PagedAttention**：动态分配，避免预分配浪费
4. **Sliding Window**：只缓存最近 N 个 token（牺牲长程依赖）
5. **KV Cache Offloading**：长序列缓存卸载到 CPU（牺牲速度）
6. **Streaming LLM**：滚动窗口 + 初始 token anchors

```python
# 128K 序列，LLaMA-7B FP16
# 标准方案: 344 GB (不可能)
# GQA (group=8): 43 GB (单卡不行，需要多卡)
# INT8: 21.5 GB (可行)
# INT8 + GQA: 5.4 GB (舒适)
```

### 7. Q: KV cache 的 position id 怎么管理？
**A**: 
```python
# Prefill 阶段
position_ids = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]

# Decode 阶段（单步）
position_id = current_seq_len  # 已生成序列长度

# 多轮对话
# Round 1: [0, 1, 2, 3]
# Round 2: 继续从 4 开始 → [4, 5, 6, ...]
# 注意：需要和 RoPE 位置编码配合

# 窗口滑动（滑动窗口注意力）
# 保留最近 window_size 个 token
# position_ids 需要重新映射
```

## 常见错误（至少 3 个）

### 1. **错误：KV cache 没有按层初始化，混用不同层的 cache**
**现象**：模型输出乱码或 loss 爆炸

**原因**：每层的 K/V 是独立的，不能跨层共享

**正确做法**：
```python
# ❌ 错误：所有层共享一个 cache
kv_cache = torch.zeros(batch, num_heads, seq_len, head_dim)

# ✅ 正确：每层独立 cache
kv_cache = [
    {
        'key': torch.zeros(batch, num_heads, seq_len, head_dim),
        'value': torch.zeros(batch, num_heads, seq_len, head_dim)
    }
    for _ in range(num_layers)
]
```

### 2. **错误：Decode 阶段没有增量更新 position id**
**现象**：长序列生成质量下降，重复或偏离主题

**原因**：位置编码错误，模型无法正确理解当前位置

**正确做法**：
```python
# ❌ 错误：position id 固定为 0
position_ids = torch.zeros(batch_size, 1)

# ✅ 正确：递增 position id
position_ids = torch.tensor([[current_seq_len]]).expand(batch_size, -1)
```

### 3. **错误：多轮对话没有正确拼接 cache**
**现象**：第二轮对话时 cache 错位，模型回复混乱

**原因**：新输入的 token 没有正确追加到 cache

**正确做法**：
```python
# ❌ 错误：直接覆盖
kv_cache['key'] = new_K

# ✅ 正确：追加到末尾
kv_cache['key'] = torch.cat([kv_cache['key'], new_K], dim=2)
kv_cache['value'] = torch.cat([kv_cache['value'], new_V], dim=2)

# 同时更新 seq_len
current_seq_len += new_tokens_count
```

### 4. **错误：Batch 中序列长度不同，cache 拼接维度错误**
**现象**：batch > 1 时维度不匹配

**原因**：不同样本的 seq_len 不同，需要 padding 或分离处理

**正确做法**：
```python
# ❌ 错误：假设所有样本长度相同
kv_cache = torch.cat([old_cache, new_token], dim=2)

# ✅ 正确：逐样本处理或使用 padding
# 方案1：逐样本处理（慢但正确）
for i in range(batch_size):
    kv_cache[i] = torch.cat([kv_cache[i], new_token[i:i+1]], dim=2)

# 方案2：使用 attention mask（推荐）
# PagedAttention 会自动处理变长序列
```

### 5. **错误：使用 FP32 存储 KV cache**
**现象**：显存占用翻倍，推理速度下降

**原因**：训练习惯带入推理，过度追求精度

**正确做法**：
```python
# ❌ 错误
kv_cache = torch.zeros(..., dtype=torch.float32)  # 4 bytes per element

# ✅ 正确
kv_cache = torch.zeros(..., dtype=torch.float16)  # 2 bytes per element
# 或
kv_cache = torch.zeros(..., dtype=torch.bfloat16)  # 更稳定
```

## 反问面试官的问题

### 1. 技术深度类
- "你们线上最长支持多长的上下文？KV cache 的显存瓶颈一般怎么解决？"
- "对于多轮对话场景，prefix cache 的命中率能做到多少？有什么优化策略？"
- "你们有尝试过 KV cache 量化吗？对模型质量的影响如何评估的？"

### 2. 业务场景类
- "业务中 batch size 一般多大？KV cache 显存占用的大头是在 prefill 还是 decode？"
- "对于长文本任务（如 RAG、文档摘要），你们怎么平衡延迟和显存成本？"
- "多卡推理时，通信开销和 KV cache 显存哪个是主要瓶颈？"

## 自测题

### 口述（能流畅讲清楚的知识点）
1. KV cache 存储的内容、形状、为什么能避免重复计算
2. 为什么没有 Q cache，从自回归生成的角度解释
3. 显存估算公式及各参数含义（batch、seq、layer、head、dim）
4. GQA/MQA 如何节省 KV cache 显存，节省比例如何计算
5. KV cache 与内存带宽的关系，为什么 decode 更受带宽限制

### 手写（5 分钟能写出的代码/公式）
1. **KV cache 显存估算函数**：给定模型配置和 batch、seq，计算显存占用
2. **Decode 单步更新代码**：实现 append 新 K/V 到 cache
3. **GQA cache 形状推导**：给定 num_heads 和 group_size，计算每组的 cache 形状

```python
# 练习：实现 KV cache 显存估算
def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "fp16"
) -> float:
    """
    计算 KV cache 显存占用（GB）
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        head_dim: 每个头的维度
        dtype: 数据类型 ("fp16", "bf16", "int8", "int4")
    
    Returns:
        显存占用（GB）
    """
    # TODO: 实现计算逻辑
    pass
```

## 标签
#推理 #kv_cache #工程 #prefill #decode #GQA #MQA #显存优化 #batching

## 相关文档
- [[01-Prefill与Decode]] - KV cache 在 prefill/decode 两阶段的不同作用
- [[03-Paged Attention]] - KV cache 的动态分配与碎片优化
- [[../06-模型架构对比/03-GQA与MQA]] - GQA/MQA 如何减少 KV cache 显存
- [[../07-分布式训练/04-TP并行]] - 多卡推理下 KV cache 的切分策略
- [[../08-数值精度/02-推理量化]] - KV cache 量化的精度与性能权衡
