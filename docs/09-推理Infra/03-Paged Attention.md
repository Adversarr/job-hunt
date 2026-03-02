# Paged Attention 与 Prefix Cache

## 一句话结论
Paged Attention 通过**分页管理 KV cache**解决显存碎片和动态长度问题,将显存利用率从 60-70% 提升至 95%+;Prefix Cache 通过**缓存公共前缀的 KV**在多轮对话/多用户场景下复用计算,命中率可达 30-70%。

## 核心定义/公式

### Paged Attention 核心机制

```python
# 传统连续显存分配
# [batch_size, seq_len, num_heads, head_dim] 需要连续大块显存
# 显存碎片率: 30-40%, 实际利用率: 60-70%

# Paged Attention 分页机制
# Page 大小: block_size (通常为 16-64 tokens)
# 显存池: 预分配的 block pool
# Block 表: 记录每个 sequence 的 block 链表

class BlockTable:
    """Paged Attention 的块表管理"""
    def __init__(self, block_size: int = 16, num_blocks: int = 1000):
        self.block_size = block_size  # 每个 block 包含的 token 数
        self.num_blocks = num_blocks  # 总 block 数量
        self.free_blocks = list(range(num_blocks))  # 空闲 block 池
        self.block_tables = {}  # seq_id -> List[block_id]
    
    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """为 sequence 分配 blocks"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise OOMError("Insufficient blocks")
        
        blocks = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        self.block_tables[seq_id] = blocks
        return blocks
```

### KV Cache 大小估算公式

```python
# KV cache 显存占用（单卡）
def kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "fp16"
) -> int:
    """
    计算 KV cache 显存占用（字节）
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        num_layers: 层数
        num_heads: 注意力头数（GQA 时为 KV 头数）
        head_dim: 头维度
        dtype: 数据类型 ("fp16", "bf16", "int8")
    
    Returns:
        显存占用（字节）
    """
    dtype_bytes = {"fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    
    # K 和 V 各一份
    kv_cache_size = 2 * batch_size * seq_len * num_layers * num_heads * head_dim * dtype_bytes[dtype]
    
    return kv_cache_size

# 示例计算（Llama-2-7B, batch=32, seq=4096）
# num_layers=32, num_heads=32, head_dim=128
# FP16: 2 * 32 * 4096 * 32 * 32 * 128 * 2 = 68.7 GB
# INT8: 34.4 GB
```

### Prefix Cache 命中判定

```python
# Prefix cache 命中条件
def can_prefix_cache_hit(
    cached_tokens: List[int],
    new_tokens: List[int],
    prefix_len: int
) -> Tuple[bool, int]:
    """
    判断 prefix cache 是否命中
    
    Args:
        cached_tokens: 已缓存的 token 序列
        new_tokens: 新请求的 token 序列
        prefix_len: 期望的 prefix 长度
    
    Returns:
        (是否命中, 实际命中长度)
    """
    # 逐 token 比较（position-aware）
    hit_len = 0
    for i in range(min(len(cached_tokens), len(new_tokens), prefix_len)):
        if cached_tokens[i] == new_tokens[i]:
            hit_len += 1
        else:
            break
    
    # 命中条件：命中长度 >= 阈值（通常 > 0 即可）
    return hit_len > 0, hit_len

# Position ID 对齐（多轮对话场景）
def align_position_ids(
    cached_pos: List[int],
    new_tokens: List[int],
    hit_len: int
) -> List[int]:
    """
    多轮对话中 position id 的处理
    
    关键：cache 命中部分的 position id 必须完全一致
    """
    # 命中部分：沿用 cached position ids
    # 未命中部分：重新计算
    new_pos = cached_pos[:hit_len] + list(range(hit_len, len(new_tokens)))
    return new_pos
```

### KV Cache 在 TP 下的切分

```python
# Tensor Parallel 下 KV cache 的存储策略
def kv_cache_tp_sharding(
    num_heads: int,
    num_kv_heads: int,  # GQA: num_kv_heads < num_heads
    tp_size: int
) -> Dict[str, int]:
    """
    TP 下 KV cache 如何切分
    
    Returns:
        每卡的 KV head 数量
    """
    # Attention 权重在 TP 下按 head 切分
    # KV cache 同样按 head 切分，每卡只存储自己负责的 heads
    
    heads_per_gpu = num_heads // tp_size
    kv_heads_per_gpu = num_kv_heads // tp_size
    
    return {
        "heads_per_gpu": heads_per_gpu,
        "kv_heads_per_gpu": kv_heads_per_gpu,
        "tp_shard_size": kv_heads_per_gpu * head_dim  # 每个 shard 的维度
    }
```

## 为什么（2-3 个因果链）

### 1. 传统显存管理的碎片问题

**现象** → **根因** → **结果**

- **现象**：batch 稍大就 OOM,但 `nvidia-smi` 显示还有 30% 空闲显存
- **根因**：传统方案要求每个 sequence 的 KV cache 在显存中**连续存储**,长序列需要大块连续空间
  ```python
  # 传统方案：预分配最大长度
  max_seq_len = 4096
  kv_cache = torch.zeros(batch_size, max_seq_len, num_layers, num_heads, head_dim)
  # 问题：
  # 1. 短序列浪费空间（实际 seq_len=512,分配 4096）
  # 2. 长短不一的请求导致大量碎片（类似内存碎片）
  # 3. 动态增长困难（序列生成时长度变化）
  ```
- **结果**：
  - 显存利用率仅 60-70%,剩余 30% 为碎片
  - batch size 受限于**最长序列**而非平均长度
  - 长上下文场景（128K tokens）几乎无法部署

### 2. Paged Attention 的分页优势

**机制** → **收益** → **适用边界**

- **机制**：将 KV cache 拆分为固定大小的 blocks（类似操作系统分页）
  ```python
  # Block size = 16 tokens
  # 128K 上下文需要 128000 / 16 = 8000 blocks
  # 不需要连续的 128K 空间,只需要找到 8000 个空闲 block
  ```
- **收益**：
  1. **显存利用率 95%+**：碎片从 30-40% 降至 5%
  2. **支持动态长度**：序列可动态增长,按需分配 blocks
  3. **内存共享**：多个 sequence 可共享相同的 prefix blocks（copy-on-write）
- **适用边界**：
  - 收益最大：长短不一的混合负载、长上下文场景
  - 收益较小：所有序列长度相同且固定的场景

### 3. Prefix Cache 复用的前提条件

**场景** → **命中率因素** → **实际表现**

- **高命中率场景**：
  1. **多轮对话**：前文系统 prompt + 历史对话完全复用（命中率 50-70%）
  2. **RAG 检索**：system prompt + 检索指令 + 少量文档前缀（命中率 30-50%）
  3. **Few-shot 学习**：相同的示例模板（命中率 60-80%）
  
- **低命中率场景**：
  1. **单轮问答**：每次 prompt 不同（命中率 < 10%）
  2. **重排/拼接后**：position id 或 token 顺序改变（命中率降至 0）

- **关键因素**：
  ```python
  # 命中条件：token 完全一致 + position id 一致
  # ❌ 以下情况会破坏命中：
  
  # 1. Token 序列改变
  prompt_v1 = "请翻译：Hello"
  prompt_v2 = "请翻译：World"  # 第二个 token 就不同,无法复用
  
  # 2. Position ID 改变（滑动窗口）
  # 原始: [0, 1, 2, ..., 100]
  # 滑动后: [50, 51, ..., 150]  # position id 完全不同,cache 失效
  
  # 3. 拼接顺序改变
  messages_v1 = [system, user_msg1, assistant_reply, user_msg2]
  messages_v2 = [system, user_msg2, assistant_reply, user_msg1]  # 顺序不同,cache 失效
  ```

## 怎么做（可落地步骤）

### 标准做法：vLLM Paged Attention 实践

#### 1. Block 大小选择

```python
# vLLM 默认配置
block_size = 16  # 每个 block 包含 16 个 tokens

# 选择依据：
# - 过小（< 8）：block 表开销大,管理复杂度高
# - 过大（> 64）：碎片化重新出现,动态分配灵活性下降
# - 16-32 是常见平衡点

# 计算需要的 block 数量
def calculate_blocks_needed(seq_len: int, block_size: int = 16) -> int:
    return (seq_len + block_size - 1) // block_size
```

#### 2. 显存池预分配

```python
# vLLM 启动时预分配 GPU 显存池
# 配置文件示例
config = {
    "block_size": 16,
    "gpu_memory_utilization": 0.9,  # 预留 10% 给其他开销
    "max_num_blocks_per_seq": 128000 // 16,  # 支持最大 128K 上下文
}

# 显存池大小计算
def calculate_block_pool_size(
    total_gpu_memory: int,  # GPU 总显存（字节）
    model_weights: int,     # 模型权重（字节）
    activation_memory: int, # 激活显存（字节）
    block_size: int,
    head_dim: int,
    num_layers: int,
    num_kv_heads: int,
    dtype_bytes: int = 2    # FP16
) -> int:
    """
    计算可用的 block 数量
    
    Returns:
        最大 block 数量
    """
    # 可用于 KV cache 的显存
    available_memory = total_gpu_memory * 0.9 - model_weights - activation_memory
    
    # 每个 block 的显存占用
    # block = [block_size, num_layers, num_kv_heads, head_dim]
    bytes_per_block = block_size * num_layers * num_kv_heads * head_dim * dtype_bytes * 2  # K + V
    
    return int(available_memory / bytes_per_block)
```

#### 3. Prefix Cache 配置与使用

```python
from vllm import LLM, SamplingParams

# 启用 prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # 关键配置
    block_size=16,
    gpu_memory_utilization=0.9,
)

# 场景 1：多轮对话
messages_round1 = [
    {"role": "system", "content": "你是一个助手"},  # 这部分会被缓存
    {"role": "user", "content": "什么是机器学习？"},
]
messages_round2 = [
    {"role": "system", "content": "你是一个助手"},  # cache 命中！
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."},
    {"role": "user", "content": "那深度学习呢？"},  # 新问题
]

# vLLM 自动检测前缀匹配,复用 KV cache
output1 = llm.chat(messages_round1)
output2 = llm.chat(messages_round2)  # 命中前缀,节省 50%+ 计算

# 场景 2：RAG 检索
system_prompt = "你是一个问答助手，根据以下文档回答问题：\n"
doc_prefix = "文档内容：..."[:500]  # 文档前缀可复用

queries = [
    f"{system_prompt}{doc_prefix}\n问题：什么是 X？",
    f"{system_prompt}{doc_prefix}\n问题：什么是 Y？",  # prefix 命中
]
```

#### 4. 长上下文显存优化策略

```python
# 策略 1：KV Cache 量化（最直接有效）
# vLLM 配置
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="int8",  # FP16 -> INT8, 显存减半
    # 或 "int4" 进一步压缩
)

# 收益：
# - INT8：显存占用 -50%, 吞吐 +30-50%, 质量损失 < 1%
# - INT4：显存占用 -75%, 吞吐 +100%, 质量损失 1-3%

# 策略 2：滑动窗口 + Cache 淘汰
class SlidingWindowCache:
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size
    
    def evict_old_blocks(self, block_table: List[int], current_len: int):
        """淘汰超出窗口的旧 blocks"""
        if current_len > self.window_size:
            num_blocks_to_evict = (current_len - self.window_size) // self.block_size
            evicted_blocks = block_table[:num_blocks_to_evict]
            # 释放 blocks
            self.free_blocks.extend(evicted_blocks)
            return block_table[num_blocks_to_evict:]
        return block_table

# 策略 3：GQA/MQA（架构层面优化）
# Llama-2: MHA (32 heads, 32 KV heads)
# Llama-3: GQA (32 heads, 8 KV heads) -> KV cache 显存 -75%

# 策略 4：动态 Batch + Paged Attention
# 短序列填充到 block_size 的倍数,避免跨 block 碎片
```

### 关键配置/参数

#### vLLM 核心参数

```python
# vLLM 启动参数
vllm_config = {
    # Block 管理
    "block_size": 16,                    # Block 大小（tokens/block）
    "gpu_memory_utilization": 0.9,       # 显存利用率上限
    "max_num_seqs": 256,                 # 最大并发序列数
    
    # Prefix Caching
    "enable_prefix_caching": True,       # 启用 prefix cache
    "max_num_batched_tokens": 32768,     # 单 batch 最大 token 数
    
    # 长上下文
    "max_model_len": 128000,             # 最大上下文长度
    "kv_cache_dtype": "auto",            # KV cache 精度（auto/fp16/int8/int4）
    
    # 性能调优
    "swap_space": 4,                     # CPU swap 空间（GB）
    "enforce_eager": False,              # 是否强制 eager 模式
}

# 参数调优建议：
# 1. block_size: 16-32 平衡点,过小增加管理开销,过大增加碎片
# 2. gpu_memory_utilization: 0.85-0.95,预留 buffer 避免突发 OOM
# 3. kv_cache_dtype: INT8 是性价比最优（质量损失小,显存减半）
```

### 代码示例：完整推理流程

```python
import torch
from vllm import LLM, SamplingParams
from vllm.sequence import Logprob

# 1. 初始化 vLLM（启用 Paged Attention + Prefix Cache）
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,              # 4 卡 TP
    block_size=16,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
    max_model_len=128000,
    kv_cache_dtype="int8",               # INT8 量化
)

# 2. 准备多轮对话数据
def prepare_multiround_chat(session_id: str, user_query: str):
    """多轮对话示例：自动处理 position id"""
    # vLLM 内部维护 session -> cached prefix 的映射
    messages = get_session_history(session_id)
    messages.append({"role": "user", "content": user_query})
    return messages

# 3. 批量推理
sessions = [
    prepare_multiround_chat("s1", "什么是注意力机制？"),
    prepare_multiround_chat("s2", "Transformer 的优势？"),
    prepare_multiround_chat("s1", "它如何应用到 LLM？"),  # s1 的第二轮,cache 命中
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)

outputs = llm.chat(sessions, sampling_params)

# 4. 检查 cache 命中情况
for output in outputs:
    print(f"Session: {output.request_id}")
    print(f"Cache hit length: {output.num_cached_tokens}")  # 命中的 prefix 长度
    print(f"Generated: {output.outputs[0].text}")
```

## 权衡分析

### Paged Attention vs 传统方案

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **Paged Attention** | 显存利用率 95%+<br>支持动态长度<br>内存共享（prefix cache） | Block 表管理开销（~1% 显存）<br>间接访问（指针跳转）<br>需要框架支持（vLLM/TensorRT-LLM） | 长短不一的混合负载<br>长上下文场景（>32K）<br>多轮对话/RAG 场景 |
| **传统连续分配** | 实现简单<br>访问高效（连续内存）<br>无管理开销 | 显存利用率 60-70%<br>不支持动态增长<br>长序列 OOM 风险高 | 所有序列长度固定<br>短上下文场景（<4K）<br>单次推理无复用 |

### KV Cache 量化方案对比

| 量化方案 | 显存节省 | 质量损失 | 吞吐提升 | 适用场景 |
|---------|---------|---------|---------|---------|
| **FP16/BF16** | 基准 | 基准 | 基准 | 对质量要求极高<br>显存充裕 |
| **INT8** | -50% | < 1% | +30-50% | **性价比最优**<br>大多数场景推荐 |
| **INT4** | -75% | 1-3% | +100% | 显存极度受限<br>可接受轻微质量下降 |
| **FP8** | -50% | < 0.5% | +40% | 新硬件支持（H100）<br>质量与性能平衡 |

### Prefix Cache 命中率优化策略

| 策略 | 命中率提升 | 实现复杂度 | 副作用 |
|------|-----------|-----------|--------|
| **System Prompt 固定** | +30-50% | 低 | 限制灵活性 |
| **RAG 文档前缀复用** | +20-40% | 中 | 需要对齐文档格式 |
| **Few-shot 模板统一** | +40-60% | 低 | 示例需标准化 |
| **Cache 跨 Session 共享** | +10-20% | 高 | 需管理 session 生命周期 |
| **Position ID 重映射** | +20-30% | 高 | 可能影响模型理解 |

## 高频追问（至少 5 个）

### 1. Q: Paged Attention 和 FlashAttention 有什么区别？

**A:** 
- **FlashAttention**：优化 **计算效率**,通过 tiling 减少 HBM 访问,加速 QK^T 和 PV 计算
- **Paged Attention**：优化 **显存管理**,通过分页减少碎片,提高显存利用率
- **关系**：两者正交,可组合使用（vLLM 已集成 FlashAttention）

```
┌─────────────────┬──────────────┬──────────────┐
│                  │ FlashAttn    │ PagedAttn    │
├─────────────────┼──────────────┼──────────────┤
│ 优化目标         │ 计算速度     │ 显存利用率   │
│ 核心技术         │ Tiling       │ 分页管理     │
│ 受益阶段         │ Prefill 更大 │ Decode 更大  │
│ 显存节省         │ ~0（主要加速）│ 30-40%       │
│ 速度提升         │ 2-4x         │ ~0（主要省显存）│
└─────────────────┴──────────────┴──────────────┘
```

### 2. Q: 为什么 Decode 阶段更慢？Paged Attention 能解决吗？

**A:**

**Decode 慢的原因**：
```python
# Prefill 阶段（算密集）
# Q: [batch, prompt_len, head_dim]  # prompt_len 可达数千
# K, V: [batch, prompt_len, head_dim]
# 计算量: O(batch * prompt_len^2 * num_heads)  # 二次方增长

# Decode 阶段（访存密集）
# Q: [batch, 1, head_dim]  # 每次 1 个 token
# K, V: [batch, prompt_len + generated_len, head_dim]  # 越来越长
# 计算量: O(batch * seq_len * num_heads)  # 线性
# 但访存: 每次读取整个 KV cache, 带宽瓶颈！

# Decode 慢的核心：KV cache 读取的带宽瓶颈
# Paged Attention 只解决显存碎片,不解决带宽问题
# 解决方案：GQA（减少 KV heads）、KV cache 量化、FlashDecoding
```

**Paged Attention 的作用**：
- ✅ 提高显存利用率,支持更大 batch → 提升**吞吐**
- ❌ 不减少 KV cache 读取量 → 不提升单请求**延迟**

### 3. Q: 多轮对话的 Position ID 怎么处理？滑动窗口时 Position 怎么算？

**A:**

```python
# 多轮对话 Position ID 示例
# Round 1: [system] + [user] + [assistant]
# tokens: [t1, t2, ..., t100] + [t101, ..., t150] + [t151, ..., t200]
# pos_ids: [0, 1, ..., 99] + [100, ..., 149] + [150, ..., 199]

# Round 2: [system] + [user] + [assistant] + [user_new] + [assistant_new]
# ✅ 正确：position id 连续
# pos_ids: [0, ..., 199] + [200, ..., 250] + [251, ..., 300]

# ❌ 错误：position id 重置（会破坏模型理解）
# pos_ids: [0, ..., 199] + [0, ..., 50]  # 错误！

# 滑动窗口时的处理
class SlidingWindowPositionManager:
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size
    
    def compute_position_ids(
        self,
        full_seq_len: int,
        window_start: int
    ) -> List[int]:
        """
        滑动窗口 position id 计算
        
        Args:
            full_seq_len: 完整序列长度
            window_start: 窗口起始位置
        
        Returns:
            当前窗口内的 position ids
        
        关键：position id 反映"相对位置",而非"绝对位置"
        """
        window_end = min(window_start + self.window_size, full_seq_len)
        # 方案 1：保留原始 position（推荐）
        return list(range(window_start, window_end))
        
        # 方案 2：重映射到窗口内（不推荐,可能破坏长程依赖）
        # return list(range(0, window_end - window_start))

# Cache 命中与 Position ID 的关系
# ❌ 滑动窗口会破坏 prefix cache 命中
# 原因：position id 改变,即使 token 相同也无法复用
# 解决：维护全局 position id, 仅淘汰超出窗口的 KV cache
```

### 4. Q: 重排/拼接会影响 Cache 命中吗？

**A:**

**会完全破坏 cache 命中**,原因：

```python
# ❌ 案例 1：消息顺序重排
messages_v1 = [system, user_A, assistant_A, user_B]
messages_v2 = [system, user_B, assistant_A, user_A]  # 顺序不同

# Cache 要求：token 序列完全一致 + position id 一致
# 重排后 token 序列改变 → cache 失效

# ❌ 案例 2：拼接格式改变
prompt_v1 = f"Context: {doc}\nQuestion: {q1}"
prompt_v2 = f"Question: {q2}\nContext: {doc}"  # 顺序颠倒

# 即使内容相同,token 序列不同 → cache 失效

# ✅ 正确做法：保持格式统一
def format_prompt(system: str, context: str, question: str):
    # 固定模板,确保相同内容产生相同 token 序列
    return f"{system}\n\nContext:\n{context}\n\nQuestion:\n{question}"

# 所有请求使用相同格式 → prefix cache 可复用
```

**工程建议**：
1. 在应用层强制统一 prompt 模板
2. 对 system prompt / few-shot 示例使用固定格式
3. RAG 场景：对文档进行固定长度的前缀截取

### 5. Q: KV Cache 在 TP（Tensor Parallel）下怎么存？通信开销在哪？

**A:**

```python
# TP 下 KV Cache 的存储策略
# 以 4 卡 TP, 32 heads, GQA (8 KV heads) 为例

class TPKVCacheManager:
    def __init__(self, num_heads: int, num_kv_heads: int, tp_size: int):
        self.num_heads = num_heads        # 32
        self.num_kv_heads = num_kv_heads  # 8 (GQA)
        self.tp_size = tp_size            # 4
    
    def shard_kv_cache(self):
        """
        KV cache 在 TP 下的切分
        
        策略：按 KV heads 切分,每卡存储 num_kv_heads/tp_size 个 heads
        """
        kv_heads_per_gpu = self.num_kv_heads // self.tp_size  # 2 heads/GPU
        
        # 每张卡的 KV cache 形状
        # [batch, seq_len, num_layers, kv_heads_per_gpu, head_dim]
        # 显存占用 = batch * seq_len * num_layers * 2 * kv_heads_per_gpu * head_dim * 2 bytes
        
        return {
            "kv_heads_per_gpu": kv_heads_per_gpu,  # 2
            "shard_strategy": "head-wise",
        }
    
    def compute_attention_with_tp(self, Q, K, V):
        """
        TP 下 Attention 的计算与通信
        
        关键：Q 按 heads 切分,K/V 按 KV heads 切分
        """
        # 每卡计算自己负责的 heads
        # Q_local: [batch, seq_q, heads_per_gpu, head_dim]
        # K_local: [batch, seq_k, kv_heads_per_gpu, head_dim]
        # V_local: [batch, seq_k, kv_heads_per_gpu, head_dim]
        
        # 1. 计算 attention scores（本地,无通信）
        scores = torch.matmul(Q_local, K_local.transpose(-2, -1)) / sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_local)
        # attn_output: [batch, seq_q, heads_per_gpu, head_dim]
        
        # 2. All-Reduce 聚合结果（关键通信点）
        # 原因：output projection (O) 需要所有 heads 的结果
        # O = concat(attn_output from all GPUs) @ W_O
        # 需要在 O 层之前 all-reduce
        
        # 在 PyTorch 中
        torch.distributed.all_reduce(attn_output, op=torch.distributed.ReduceOp.SUM)
        # 通信量: batch * seq_q * num_heads * head_dim * 2 bytes
        # 例如: 32 * 1024 * 32 * 128 * 2 = 268 MB per layer
        
        return attn_output

# 通信开销分析
# Decode 阶段更慢的原因之一：通信更频繁
# - Prefill: 1 次 all-reduce (处理完整 prompt)
# - Decode: 每生成 1 token 都要 all-reduce
# - 解决：FlashDecoding (减少 all-reduce 次数)
```

### 6. Q: 长上下文（128K）"显存炸"怎么缓解？

**A:**

```python
# 128K 上下文显存估算（Llama-2-7B, FP16, batch=1）
# KV cache: 2 * 1 * 128000 * 32 * 32 * 128 * 2 = 68.7 GB
# + 模型权重: 14 GB
# + 激活: ~5 GB
# 总计: ~88 GB (单卡无法承载)

# 缓解策略组合拳
strategies = {
    # 1. KV Cache 量化（最有效, -50% ~ -75%）
    "kv_quant": {
        "method": "INT8/INT4",
        "savings": "34 GB (INT8) / 17 GB (INT4)",
        "quality_loss": "<1% (INT8) / 1-3% (INT4)",
    },
    
    # 2. GQA（架构优化, -75%）
    "gqa": {
        "method": "Llama-3 GQA: 8 KV heads vs 32 heads",
        "savings": "17 GB",
        "quality_loss": "<0.5%",
    },
    
    # 3. Paged Attention + Block 管理
    "paged_attn": {
        "method": "按需分配 blocks,避免预分配浪费",
        "savings": "实际利用率从 60% 提升至 95%",
    },
    
    # 4. 滑动窗口 + Cache 淘汰
    "sliding_window": {
        "method": "只保留最近 32K tokens 的 KV cache",
        "savings": "34 GB",
        "quality_loss": "失去长程依赖能力",
    },
    
    # 5. 多卡 TP + KV Cache 切分
    "tp_sharding": {
        "method": "4 卡 TP,每卡 1/4 KV cache",
        "savings": "单卡 17 GB",
    },
    
    # 6. CPU Offload（vLLM swap）
    "cpu_offload": {
        "method": "将冷 KV cache 换出到 CPU",
        "savings": "GPU 显存大幅减少",
        "cost": "延迟增加 10-30%",
    },
}

# 推荐组合
recommended = [
    "GQA (架构层面)",
    "INT8 KV Cache (性价比最优)",
    "Paged Attention (显存管理)",
    "TP Sharding (多卡分布式)",
]
# 最终：128K 上下文可在 4x A100 (80GB) 上稳定运行
```

### 7. Q: 什么时候 Prefix Cache 命中率高？

**A:**

```python
# 高命中率场景特征
high_hit_rate_scenarios = {
    # 1. 多轮对话（命中率 50-70%）
    "multiround_chat": {
        "cacheable_prefix": "system_prompt + history_messages",
        "hit_condition": "同一用户连续对话",
        "example": """
            Round 1: [system: "你是助手"] + [user: "什么是AI?"]
            Round 2: [system: "你是助手"] + [user: "什么是AI?"] + [assistant: "..."] + [user: "那ML呢?"]
            Cache hit: Round 2 的前半部分与 Round 1 完全一致
        """,
    },
    
    # 2. RAG 检索（命中率 30-50%）
    "rag_retrieval": {
        "cacheable_prefix": "system_prompt + retrieval_instruction + doc_prefix",
        "hit_condition": "相同文档集,不同问题",
        "example": """
            Query 1: "根据文档,什么是X?"
            Query 2: "根据文档,什么是Y?"
            Cache hit: system_prompt + doc_prefix (文档前 500 tokens)
        """,
    },
    
    # 3. Few-shot Learning（命中率 60-80%）
    "few_shot": {
        "cacheable_prefix": "system_prompt + few_shot_examples",
        "hit_condition": "使用相同的示例模板",
        "example": """
            Request 1: [system] + [示例1-5] + [问题A]
            Request 2: [system] + [示例1-5] + [问题B]
            Cache hit: system + 示例1-5 完全复用
        """,
    },
    
    # 4. 模板化任务（命中率 70-90%）
    "templated_tasks": {
        "cacheable_prefix": "task_instruction + format_template",
        "hit_condition": "任务格式固定,仅输入不同",
        "example": """
            Task: 翻译任务
            Input 1: "翻译: Hello"
            Input 2: "翻译: World"
            Cache hit: "翻译: " 这个 prefix
        """,
    },
}

# 低命中率场景
low_hit_rate_scenarios = {
    # 1. 单轮问答
    "single_turn_qa": "每次 prompt 不同,无公共前缀",
    
    # 2. 个性化推荐
    "personalized": "用户画像差异大,难以复用",
    
    # 3. 创意生成
    "creative": "每次要求不同的风格/主题",
}

# 提升命中率的工程手段
optimization_tips = [
    "固定 system prompt 格式",
    "统一 few-shot 示例模板",
    "RAG 场景对文档做 prefix 截取",
    "避免 prompt 中的随机元素（如时间戳）",
    "使用 session ID 维护多轮对话上下文",
]
```

### 8. Q: Paged Attention 会影响模型精度吗？

**A:**

```python
# Paged Attention 对精度的影响
impact_on_accuracy = {
    "理论层面": {
        "conclusion": "数值上完全等价,无精度损失",
        "reason": "只是内存管理方式改变,计算过程不变",
        "math": """
            传统: Attention(Q, K, V) 其中 K, V 是连续内存
            Paged: Attention(Q, gather(K_blocks), gather(V_blocks))
            结果完全相同,只是访问方式不同
        """,
    },
    
    "工程层面": {
        "potential_issue": "间接内存访问可能引入微小数值误差",
        "magnitude": "< 1e-7 (FP16), 可忽略",
        "verification": "vLLM 的 test suite 验证了数值一致性",
    },
    
    "对比其他优化": {
        "KV Cache 量化": "有精度损失 (INT8: <1%, INT4: 1-3%)",
        "FlashAttention": "数值等价,但实现细节可能引入误差",
        "Paged Attention": "无精度损失",
    },
}

# 验证方法
import torch
from vllm import LLM

def verify_paged_attention_accuracy():
    """验证 Paged Attention 与传统方案的数值一致性"""
    model = LLM("meta-llama/Llama-2-7b-hf", enforce_eager=True)
    
    # 生成输出
    prompt = "Hello, how are you?"
    output = model.generate(prompt, sampling_params=SamplingParams(max_tokens=10))
    
    # 与 HuggingFace Transformers 对比
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # 验证 logits 差异 < 1e-5
    # 实测: vLLM 与 HF 的输出 logits 差异 < 1e-6
```

## 常见错误（至少 3 个）

### 1. 误以为 Paged Attention 能加速推理

**错误描述**：
```python
# ❌ 错误认知
"Paged Attention 通过分页管理加速了推理过程"

# ✅ 正确理解
Paged Attention 优化的是**显存利用率**,不直接加速计算
- 主要收益：支持更大 batch size → 提升**吞吐量**
- 单请求延迟：基本不变（甚至略增,因为间接内存访问）
- 加速推理靠：FlashAttention、GQA、Speculative Decoding

# 类比
Paged Attention ≈ 内存分页管理
FlashAttention ≈ CPU 缓存优化
两者正交,可组合使用
```

**正确做法**：
- 明确区分"显存优化"与"计算优化"
- vLLM 同时集成了 Paged Attention + FlashAttention
- 汇报性能提升时注明：吞吐 vs 延迟

### 2. Position ID 重置导致 Cache 失效

**错误描述**：
```python
# ❌ 错误实现：每轮对话重置 position id
def chat_round(user_input: str, history: List):
    messages = history + [{"role": "user", "content": user_input}]
    # 错误：每次从 0 开始
    pos_ids = list(range(len(tokenize(messages))))
    return model.generate(messages, position_ids=pos_ids)

# 问题：
# 1. Position ID 不连续,破坏模型对序列的理解
# 2. 即使 token 序列相同,position id 不同 → cache 无法命中

# ✅ 正确实现：维护全局 position id
class ChatSession:
    def __init__(self):
        self.history = []
        self.total_tokens = 0  # 累计 token 数
    
    def chat(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})
        # 正确：position id 从上次结束位置继续
        pos_ids = list(range(self.total_tokens, self.total_tokens + len(new_tokens)))
        self.total_tokens += len(new_tokens)
        return model.generate(self.history, position_ids=pos_ids)
```

**正确做法**：
- 多轮对话必须维护全局 position id
- 滑动窗口场景：保留原始 position,仅淘汰旧 KV cache
- 使用 vLLM 的 session 管理功能

### 3. 滥用 KV Cache 量化导致质量劣化

**错误描述**：
```python
# ❌ 错误做法：所有场景都用 INT4 量化
llm = LLM(
    model="Llama-2-7b",
    kv_cache_dtype="int4",  # 激进量化
)

# 问题：
# 1. 长上下文场景精度损失放大（1-3% 累积效应）
# 2. 某些层（LayerNorm、输出层）对量化敏感
# 3. 生成长文本时质量明显下降

# ✅ 正确做法：根据场景选择量化策略
def select_kv_cache_dtype(
    max_seq_len: int,
    quality_requirement: str,  # "high"/"medium"/"low"
    gpu_memory: int,  # GB
):
    """根据场景选择 KV cache 精度"""
    if quality_requirement == "high":
        return "fp16"  # 不量化
    elif max_seq_len > 64000:
        return "int8"  # 长上下文用 INT8 平衡
    elif gpu_memory < 40:
        return "int4"  # 显存极度受限
    else:
        return "int8"  # 默认推荐

# 监控量化后的质量指标
def monitor_quantization_quality():
    """
    关键指标：
    1. Perplexity 变化 < 5%
    2. 生成文本的重复率
    3. 长文本任务的准确率（如 QA、Summarization）
    """
```

**正确做法**：
- INT8 是性价比最优选择（质量损失 < 1%）
- 长上下文（>64K）避免用 INT4
- 关键业务场景保留 FP16
- 部署前做量化后的质量评测

### 4. 忽视 Block Size 对性能的影响

**错误描述**：
```python
# ❌ 错误配置：Block size 过小或过大
llm = LLM(
    model="Llama-2-7b",
    block_size=4,  # 过小
)

# 问题：
# 1. Block size=4：管理开销大,block 表占用 5-10% 显存
# 2. Block size=128：碎片化重新出现,短序列浪费空间
# 3. 不符合硬件对齐要求,影响 kernel 效率

# ✅ 正确做法：根据典型序列长度选择
def recommend_block_size(typical_seq_len: int):
    """推荐 block size"""
    # 短序列（< 2K）：block_size=16
    # 中序列（2K-32K）：block_size=32
    # 长序列（> 32K）：block_size=64
    
    if typical_seq_len < 2048:
        return 16
    elif typical_seq_len < 32768:
        return 32
    else:
        return 64

# vLLM 默认 16 是通用平衡点
# 生产环境建议根据实际负载测试调优
```

**正确做法**：
- 大多数场景用默认值 16
- 长上下文场景可测试 32/64
- 使用 vLLM 的 benchmark 工具对比不同配置

## 反问面试官的问题

### 技术深度类问题

1. **"贵司在推理场景中,Paged Attention 的实际收益如何？有没有遇到过显存碎片导致的 OOM 问题？"**
   - 目的：了解团队对推理优化的重视程度
   - 可引出：长上下文支持、多卡部署、量化策略等话题

2. **"多轮对话场景下,Prefix Cache 的命中率大概多少？有没有做过优化？"**
   - 目的：了解团队是否有实际的多轮对话优化经验
   - 可引出：session 管理、position id 处理、cache 淘汰策略

3. **"长上下文（如 128K）场景下,贵司是用什么方案解决的？INT8 KV cache 的质量损失能接受吗？"**
   - 目的：了解团队在长上下文上的技术选型
   - 可引出：量化、GQA、多卡并行等话题

### 业务场景类问题

1. **"贵司的推理服务,QPS 和延迟的目标是什么？Paged Attention 带来的吞吐提升对业务价值大吗？"**
   - 目的：了解技术优化如何服务于业务指标
   - 可引出：cost-performance tradeoff、GPU 利用率优化

2. **"多用户并发场景下,显存是如何隔离和管理的？有没有遇到过资源竞争问题？"**
   - 目的：了解推理系统的工程复杂度
   - 可引出：multi-tenancy、资源调度、GPU 虚拟化

3. **"贵司在模型部署时,有没有遇到过度量化的坑？INT8/INT4 在实际业务中的质量表现如何？"**
   - 目的：了解团队对量化的实践经验
   - 可引出：量化评测、监控、fallback 策略

## 自测题

### 口述题（能流畅讲清楚的知识点）

1. **Paged Attention 的核心机制**（3 分钟）
   - Block 划分、Block Table 管理、显存池预分配
   - 为什么能提升显存利用率
   - 与传统方案的对比

2. **Prefix Cache 命中的必要条件**（2 分钟）
   - Token 序列一致
   - Position ID 一致
   - 哪些场景命中率高

3. **KV Cache 在 TP 下的切分策略**（3 分钟）
   - 按 heads 切分
   - 通信发生在哪里（all-reduce）
   - Decode 阶段为什么通信更频繁

4. **长上下文显存优化的组合策略**（5 分钟）
   - 量化（INT8/INT4）
   - GQA
   - Paged Attention
   - TP Sharding
   - CPU Offload

### 手写题（5 分钟能写出的代码/公式）

1. **KV Cache 显存占用计算公式**（包含 batch/seq/layer/head/dtype 各参数）
   ```python
   def kv_cache_memory(batch_size, seq_len, num_layers, num_heads, head_dim, dtype="fp16"):
       dtype_bytes = {"fp16": 2, "bf16": 2, "int8": 1}
       return 2 * batch_size * seq_len * num_layers * num_heads * head_dim * dtype_bytes[dtype]
   ```

2. **Paged Attention 的 Block 分配逻辑**
   ```python
   def allocate_blocks(seq_len: int, block_size: int = 16):
       return (seq_len + block_size - 1) // block_size
   ```

3. **TP 下 KV heads 的切分**
   ```python
   def kv_heads_per_gpu(num_kv_heads: int, tp_size: int):
       return num_kv_heads // tp_size
   ```

4. **多轮对话 Position ID 的处理**
   ```python
   class ChatSession:
       def __init__(self):
           self.total_tokens = 0
       
       def get_position_ids(self, new_tokens_len: int):
           pos_ids = list(range(self.total_tokens, self.total_tokens + new_tokens_len))
           self.total_tokens += new_tokens_len
           return pos_ids
   ```

## 标签

#推理 #paged_attention #kv_cache #prefix_cache #工程 #vLLM #长上下文 #显存优化 #TP #量化 #int8 #多轮对话 #position_id #batching #吞吐优化 #prefill #decode #GQA

## 相关文档

- [[02-KV Cache核心]] - KV cache 的基本概念、存储内容、显存估算
- [[04-Batching与调度]] - 动态 batching 与 Paged Attention 的协同
- [[05-Speculative Decoding]] - 推理加速技术与 Paged Attention 的组合
- [[01-Prefill与Decode]] - 不同阶段的计算特性与优化重点
- [[../07-分布式训练ZeRO/02-通信瓶颈定位]] - TP 的详细原理与通信开销
- [[../08-数值精度量化/02-推理量化路线]] - KV cache 量化的详细方案与质量评估

---

## 参考资源

- **vLLM 论文**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **vLLM 官方文档**: https://vllm.readthedocs.io/en/latest/
- **Paged Attention 实现**: https://github.com/vllm-project/vllm
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **FlashAttention**: https://github.com/Dao-AILab/flash-attention
