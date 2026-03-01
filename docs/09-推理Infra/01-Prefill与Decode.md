# Prefill vs Decode

## 一句话结论
Prefill 是并行处理整个 prompt，计算密集型，主要消耗 GPU 算力；Decode 是逐 token 串行生成，访存密集型，主要瓶颈在 KV cache 读写带宽，分离部署可针对性优化。

## 核心定义/公式

### Prefill 阶段
- **定义**：一次性处理完整 prompt，计算所有 token 的 KV cache
- **输入**：`[batch, seq_len, hidden_dim]`
- **输出**：所有位置的 logits + 完整 KV cache
- **计算复杂度**：`O(batch × seq_len² × hidden_dim)` （注意力矩阵为 seq_len × seq_len）
- **访存量**：权重读取（一次性） + 激活写入（KV cache）

### Decode 阶段
- **定义**：基于已生成的 token，逐个生成新 token
- **输入**：`[batch, 1, hidden_dim]` （单个 token）
- **输出**：下一个 token 的 logits + 更新 KV cache
- **计算复杂度**：`O(batch × seq_len × hidden_dim)` （注意力为 1 × seq_len）
- **访存量**：权重读取 + **KV cache 读取**（每个 layer 都要读 seq_len 长度的 KV）

### 核心公式

**注意力计算量对比**：
```python
# Prefill: Q @ K^T -> [batch, heads, seq_len, seq_len]
FLOPs_prefill = 2 × batch × heads × seq_len² × head_dim

# Decode: Q @ K^T -> [batch, heads, 1, seq_len]  
FLOPs_decode = 2 × batch × heads × seq_len × head_dim

# 比例：prefill 是 decode 的 seq_len 倍（针对单步）
```

**KV Cache 显存占用**：
```python
# 每层 KV cache 大小
kv_cache_size = 2 × batch × num_layers × seq_len × num_kv_heads × head_dim × dtype_size

# 例如：LLaMA-7B, batch=32, seq_len=2048, bf16
# num_layers=32, num_kv_heads=32, head_dim=128
kv_per_token = 2 × 32 × 32 × 128 × 2 = 0.5 MB/token
total_kv = 0.5 MB × 2048 = 1 GB
```

**Decode 单步访存量**：
```python
# 每个 decode step 需要读取的 KV cache
bytes_read = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size

# seq_len=2048 时，每步需读取约 1GB（整个 KV cache）
# 权重读取 + KV cache 读取 = 主要瓶颈
```

## 为什么（2-3 个因果链）

### 1. Decode 为什么更慢：访存瓶颈主导

**因果链**：
1. **Kernel 形态差异**：Prefill 的注意力矩阵是 `[seq_len, seq_len]`，可以利用高密度矩阵乘法，GPU 利用率高；Decode 的注意力矩阵是 `[1, seq_len]`，是极度"瘦长"的矩阵乘法，GPU 利用率低（计算密度 < 10 FLOPs/Byte）。

2. **KV Cache 读写开销**：Decode 每生成一个 token，需要从显存读取**整个历史 KV cache**（所有 layer × 所有 head × 整个序列长度），访存量 = `O(seq_len)`。当 seq_len=4096 时，每步访存可达数 GB，远超权重读取。

3. **带宽 vs 算力匹配**：
   - Prefill：计算密集，`FLOPs/Bytes > 100`，算力饱和
   - Decode：访存密集，`FLOPs/Bytes ≈ 10`，算力利用率 < 20%

**公式推导**：
```python
# 计算密度（Arithmetic Intensity）
# Prefill
compute_intensity_prefill = FLOPs_prefill / (weights_read + kv_write)
                          ≈ (seq_len² × hidden_dim) / (hidden_dim²)
                          ≈ seq_len² / hidden_dim  # 通常 > 100

# Decode
compute_intensity_decode = FLOPs_decode / (weights_read + kv_read)
                         ≈ (seq_len × hidden_dim) / (hidden_dim² + seq_len × hidden_dim)
                         ≈ seq_len / (hidden_dim + seq_len)  # 通常 < 10
```

### 2. 一轮生成的瓶颈在哪里

**因果链**：
1. **短序列（seq_len < 512）**：Prefill 很快，瓶颈在 Decode 的启动延迟（kernel launch、队列调度、通信开销）。Decode 虽然单步访存小，但需要 N 次 kernel 调用。

2. **长序列（seq_len > 2048）**：Prefill 瓶颈在计算（注意力矩阵 `seq_len²` 增长），Decode 瓶颈在 KV cache 读取带宽。例如 seq_len=8192 时，每步 decode 需读取 4GB KV cache（A100 带宽 2TB/s → 单步至少 2ms）。

3. **大 batch（batch > 64）**：
   - Prefill：显存瓶颈（KV cache 需要存储 `batch × seq_len` 个 token 的 KV）
   - Decode：显存 + 带宽双重瓶颈，KV cache 呈线性增长 → OOM

**实际案例**：
```python
# LLaMA-70B, A100-80GB, batch=32, seq_len=4096
# Prefill 阶段
time_prefill ≈ 200ms  # 计算密集，算力饱和

# Decode 阶段（生成 500 tokens）
time_decode_per_token ≈ 30ms  # 访存瓶颈
time_decode_total ≈ 500 × 30ms = 15s

# 结论：长文本生成场景下，decode 占总时间 > 90%
```

### 3. Prefill/Decode 分离的价值

**因果链**：
1. **差异化硬件配置**：
   - Prefill：需要高算力 GPU（如 H100），可共享权重
   - Decode：需要高带宽显存（如 HBM3），对算力要求低 50%

2. **并行优化机会**：
   - 分离后可同时处理多个请求的 prefill 和 decode（continuous batching 的基础）
   - 避免"长 prompt 阻塞短 decode"的队头阻塞问题

3. **资源利用率提升**：
   - 未分离：长 prompt 的 prefill 会卡住所有 decode 请求
   - 分离：prefill 和 decode 可在不同 GPU 上并行，吞吐提升 2-3x

## 怎么做（可落地步骤）

### 标准做法

#### 1. Prefill 阶段优化

**Step 1：使用 FlashAttention**
```python
# 标准实现（PyTorch）
from flash_attn import flash_attn_func

# Prefill 时启用 FlashAttention
# - 分块计算 QK^T，避免 O(seq_len²) 显存
# - 保持 O(seq_len) 显存占用
output = flash_attn_func(
    q, k, v,
    causal=True,        # 因果 mask
    softmax_scale=1.0 / math.sqrt(head_dim)
)
```

**Step 2：Chunked Prefill（长序列）**
```python
# 对于超长序列（seq_len > 16K），分块处理
def chunked_prefill(model, input_ids, chunk_size=4096):
    kv_cache = None
    for start in range(0, len(input_ids), chunk_size):
        chunk = input_ids[start:start+chunk_size]
        logits, kv_cache = model(chunk, kv_cache=kv_cache)
    return logits, kv_cache
```

**Step 3：Paged Attention 内存管理**
```python
# vLLM 实现示例
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,  # 控制显存占用
    max_model_len=4096,
    block_size=16,  # KV cache 分块大小
)
```

#### 2. Decode 阶段优化

**Step 1：Continuous Batching**
```python
# vLLM 的 continuous batching
# 自动合并不同长度的 decode 请求
outputs = llm.generate(
    prompts,
    use_beam_search=False,
    max_tokens=100,
    # 内部自动实现：
    # - 动态调整 batch
    # - 提前结束的请求释放 KV cache
    # - 新请求复用显存
)
```

**Step 2：Speculative Decoding（投机解码）**
```python
# 用小模型（draft）快速生成多个候选
# 用大模型（target）并行验证
def speculative_decode(draft_model, target_model, prompt, max_steps=5):
    # Draft 阶段：快速生成 k 个 token
    draft_tokens = []
    for _ in range(max_steps):
        token = draft_model.generate_one(prompt + draft_tokens)
        draft_tokens.append(token)
    
    # Target 阶段：并行验证
    # 如果第 i 个 token 不匹配，只保留前 i-1 个
    target_probs = target_model(prompt + draft_tokens)
    accepted = verify_tokens(draft_tokens, target_probs)
    
    return prompt + accepted
```

**Step 3：KV Cache 量化**
```python
# INT8 KV cache 量化
# - 显存减少 50%
# - 带宽压力减半
# - 精度损失 < 1%（短文本）
# - 长文本需要校准和测试

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    load_in_8bit=True,  # 权重量化
    device_map="auto",
)

# vLLM 支持 KV cache 量化
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="int8",  # KV cache 量化
)
```

#### 3. Prefill/Decode 分离部署

**架构设计**：
```
┌─────────────┐         ┌─────────────┐
│   Client    │─────────│ Load Balancer│
└─────────────┘         └─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼────────┐         ┌────────▼────────┐
        │ Prefill Worker │         │ Decode Worker   │
        │  (High Compute)│         │ (High Bandwidth)│
        │  - H100 GPU    │         │  - A100 GPU     │
        │  - 共享权重     │         │  - 更多显存     │
        └────────────────┘         └─────────────────┘
                │                           │
                └───────────┬───────────────┘
                            │
                    ┌───────▼────────┐
                    │  Shared Storage│
                    │  (Weights +    │
                    │   KV Cache)    │
                    └────────────────┘
```

**实现步骤**：
```python
# Step 1: Prefill Worker
def prefill_worker(prompt_ids):
    # 高算力 GPU
    with torch.cuda.device(prefill_gpu):
        logits, kv_cache = model(prompt_ids)
    # 发送 KV cache 到 decode worker
    send_kv_cache(kv_cache, decode_worker_id)
    return logits

# Step 2: Decode Worker
def decode_worker(kv_cache, max_tokens):
    # 高带宽 GPU
    with torch.cuda.device(decode_gpu):
        generated = []
        for _ in range(max_tokens):
            # 只需读取 KV cache + 计算单个 token
            token, kv_cache = model.generate_one(kv_cache)
            generated.append(token)
    return generated

# Step 3: 协调器
def inference_orchestrator(prompt):
    # 阶段 1: Prefill
    kv_cache = prefill_worker(prompt)
    
    # 阶段 2: Decode
    output = decode_worker(kv_cache, max_tokens=100)
    return output
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `max_batch_size` | Prefill: 8-16<br>Decode: 32-128 | Prefill 受显存限制，Decode 受带宽限制 |
| `block_size` (Paged Attn) | 16 | 平衡碎片和内存管理开销 |
| `chunk_size` (Chunked Prefill) | 4096 | 避免 OOM，保持并行度 |
| `speculative_steps` | 4-8 | 过大反而降低接受率 |
| `kv_cache_dtype` | bf16 (默认)<br>int8 (长序列) | 权衡精度与带宽 |
| `gpu_memory_utilization` | 0.85-0.95 | 预留空间避免 OOM |

### 代码示例

**完整 Prefill + Decode 流程**：
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInference:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.kv_cache = None
    
    def prefill(self, prompt):
        """Prefill 阶段：处理完整 prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        with torch.no_grad():
            # 一次性计算所有 token
            outputs = self.model(
                input_ids,
                use_cache=True,
                return_dict=True
            )
        
        self.kv_cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # 只需要最后一个 token
        return logits
    
    def decode_step(self, last_token_id):
        """Decode 阶段：生成单个 token"""
        with torch.no_grad():
            # 只传入单个 token，复用 KV cache
            outputs = self.model(
                last_token_id.unsqueeze(0),
                past_key_values=self.kv_cache,
                use_cache=True,
                return_dict=True
            )
        
        self.kv_cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        return logits
    
    def generate(self, prompt, max_new_tokens=100):
        """完整生成流程"""
        # Step 1: Prefill
        logits = self.prefill(prompt)
        next_token = torch.argmax(logits, dim=-1)
        generated = [next_token.item()]
        
        # Step 2: Decode loop
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_token)
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token.item())
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)

# 使用示例
llm = LLMInference("meta-llama/Llama-2-7b-hf")
output = llm.generate("Explain machine learning in simple terms:")
print(output)
```

**KV Cache 显存监控**：
```python
def estimate_kv_cache_size(model, batch_size, seq_len, dtype_size=2):
    """估算 KV cache 显存占用"""
    config = model.config
    
    # 每层的 KV cache
    kv_per_layer = 2 * batch_size * seq_len * config.num_attention_heads * config.hidden_size // config.num_attention_heads
    
    # 总计
    total_kv = kv_per_layer * config.num_hidden_layers * dtype_size
    
    return total_kv / (1024**3)  # GB

# 示例：LLaMA-7B, batch=32, seq_len=2048
# estimate_kv_cache_size(model, 32, 2048) ≈ 16 GB
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **FlashAttention** | Prefill 显存从 O(seq_len²) → O(seq_len)<br>长序列速度提升 2-4x | 需要特定 GPU 架构（Ampere+）<br>实现复杂度高 | seq_len > 1024<br>训练 + 推理 |
| **Continuous Batching** | 吞吐提升 2-3x<br>降低尾部延迟 | 调度逻辑复杂<br>可能影响短请求延迟 | 高并发场景<br>batch > 8 |
| **Speculative Decoding** | Decode 速度提升 1.5-3x<br>保持生成质量 | 需要 draft model<br>接受率不稳定时反而慢 | 长文本生成<br>预算充足 |
| **KV Cache 量化（INT8）** | 显存减半<br>带宽压力减半 | 长序列精度损失<br>需要校准数据 | 长上下文<br>显存受限 |
| **Prefill/Decode 分离** | 吞吐提升 2-3x<br>差异化硬件配置 | 架构复杂<br>需要 KV cache 传输 | 大规模服务<br>有独立 GPU 池 |
| **Paged Attention** | 显存利用率 > 95%<br>支持变长序列 | 实现复杂<br>额外内存管理开销 | 多租户场景<br>变长输入 |

## 高频追问（至少 5 个）

### 1. Q: Prefill 和 Decode 的具体时间占比大概是多少？

**A**: 取决于序列长度和生成长度：
- **短序列 + 短生成**（prompt=128, gen=50）：Prefill 占 20-30%，Decode 占 70-80%
- **长序列 + 长生成**（prompt=2048, gen=500）：Prefill 占 5-10%，Decode 占 90-95%
- **长序列 + 短生成**（prompt=4096, gen=50）：Prefill 占 40-50%，Decode 占 50-60%

公式：`prefill_ratio ≈ (prompt_len)² / (prompt_len² + gen_len × prompt_len)`，实际还受 batch size 和硬件影响。

### 2. Q: KV Cache 影响的是算力还是带宽？

**A**: **主要影响带宽**，具体分析：
1. **Decode 阶段**：每步需读取整个 KV cache（`O(seq_len)`），计算量只有 `O(1)`，是典型的**带宽瓶颈**
2. **Prefill 阶段**：只需写入 KV cache 一次，后续计算密集，主要影响**显存容量**
3. **量化视角**：KV cache 从 fp16 → int8，带宽需求减半，但对算力需求几乎不变
4. **实测数据**：A100 带宽 2TB/s，读取 4GB KV cache 至少 2ms，而单步 decode 计算只需 0.5ms，带宽是瓶颈的 4x

### 3. Q: 为什么 Batch 一大就 OOM？

**A**: **KV cache 随 batch 线性增长**：
```python
# KV cache 公式
kv_size = 2 × num_layers × batch × seq_len × num_kv_heads × head_dim × 2 bytes

# LLaMA-70B 示例（num_layers=80, num_kv_heads=8, head_dim=128）
# batch=64, seq_len=2048
kv_size = 2 × 80 × 64 × 2048 × 8 × 128 × 2 = 80 GB

# batch=128 → 160 GB（超过 A100-80GB）
```

**缓解方案**：
1. Paged Attention：按需分配，避免预留
2. KV Cache 量化：int8/fp8 减半
3. GQA/MQA：减少 kv_heads 数量
4. 限制 max_batch_size
5. 梯度检查点（训练时）

### 4. Q: Prefill/Decode 分离后，KV Cache 怎么传输？

**A**: 有三种方案：
1. **共享显存**（单机多卡）：
   - KV cache 放在共享内存池
   - Prefill worker 写入，Decode worker 读取
   - 延迟 < 1ms，带宽 > 500GB/s（NVLink）

2. **网络传输**（跨机）：
   - 序列化为 tensor + metadata
   - 使用 RDMA 或 NCCL
   - 延迟 5-20ms，带宽 10-50GB/s
   - **只适合长序列场景**（prefill 占比 > 50%）

3. **分层存储**：
   - 近期 KV cache 在 GPU
   - 历史 KV cache 在 CPU/SSD
   - 按需加载（类似虚拟内存）

### 5. Q: Continuous Batching 为什么能提吞吐？

**A**: **核心是消除空闲和队头阻塞**：
1. **传统 batching**：
   - 一个 batch 内所有请求必须等最慢的那个
   - 短请求完成后 GPU 空闲
   - 长请求阻塞后续请求

2. **Continuous batching**：
   - 某个请求生成完成后，立即释放 KV cache
   - 新请求可以加入正在运行的 batch
   - GPU 利用率从 40-60% → 80-95%

3. **数学直觉**：
   - 传统：吞吐 ≈ batch_size / max_latency
   - Continuous：吞吐 ≈ batch_size / avg_latency
   - 当长短请求混合时，avg_latency << max_latency

### 6. Q: FlashAttention 主要优化 Prefill 还是 Decode？

**A**: **主要优化 Prefill，Decode 用 FlashDecoding**：
1. **FlashAttention**（Prefill）：
   - 问题：`QK^T` 产生 `seq_len × seq_len` 矩阵，显存 O(seq_len²)
   - 优化：分块计算 + 在线 softmax，显存 O(seq_len)
   - 收益：长序列（> 2048）速度提升 2-4x

2. **FlashDecoding**（Decode）：
   - 问题：`QK^T` 是 `1 × seq_len`，已很小，但需要并行化读取 KV
   - 优化：将 KV cache 分成多个 chunk 并行计算，最后 all-reduce
   - 收益：长序列下 decode 速度提升 1.5-2x

3. **结论**：
   - Prefill：FlashAttention 是**必需品**
   - Decode：FlashDecoding 是**优化项**，基础实现已足够

### 7. Q: Speculative Decoding 接受率低会怎样？

**A**: **反而比标准 decode 更慢**：
1. **理想情况**（接受率 > 70%）：
   - Draft 模型生成 5 个 token
   - Target 模型接受 4 个
   - 等价于 5 个 token 用 2 次前向，速度 2.5x

2. **接受率低**（< 50%）：
   - Draft + Target 都要计算，且 Target 需要完整前向
   - 开销 = draft_time + target_time > target_time（标准 decode）

3. **何时接受率高**：
   - 简单任务（翻译、摘要）
   - Draft 模型与 Target 模型差距小（同系列小大模型）
   - 确定性高的输出（代码、格式化文本）

4. **何时不推荐**：
   - 创造性任务（接受率 < 40%）
   - Draft 和 Target 差异大
   - 单次请求延迟敏感场景

### 8. Q: 一轮生成的瓶颈在 Prefill 还是 Decode？

**A**: **取决于 prompt 长度和生成长度**：
```python
# 时间估算公式
time_prefill ≈ α × prompt_len² × hidden_dim / FLOPs
time_decode ≈ β × gen_len × prompt_len × hidden_dim / bandwidth

# 典型场景分析
# 场景 1: 翻译（prompt=100, gen=100）
# prefill ≈ 10ms, decode ≈ 100 × 0.5ms = 50ms
# 瓶颈：Decode

# 场景 2: 长文档摘要（prompt=8000, gen=500）
# prefill ≈ 800ms, decode ≈ 500 × 5ms = 2500ms
# 瓶颈：Decode（但 Prefill 显著）

# 场景 3: 代码补全（prompt=2000, gen=20）
# prefill ≈ 100ms, decode ≈ 20 × 2ms = 40ms
# 瓶颈：Prefill

# 场景 4: 多轮对话（prompt=4000, gen=150）
# prefill ≈ 300ms, decode ≈ 150 × 3ms = 450ms
# 瓶颈：Decode（但两者接近）
```

**实践建议**：
- 短生成（< 50 tokens）：关注 Prefill 优化（FlashAttention）
- 长生成（> 100 tokens）：关注 Decode 优化（Continuous Batching、KV Cache 量化）

### 9. Q: 多卡推理时，KV Cache 怎么切分？

**A**: 取决于并行策略：

**1. Tensor Parallelism（TP）**：
```python
# KV cache 按 head 切分
# 每张卡存储：num_heads/tp_size 个 head 的 KV

# 例如 LLaMA-70B, tp_size=4
# num_kv_heads = 8 → 每张卡 2 个 kv_head
# 优点：无需通信，每张卡独立
# 缺点：单卡显存压力大

# 伪代码
for layer in model.layers:
    kv_cache[layer] = {
        'k': split(kv[layer].k, num_kv_heads, tp_size),
        'v': split(kv[layer].v, num_kv_heads, tp_size)
    }
```

**2. Pipeline Parallelism（PP）**：
```python
# KV cache 按 layer 切分
# 第 i 张卡存储 layer [i*layers/pp_size, (i+1)*layers/pp_size) 的 KV

# 例如 LLaMA-70B, pp_size=4, num_layers=80
# 每张卡 20 层
# 优点：单卡显存压力小
# 缺点：需要流水线调度

# 需要在 PP 边界传递 KV cache（通信开销）
```

**3. Sequence Parallelism（SP）**：
```python
# KV cache 按 sequence 切分（用于超长序列）
# 每张卡存储 seq_len/sp_size 的 KV

# 优点：支持超长序列（> 100K）
# 缺点：每步都需要 all-gather
# 目前主要用于训练，推理较少
```

### 10. Q: 长上下文下，Prefill 会有什么问题？

**A**: **三个主要瓶颈**：

**1. 计算量爆炸**：
```python
# 注意力复杂度 O(seq_len²)
FLOPs = 2 × batch × layers × heads × seq_len² × head_dim

# seq_len 从 4K → 32K
# FLOPs 增加 64 倍
# 时间从 100ms → 6.4s
```

**缓解方案**：
- FlashAttention-2（优化并行性）
- Ring Attention（分布式注意力）
- 稀疏注意力（Longformer、BigBird）

**2. 显存不足**：
```python
# KV cache 大小
kv_size = 2 × batch × layers × seq_len × kv_heads × head_dim × 2 bytes

# LLaMA-70B, batch=1, seq_len=32K
kv_size = 2 × 80 × 1 × 32768 × 8 × 128 × 2 = 10 GB

# 加上激活、权重，需要 > 100 GB 显存
```

**缓解方案**：
- Paged Attention
- KV Cache 量化（INT8/FP8）
- Gradient Checkpointing（训练）

**3. 距离衰减**：
- 长序列下，远距离 token 的注意力权重被稀释
- 需要位置编码外推（YaRN、NTK-aware）

## 常见错误（至少 3 个）

### 1. 错误：认为 Prefill 和 Decode 计算量相同

**错误理解**：
"Prefill 和 Decode 都是做注意力计算，计算量应该差不多"

**正确理解**：
```python
# Prefill: O(seq_len²)
# Decode: O(seq_len)

# 示例：seq_len=2048
prefill_ops ≈ 2048² = 4M operations
decode_ops ≈ 2048 operations
# Prefill 计算量是 Decode 的 2048 倍（单步对比）
```

**正确做法**：
- Prefill 优化重点：算力（FlashAttention、并行化）
- Decode 优化重点：带宽（KV Cache 量化、Continuous Batching）

### 2. 错误：忽略 Decode 阶段的带宽瓶颈

**错误实现**：
```python
# 只关注算力，忽略 KV cache 读取
def decode_step(model, token, kv_cache):
    # 假设算力是瓶颈
    logits = model(token, kv_cache)
    return logits
```

**问题**：
- Decode 每步需要读取整个 KV cache（数 GB）
- 在 A100 上，带宽 2TB/s → 读取 4GB 至少 2ms
- 实际计算只需 0.5ms → **带宽是瓶颈的 4x**

**正确做法**：
```python
# 1. KV Cache 量化
kv_cache_int8 = quantize(kv_cache, dtype=torch.int8)

# 2. Paged Attention（减少碎片）
from vllm import LLM
llm = LLM(model="...", block_size=16)

# 3. Continuous Batching（提高带宽利用率）
# vLLM 自动处理

# 4. 监控带宽使用
# nvidia-smi dmon -s u
```

### 3. 错误：KV Cache 没有显式管理，导致 OOM

**错误场景**：
```python
# 直接使用 HuggingFace generate
model.generate(prompt, max_length=4096)
# 当 batch_size 大或序列长时 OOM
```

**根因**：
- HuggingFace 默认预留 `max_length` 的连续显存
- 实际生成可能提前结束 → 浪费
- 不同请求长度不一 → 碎片化

**正确做法**：
```python
# 使用 Paged Attention（vLLM）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,  # 控制显存占用
    max_model_len=4096,
    block_size=16,  # 分块管理 KV cache
)

sampling_params = SamplingParams(
    max_tokens=500,
    temperature=0.8,
)

outputs = llm.generate(prompts, sampling_params)
# 自动管理 KV cache，按需分配，提前释放
```

### 4. 错误：Speculative Decoding 不评估接受率直接上

**错误场景**：
```python
# 直接部署，未测试接受率
draft_model = LLM("small-model")
target_model = LLM("large-model")
output = speculative_decode(draft_model, target_model, prompt)
# 结果发现接受率只有 30%，比标准 decode 还慢
```

**正确流程**：
```python
# Step 1: 评估接受率
def measure_acceptance_rate(draft, target, prompts, max_steps=5):
    total_tokens = 0
    accepted_tokens = 0
    
    for prompt in prompts:
        draft_tokens = draft.generate(prompt, max_tokens=max_steps)
        target_probs = target.get_probs(prompt + draft_tokens)
        accepted = verify(draft_tokens, target_probs)
        
        total_tokens += len(draft_tokens)
        accepted_tokens += len(accepted)
    
    return accepted_tokens / total_tokens

acceptance_rate = measure_acceptance_rate(draft_model, target_model, test_prompts)

# Step 2: 决策
if acceptance_rate > 0.6:
    print(f"Acceptance rate {acceptance_rate:.2f}, use speculative decoding")
else:
    print(f"Acceptance rate {acceptance_rate:.2f}, use standard decoding")

# Step 3: 监控线上接受率
# 如果持续 < 0.5，关闭 speculative decoding
```

### 5. 错误：混淆 Prefill/Decode 分离和 Continuous Batching

**错误理解**：
"Prefill/Decode 分离就是 Continuous Batching"

**正确区分**：
- **Continuous Batching**：在**同一个 GPU** 上，动态调整 batch，允许请求随时加入/退出
- **Prefill/Decode 分离**：将 Prefill 和 Decode 放在**不同 GPU** 上，用不同硬件配置

**关系**：
```python
# 场景 1: 单 GPU + Continuous Batching
# 优点：简单，无需架构改动
# 缺点：Prefill 会阻塞 Decode（长 prompt 卡住所有请求）

# 场景 2: 多 GPU + Prefill/Decode 分离
# 优点：Prefill 和 Decode 并行，吞吐最高
# 缺点：架构复杂，需要 KV cache 传输

# 最佳实践：分离 + Continuous Batching
# Prefill worker: 用 Continuous Batching 处理多个 prefill
# Decode worker: 用 Continuous Batching 处理多个 decode
```

## 反问面试官的问题

### 技术深度类

1. **你们线上推理服务的 TTFT（Time To First Token）和 TPOT（Time Per Output Token）指标大概是多少？主要瓶颈在 Prefill 还是 Decode？**
   - 了解实际性能基线
   - 判断是否有优化空间
   - 确认是否需要 prefetch/prefix cache

2. **长上下文场景（比如 32K+），你们是用 Ring Attention 还是其他方案？Prefill 阶段的显存瓶颈怎么解决的？**
   - 了解技术栈选择
   - 判断是否了解最新技术（FlashAttention-2、Ring Attention）
   - 评估工程能力

3. **KV Cache 的量化在你们场景下的精度损失可接受吗？有做过对比测试吗？长序列和短序列的表现差异大吗？**
   - 了解工程权衡
   - 判断是否有系统化测试
   - 评估对精度的重视程度

### 业务场景类

1. **你们的用户请求分布是怎样的？短 prompt + 短生成，还是长 prompt + 长生成？这会影响 Prefill/Decode 分离的收益评估。**
   - 了解业务特点
   - 判断优化方向
   - 确认是否值得投入

2. **线上出现过 Prefill 阻塞 Decode 导致延迟飙升的情况吗？是怎么监控和处理的？**
   - 了解线上问题
   - 判断监控体系是否完善
   - 评估故障处理能力

3. **未来有计划支持更长的上下文（比如 128K+）吗？这会对现有的 Prefill/Decode 架构带来什么挑战？**
   - 了解技术规划
   - 判断是否有前瞻性
   - 评估技术储备

## 自测题

### 口述（能流畅讲清楚的知识点）

1. **Prefill 和 Decode 的本质区别是什么？为什么 Decode 更慢？**
   - 关键点：计算密集 vs 访存密集、矩阵形态差异、KV cache 读写开销

2. **一轮生成的瓶颈通常在哪？如何判断？**
   - 关键点：短序列/短生成 → Decode；长序列/长生成 → Decode；长序列/短生成 → Prefill
   - 公式：`time_prefill ≈ α × prompt_len²`，`time_decode ≈ β × gen_len × prompt_len`

3. **Continuous Batching 为什么能提吞吐？有什么代价？**
   - 关键点：消除队头阻塞、提高 GPU 利用率、实现复杂度高

4. **KV Cache 影响的是算力还是带宽？为什么？**
   - 关键点：带宽（Decode 每步读取整个 KV cache）、计算密度低

5. **FlashAttention 主要优化 Prefill 还是 Decode？为什么？**
   - 关键点：Prefill（解决 O(seq_len²) 显存问题）、Decode 已是 O(seq_len) 且有 FlashDecoding

### 手写（5 分钟能写出的代码/公式）

1. **写出 KV Cache 显存估算公式（给定模型配置）**
```python
def estimate_kv_cache(model, batch, seq_len, dtype_bytes=2):
    """
    LLaMA-70B: layers=80, kv_heads=8, head_dim=128
    """
    layers = model.config.num_hidden_layers
    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    kv_size = 2 * batch * layers * seq_len * kv_heads * head_dim * dtype_bytes
    return kv_size / (1024**3)  # GB
```

2. **写出 Prefill 和 Decode 的计算复杂度公式**
```python
# Prefill: 注意力矩阵 QK^T 是 [seq_len, seq_len]
FLOPs_prefill = 2 * batch * layers * heads * seq_len^2 * head_dim

# Decode: 注意力矩阵 QK^T 是 [1, seq_len]
FLOPs_decode = 2 * batch * layers * heads * seq_len * head_dim

# 比例
ratio = FLOPs_prefill / FLOPs_decode = seq_len
```

3. **写一个简单的 Continuous Batching 伪代码**
```python
def continuous_batching(requests, model, max_batch_size=32):
    running = []  # 正在运行的请求
    completed = []  # 已完成的请求
    
    while requests or running:
        # Step 1: 移除已完成的请求
        running = [r for r in running if not r.is_finished()]
        
        # Step 2: 加入新请求（如果还有空位）
        while len(running) < max_batch_size and requests:
            new_req = requests.pop(0)
            running.append(new_req)
        
        # Step 3: 批量推理（prefill + decode step）
        batch = Batch(running)
        outputs = model.forward(batch)
        
        # Step 4: 更新每个请求
        for req, out in zip(running, outputs):
            req.append_token(out.token)
            if req.is_finished():
                completed.append(req)
    
    return completed
```

4. **写一个判断瓶颈在 Prefill 还是 Decode 的函数**
```python
def identify_bottleneck(prompt_len, gen_len, seq_len_threshold=2048):
    """
    简化版判断逻辑
    """
    # 时间估算（简化）
    time_prefill = prompt_len ** 2 / 1e6  # 假设 1M FLOPs = 1ms
    time_decode = gen_len * prompt_len / 1e6
    
    # 判断
    if time_prefill > time_decode:
        return "Prefill is bottleneck"
    else:
        return "Decode is bottleneck"
    
    # 更精确的判断需要考虑硬件、batch size 等
```

## 标签
#推理 #prefill #decode #KV_cache #FlashAttention #continuous_batching #speculative_decoding #paged_attention #工程 #性能优化 #vLLM

## 相关文档
- [[02-KV Cache核心]] - KV Cache 详细原理与优化
- [[03-Paged Attention与显存管理]] - Paged Attention 机制
- [[04-Continuous Batching]] - 动态批处理详解
- [[10-FlashAttention原理]] - FlashAttention 与算子优化
- [[11-Speculative Decoding]] - 投机解码加速技术

---

## 补充：关键指标速查表

| 指标 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| **计算复杂度** | O(seq_len²) | O(seq_len) | Prefill 是 Decode 的 seq_len 倍 |
| **访存模式** | 权重读取（一次性） | 权重 + KV Cache（每步） | Decode 访存量随序列增长 |
| **GPU 利用率** | 高（80-95%） | 低（10-30%） | Decode 受带宽限制 |
| **主要瓶颈** | 算力 | 带宽 | 决定优化方向 |
| **优化重点** | FlashAttention | KV Cache 量化、Continuous Batching | 针对性优化 |
| **典型时间** | 100-500ms | 20-50ms/token | 长序列下差异更大 |
| **显存占用** | O(batch × seq_len²) | O(batch × seq_len) | Prefill 注意力矩阵大 |

## 补充：决策树

```
推理场景分析
├─ 短序列（seq < 512）
│  ├─ 短生成（gen < 50）
│  │  └─ 瓶颈：Decode 启动延迟
│  │     优化：Kernel Fusion、减少 kernel launch
│  └─ 长生成（gen > 100）
│     └─ 瓶颈：Decode 带宽
│        优化：Continuous Batching、Speculative Decoding
│
├─ 长序列（seq > 2048）
│  ├─ 短生成（gen < 50）
│  │  └─ 瓶颈：Prefill 计算
│  │     优化：FlashAttention、Chunked Prefill
│  └─ 长生成（gen > 100）
│     └─ 瓶颈：Decode 带宽 + KV Cache 显存
│        优化：KV Cache 量化、Paged Attention、Continuous Batching
│
└─ 超长序列（seq > 16K）
   └─ 所有情况
      └─ 瓶颈：显存 + 计算
         优化：Ring Attention、KV Cache Offload、Sequence Parallelism
```