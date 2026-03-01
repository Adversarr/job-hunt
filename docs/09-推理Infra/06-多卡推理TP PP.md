# 多卡推理（TP/PP）与通信

## 一句话结论
TP 通过切分权重矩阵降低单卡显存，但每层需要 all-reduce 同步；PP 通过切分层降低显存，但引入流水线 bubble 和跨卡依赖；多卡推理的瓶颈主要在通信开销和负载均衡，decode 阶段因计算量小、batch 小，通信占比更高，更容易被通信拖慢。

## 核心定义/公式

### Tensor Parallelism (TP)

**定义**：将权重矩阵按列或行切分到多张卡上，每张卡计算部分结果，通过 all-reduce 合并。

**切分方式**：
```python
# 列切分（Column Parallel）
# Y = XW, W ∈ [in_dim, out_dim]
# W 切分为 [W_1, W_2, ..., W_n], 每个 W_i ∈ [in_dim, out_dim/n]
# 每张卡计算 Y_i = X @ W_i, 然后 all-gather 得到完整 Y

# 行切分（Row Parallel）
# Y = XW, W ∈ [in_dim, out_dim]
# W 切分为 [W_1; W_2; ...; W_n], 每个 W_i ∈ [in_dim/n, out_dim]
# 输入 X 也切分, X_i ∈ [batch, in_dim/n]
# 每张卡计算 Y_i = X_i @ W_i, 然后 all-reduce 得到 Y
```

**Attention 层 TP 切分**：
```python
# QKV 投影：列切分
Q_i = X @ W_Q_i  # 每张卡计算部分头
K_i = X @ W_K_i
V_i = X @ W_V_i

# Attention 计算：本地计算，无需通信
Attn_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i

# Output 投影：行切分
O_i = Attn_i @ W_O_i
O = all_reduce(O_i)  # 合并结果
```

### Pipeline Parallelism (PP)

**定义**：将模型按层切分到不同卡上，每张卡负责连续几层的计算，数据在卡间流水传递。

**流水线调度**：
```python
# 4 卡 PP，每卡 6 层（假设 24 层模型）
# GPU 0: Layer 0-5
# GPU 1: Layer 6-11
# GPU 2: Layer 12-17
# GPU 3: Layer 18-23

# Micro-batch 流水
# Time  | GPU0 | GPU1 | GPU2 | GPU3 |
# 0     | mb0  | -    | -    | -    |
# 1     | mb1  | mb0  | -    | -    |
# 2     | mb2  | mb1  | mb0  | -    |
# 3     | mb3  | mb2  | mb1  | mb0  |
# 4     | -    | mb3  | mb2  | mb1  |
```

**Bubble 计算公式**：
$$
\text{Bubble Ratio} = \frac{(p-1) \times \text{micro\_batch\_time}}{p \times \text{micro\_batch\_time} \times \text{num\_stages}} = \frac{p-1}{p \times \text{num\_stages}}
$$
其中 $p$ 为 micro-batch 数量。

### 通信原语

**All-Reduce**：
```python
# 所有卡将数据汇总并广播给所有卡
# 实现：Reduce-Scatter + All-Gather
# 通信量：2(n-1)/n × data_size (Ring AllReduce)
```

**All-Gather**：
```python
# 所有卡将本地数据收集并广播给所有卡
# 通信量：(n-1)/n × data_size
```

### KV Cache 在 TP 下的存储

**MHA 模式**：
```python
# 每张卡存储部分头的 KV cache
# GPU i 存储 head [i * heads_per_gpu : (i+1) * heads_per_gpu]
KV_cache_i = [batch, seq_len, 2, heads_per_gpu, head_dim]
# 无需跨卡同步，本地计算即可
```

**GQA/MQA 模式**：
```python
# MQA: K/V 只有一份，需要广播到所有卡
# 或复制到每张卡（增加显存），或集中存储在某卡（通信）
# GQA: K/V 按组存储，组内广播
KV_cache = [batch, seq_len, 2, num_groups, head_dim]
# 每组 K/V 需要广播到组内的头
```

## 为什么（2-3 个因果链）

### 1. 为什么 TP/PP 各有瓶颈

**TP 瓶颈因果链**：
- 权重切分 → 每层需要 all-reduce 合并 → 通信开销随 TP 度数线性增长 → decode 阶段计算量小（batch=1, seq=1）→ 通信占比高（可能 30-50%）→ 吞吐下降

**PP 瓶颈因果链**：
- 层切分 → 流水线依赖 → 前向传播需等待上游 → 反向传播需等待下游 → micro-batch 数量不足时产生 bubble → GPU 空闲率高 → 利用率下降

**关键指标**：
```python
# TP 通信开销
tp_comm_cost = num_layers × 2 × hidden_size × sizeof(dtype) / bandwidth
# 每层一次 all-reduce（前向 + 反向）

# PP Bubble 开销
pp_bubble_ratio = (num_stages - 1) / (num_stages × num_micro_batches)
# 8 卡 PP，4 个 micro-batch：bubble = 7/32 ≈ 22%
```

### 2. 为什么通信会拖慢 decode

**因果链**：
- decode 阶段特性 → batch 小（通常 1-64）、seq 短（生成 1 token）→ 计算量极小（矩阵乘法很小）→ GPU 计算时间短（微秒级）→ 通信时间相对占比高（毫秒级）→ 通信延迟掩盖不住 → decode 性能下降

**数据对比**：
```python
# Prefill 阶段（batch=16, seq=1024）
compute_time = 50ms  # 大矩阵乘法
comm_time = 2ms      # all-reduce
comm_ratio = 4%      # 通信占比低

# Decode 阶段（batch=16, seq=1）
compute_time = 0.5ms  # 小矩阵向量乘
comm_time = 2ms       # all-reduce（不变）
comm_ratio = 80%      # 通信占比高
```

**优化方向**：
- **Tensor Parallelism 优化**：使用 NVLink/InfiniBand 降低延迟
- **计算通信重叠**：异步通信、流水化
- **减少 TP 度数**：单卡能放下时不用多卡

### 3. 为什么多机更容易抖动

**因果链**：
- 多机通信 → 网络延迟（InfiniBand/以太网）→ 延迟不稳定（网络拥塞、路由跳变）→ 同步等待时间波动 → 每步推理时间波动 → 吞吐抖动

**抖动来源**：
1. **网络延迟波动**：跨机通信经过交换机，拥塞时延迟增加
2. **时钟不同步**：各机器时钟偏差，影响同步
3. **负载不均**：各机器负载不同，处理速度差异
4. **网络拓扑**：不同机器的网络跳数不同

**量化示例**：
```python
# 单机 8 卡（NVLink）
latency_mean = 10μs
latency_var = 1μs
jitter = 10%

# 多机 16 卡（InfiniBand）
latency_mean = 50μs
latency_var = 20μs
jitter = 40%
```

## 怎么做（可落地步骤）

### 标准做法

#### 1. TP/PP 选择策略

```python
def choose_parallel_strategy(model_size, gpu_memory, batch_size, seq_len, num_gpus):
    """
    选择并行策略的经验法则
    """
    # 单卡能否放下
    if model_size <= gpu_memory:
        return "单卡推理"
    
    # 估算显存需求
    weights_memory = model_size * 2  # FP16
    kv_cache_memory = batch_size * seq_len * num_layers * 2 * num_heads * head_dim * 2
    activation_memory = batch_size * seq_len * hidden_size * 4  # 估算
    
    total_memory = weights_memory + kv_cache_memory + activation_memory
    
    # TP 度数估算
    tp_degree = math.ceil(total_memory / gpu_memory)
    
    if tp_degree <= num_gpus:
        # 优先使用 TP（通信延迟更低）
        return f"TP-{tp_degree}"
    else:
        # 需要 TP + PP 混合
        pp_degree = math.ceil(tp_degree / num_gpus)
        tp_degree_per_node = num_gpus // pp_degree
        return f"TP-{tp_degree_per_node}-PP-{pp_degree}"
```

**推荐配置**：
- **单卡能放下**：不用并行，最简单高效
- **2-8 卡**：优先 TP，NVLink 通信延迟低
- **16+ 卡**：TP + PP 混合，或考虑 EP（Expert Parallelism，用于 MoE）

#### 2. KV Cache 在 TP 下的切分策略

```python
class TensorParallelKVCache:
    def __init__(self, num_layers, num_heads, head_dim, tp_rank, tp_world_size):
        self.num_layers = num_layers
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        
        # 计算每张卡负责的头数
        self.heads_per_gpu = num_heads // tp_world_size
        self.head_start = tp_rank * self.heads_per_gpu
        self.head_end = (tp_rank + 1) * self.heads_per_gpu
        
        # 分配 KV cache 显存
        self.cache = torch.zeros(
            num_layers, 2, max_batch_size, max_seq_len,
            self.heads_per_gpu, head_dim,
            device='cuda', dtype=torch.float16
        )
    
    def update(self, layer_idx, new_k, new_v, positions):
        # 只存储本卡负责的头
        k = new_k[:, :, self.head_start:self.head_end, :]
        v = new_v[:, :, self.head_start:self.head_end, :]
        
        self.cache[layer_idx, 0, :, positions, :, :] = k
        self.cache[layer_idx, 1, :, positions, :, :] = v
    
    def get(self, layer_idx, positions):
        k = self.cache[layer_idx, 0, :, positions, :, :]
        v = self.cache[layer_idx, 1, :, positions, :, :]
        return k, v
```

#### 3. GQA/MQA 在多卡下的优化

```python
class GQAMultiGPU:
    def __init__(self, num_heads, num_groups, tp_world_size):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.tp_world_size = tp_world_size
        
        # GQA: 每组 K/V 共享
        self.groups_per_gpu = num_groups // tp_world_size
        
    def forward(self, Q, K, V):
        # Q 按头切分到各卡
        Q_local = Q[:, :, self.head_start:self.head_end, :]
        
        # K/V 按组切分
        K_local = K[:, :, self.group_start:self.group_end, :]
        V_local = V[:, :, self.group_start:self.group_end, :]
        
        # 组内广播 K/V
        K_expanded = K_local.unsqueeze(2).expand(
            -1, -1, self.heads_per_group, -1, -1
        )
        V_expanded = V_local.unsqueeze(2).expand(
            -1, -1, self.heads_per_group, -1, -1
        )
        
        # 本地计算 Attention
        scores = torch.matmul(Q_local, K_expanded.transpose(-2, -1))
        attn = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
        output = torch.matmul(attn, V_expanded)
        
        return output
```

#### 4. 性能 Profiling 关键指标

```python
import torch.distributed as dist
import time

class InferenceProfiler:
    def __init__(self):
        self.metrics = {
            'compute_time': [],
            'comm_time': [],
            'total_time': [],
            'throughput': [],
        }
    
    def profile_step(self, func):
        # 计算时间
        torch.cuda.synchronize()
        start_compute = time.time()
        output = func()
        torch.cuda.synchronize()
        end_compute = time.time()
        
        # 通信时间（通过 all-reduce 模拟）
        torch.cuda.synchronize()
        start_comm = time.time()
        if dist.is_initialized():
            dist.all_reduce(torch.zeros(1, device='cuda'))
        torch.cuda.synchronize()
        end_comm = time.time()
        
        # 记录指标
        self.metrics['compute_time'].append(end_compute - start_compute)
        self.metrics['comm_time'].append(end_comm - start_comm)
        self.metrics['total_time'].append(end_compute - start_compute + end_comm - start_comm)
        
        return output
    
    def get_summary(self):
        return {
            'avg_compute_ms': np.mean(self.metrics['compute_time']) * 1000,
            'avg_comm_ms': np.mean(self.metrics['comm_time']) * 1000,
            'comm_ratio': np.mean(self.metrics['comm_time']) / np.mean(self.metrics['total_time']),
            'throughput_tokens_per_sec': 1.0 / np.mean(self.metrics['total_time']),
        }
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `tp_degree` | 2-8（单机内） | NVLink 带宽高（>300GB/s），延迟低 |
| `pp_degree` | 2-4 | 过大导致 bubble 过多，利用率下降 |
| `micro_batch_size` | 1-8（PP） | 增加可减少 bubble，但增加显存 |
| `kv_cache_dtype` | FP16/BF16 | INT8 量化可能损失精度，长序列更明显 |
| `gqa_num_groups` | 8（32 头模型） | 平衡显存节省与精度损失 |

### 代码示例：vLLM/TensorRT-LLM 配置

```python
# vLLM TP 配置
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen-72B",
    tensor_parallel_size=8,  # 8 卡 TP
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# TensorRT-LLM PP 配置
import tensorrt_llm

# 构建引擎时指定 PP
builder_config = tensorrt_llm.builder.BuilderConfig(
    max_batch_size=16,
    max_input_len=1024,
    max_output_len=512,
    tensor_parallel=4,  # TP 度数
    pipeline_parallel=2,  # PP 度数
)
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **TP（单机多卡）** | 通信延迟低（NVLink）、实现简单 | 显存节省有限（受单机卡数限制）、每层需通信 | 单机能放下的中大模型（7B-70B） |
| **PP（单机/多机）** | 显存节省大（按层切分）、跨层通信少 | Bubble 开销、延迟增加、调度复杂 | 超大模型（70B+）或多机场景 |
| **TP + PP 混合** | 兼顾显存与通信 | 配置复杂、调优难度高 | 超大模型多机部署（如 GPT-4） |
| **EP（MoE 专用）** | 专家并行、显存节省显著 | 仅适用于 MoE 架构、路由复杂 | MoE 模型（Mixtral, DeepSeekMoE） |
| **单卡量化** | 无通信开销、部署简单 | 精度损失、量化实现复杂 | 小模型（7B 以下）或显存充足场景 |

**TP vs PP 性能对比**（72B 模型，8 卡 A100）：

| 指标 | TP-8 | PP-8 | TP-4+PP-2 |
|------|------|------|-----------|
| **显存占用** | 高（权重切分少） | 低（层切分） | 中 |
| **通信延迟** | 高（每层 all-reduce） | 低（跨层通信） | 中 |
| **吞吐（batch=32）** | 1200 tokens/s | 900 tokens/s | 1100 tokens/s |
| **延迟（batch=1）** | 50ms | 80ms | 60ms |
| **Bubble Ratio** | 0% | 21%（4 micro-batch） | 10% |

## 高频追问（至少 5 个）

### 1. Q: TP 和 PP 各自的瓶颈是什么？什么时候选哪个？

**A**:
- **TP 瓶颈**：
  1. 每层需要 all-reduce，通信频率高
  2. decode 阶段通信占比高（30-50%）
  3. 受单机卡数限制（通常 8 卡）
  
- **PP 瓶颈**：
  1. 流水线 bubble（GPU 空闲）
  2. 延迟增加（需等待上游计算）
  3. micro-batch 调度复杂

- **选择策略**：
  - 单机能放下、延迟敏感：优先 TP
  - 超大模型、吞吐优先：PP 或 TP+PP 混合
  - decode 场景、batch 小：尽量减少 TP 度数

### 2. Q: KV cache 在 TP 下怎么存？GQA/MQA 省的是什么？

**A**:
- **TP 下 KV cache 存储**：
  ```python
  # MHA：每卡存储部分头
  KV_cache_local = [batch, seq, 2, heads_per_gpu, head_dim]
  # 无需跨卡同步，本地计算
  
  # GQA/MQA：K/V 按组/单份存储
  # 需要广播到组内/所有头
  ```

- **GQA/MQA 省的是什么**：
  1. **显存**：KV cache 降低 num_heads/num_groups 倍
     - MHA: `batch × seq × 2 × num_heads × head_dim`
     - MQA: `batch × seq × 2 × 1 × head_dim`
     - GQA: `batch × seq × 2 × num_groups × head_dim`
  2. **带宽**：decode 阶段访存量大幅降低
  3. **代价**：精度轻微下降（MQA 2-5%，GQA 1-2%）

### 3. Q: 为什么通信会拖慢 decode？怎么优化？

**A**:
- **原因**：
  - decode 计算量极小（batch×1×hidden_size）
  - 通信时间不变（all-reduce 开销固定）
  - 通信占比从 5%（prefill）升至 50%（decode）

- **优化方法**：
  1. **减少 TP 度数**：单卡能放下就不并行
  2. **计算通信重叠**：异步 all-reduce
  3. **使用高速互联**：NVLink（>300GB/s） vs PCIe（32GB/s）
  4. **KV cache 优化**：GQA/MQA 降低访存
  5. **Batch 策略**：增大 batch 可摊薄通信开销

### 4. Q: all-reduce 和 all-gather 在推理中常见在哪？

**A**:
- **All-Reduce（求和后广播）**：
  1. **TP Attention Output**：每张卡计算部分头，合并结果
     ```python
     O_local = Attn_local @ W_O_local
     O = all_reduce(O_local)  # 合并各卡结果
     ```
  2. **TP FFN Output**：行切分时合并
     ```python
     FFN_local = X_local @ W_local
     FFN_out = all_reduce(FFN_local)
     ```

- **All-Gather（收集后广播）**：
  1. **TP QKV 投影后**：列切分后收集完整输出
     ```python
     Q_local = X @ W_Q_local
     Q = all_gather(Q_local)  # 收集所有头的 Q
     ```
  2. **PP 层间传递**：收集各 micro-batch 的输出

- **通信量对比**：
  - All-Reduce: 2(n-1)/n × data_size
  - All-Gather: (n-1)/n × data_size

### 5. Q: 多机推理为什么更容易抖动？怎么缓解？

**A**:
- **抖动原因**：
  1. **网络延迟不稳定**：InfiniBand 拥塞、路由跳变
  2. **时钟不同步**：各机器时钟偏差影响同步
  3. **负载不均**：各机器处理速度差异
  4. **网络拓扑差异**：不同机器的网络跳数不同

- **缓解方法**：
  1. **网络优化**：
     - 使用专用网络（InfiniBand vs 以太网）
     - 网络拓扑感知调度（减少跨交换机通信）
     - 预留带宽（避免拥塞）
  2. **算法优化**：
     - 减少同步点（异步算法）
     - 容错机制（超时重试）
     - 负载均衡（动态调整）
  3. **监控告警**：
     - 实时监控通信延迟
     - 设置抖动阈值告警
     - 自动降级策略

### 6. Q: 多卡推理 cache 怎么切？不同切分方式的影响？

**A**:
- **切分方式**：
  1. **按头切分（TP 默认）**：
     - 每卡存储部分头的 KV
     - 无需通信，本地计算
     - 显存节省：1/tp_degree
  
  2. **按序列切分（不推荐）**：
     - 每卡存储部分 token 的 KV
     - 需要频繁 all-to-all 通信
     - 实现复杂，效率低

  3. **按层切分（PP）**：
     - 每卡存储完整层的 KV
     - 跨层通信少
     - 显存节省显著

- **GQA/MQA 特殊处理**：
  ```python
  # MQA: K/V 只有一份
  # 方案1：复制到每张卡（增加显存）
  # 方案2：集中存储在某卡，广播（增加通信）
  
  # GQA: K/V 按组存储
  # 组内广播，组间独立
  ```

### 7. Q: 如何做性能 profiling？关键指标有哪些？

**A**:
- **关键指标**：
  1. **Throughput**：tokens/s 或 requests/s
  2. **Latency**：TTFT（Time To First Token）、TPOT（Time Per Output Token）
  3. **GPU Utilization**：计算/访存/通信占比
  4. **Memory**：显存峰值、KV cache 占比
  5. **Communication**：all-reduce/all-gather 时间、带宽利用率

- **Profiling 工具**：
  ```python
  # PyTorch Profiler
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA],
      record_shapes=True,
      profile_memory=True,
  ) as prof:
      output = model.generate(input_ids)
  
  print(prof.key_averages().table(sort_by="cuda_time_total"))
  
  # Nsight Systems（深度 profiling）
  # nsys profile --stats=true python inference.py
  ```

- **瓶颈定位流程**：
  1. 计算占比高 → 考虑算子优化、量化
  2. 访存占比高 → 优化 KV cache、使用 FlashAttention
  3. 通信占比高 → 减少 TP 度数、优化网络

## 常见错误（至少 3 个）

### 1. **错误：认为 TP 能无限扩展**

**正确理解**：
- TP 受单机卡数限制（通常 8 卡）
- 跨机 TP 通信开销剧增（PCIe/NVLink vs InfiniBand）
- decode 阶段 TP 度数越大，通信占比越高

**最佳实践**：
```python
# 单机 8 卡：TP-8 可行
# 单机 16 卡：TP-16 可行（如果 NVLink 全连接）
# 多机：避免纯 TP，改用 TP+PP 或 PP
```

### 2. **错误：PP 的 bubble 可以完全消除**

**正确理解**：
- Bubble 来源于流水线依赖，无法完全消除
- micro-batch 数量增加可降低 bubble 比例，但增加显存和延迟
- 实际 bubble ratio 通常 10-30%

**计算公式**：
```python
# 理论最小 bubble（GPipe）
bubble_ratio = (num_stages - 1) / (num_stages * num_micro_batches)

# 实际优化策略
# 1. 增加micro-batch 数量（显存允许）
# 2. 使用 1F1B 调度（降低峰值显存）
# 3. Interleaved PP（交错流水线）
```

### 3. **错误：GQA/MQA 只省显存，不影响性能**

**正确理解**：
- GQA/MQA 省显存，更关键的是省**访存带宽**
- decode 阶段是**带宽瓶颈**而非计算瓶颈
- KV cache 访存量降低 → decode 加速显著

**性能对比**（LLaMA2-70B，batch=16，seq=1024）：

| 模式 | KV Cache 显存 | Decode Throughput |
|------|---------------|-------------------|
| MHA | 28 GB | 800 tokens/s |
| GQA-8 | 7 GB | 1200 tokens/s (+50%) |
| MQA | 3.5 GB | 1400 tokens/s (+75%) |

### 4. **错误：多机推理只需配置 tensor_parallel_size**

**正确做法**：
```python
# 错误：只配置 TP
llm = LLM(model="model", tensor_parallel_size=16)

# 正确：需要初始化分布式环境
import torch.distributed as dist
dist.init_process_group(backend='nccl')

llm = LLM(
    model="model",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,  # 多机推荐 TP+PP
)

# 还需要配置：
# 1. hostfile（节点列表）
# 2. SSH 免密登录
# 3. 网络拓扑优化
# 4. NCCL 环境变量（IB/GID）
```

## 反问面试官的问题

### 1. 技术深度类
- "在你们的生产环境中，多卡推理的主要瓶颈是在通信、显存还是计算？有没有做过量化的 profiling？"
- "对于长上下文场景（如 128K context），KV cache 的显存管理有哪些实践经验？是否考虑过 PagedAttention 或 KV cache 量化？"

### 2. 业务场景类
- "你们的推理延迟目标是多少？TP99 和 TP50 的差距大吗？怎么应对长尾延迟？"
- "多机部署时，网络抖动对服务稳定性的影响有多大？有哪些容错或降级策略？"

## 自测题

### 口述（能流畅讲清楚的知识点）
1. **TP 和 PP 的核心差异**：从切分方式、通信模式、适用场景三个角度对比
2. **为什么 decode 阶段通信占比高**：从计算量、通信开销、batch 大小三个因素分析
3. **GQA/MQA 的收益与代价**：显存、带宽、精度、实现复杂度

### 手写（5 分钟能写出的代码/公式）
1. **KV cache 显存估算公式**（TP 模式）
```python
def estimate_kv_cache_memory_tp(batch_size, seq_len, num_layers, 
                                 num_heads, head_dim, tp_degree, dtype='fp16'):
    bytes_per_element = 2 if dtype == 'fp16' else 4
    heads_per_gpu = num_heads // tp_degree
    
    memory_per_layer = batch_size * seq_len * 2 * heads_per_gpu * head_dim * bytes_per_element
    total_memory = num_layers * memory_per_layer
    
    return total_memory / (1024**3)  # GB
```

2. **All-Reduce 通信量计算**
```python
def all_reduce_communication(data_size_bytes, world_size, bandwidth_gb_per_sec):
    # Ring AllReduce: 2(n-1)/n × data_size
    comm_bytes = 2 * (world_size - 1) / world_size * data_size_bytes
    comm_time_sec = comm_bytes / (bandwidth_gb_per_sec * 1024**3)
    return comm_time_sec * 1000  # ms
```

3. **PP Bubble Ratio 计算**
```python
def calculate_bubble_ratio(num_stages, num_micro_batches):
    # (p-1) / (p * num_stages)
    return (num_stages - 1) / (num_stages * num_micro_batches)

# 例子：8 卡 PP，4 个 micro-batch
bubble = calculate_bubble_ratio(8, 4)  # 21.875%
```

## 标签

#推理 #TP #PP #工程 #KV_cache #GQA #MQA #all_reduce #通信 #多卡 #性能优化 #vLLM #TensorRT-LLM #字节 #阿里 #腾讯 #美团 #百度

## 相关文档

- [[02-KV Cache核心]] - KV cache 存储机制、显存估算、优化策略
- [[01-Attention机制]] - MHA/MQA/GQA 原理与实现
- [[05-PagedAttention]] - vLLM 的 PagedAttention 机制
- [[../07-分布式训练ZeRO/01-3D并行总览]] - 训练侧的 TP/PP/DP 原理
- [[../10-FlashAttention/01-FlashAttention原理]] - FlashAttention 的 IO 优化
