# 性能 Profiling 与线上故障

## 一句话结论
线上推理性能问题定位遵循"先指标后算子"原则，关键指标监控 → 定位瓶颈阶段（prefill/decode/kv_cache） → 工具剖析（nsys/nvprof）→ 根因修复，故障排查要建立"现象-指标-根因"映射表，避免盲改。

## 核心定义/公式

### 关键性能指标

#### 1. **吞吐指标**

```python
# tokens/s（总吞吐）
throughput = total_tokens_generated / total_time

# TTFT（Time To First Token，首字延迟）
ttft = time_to_first_token - request_arrival_time
# prefill 阶段耗时，影响用户感知延迟

# TPOT（Time Per Output Token，生成速度）
tpot = total_generation_time / (num_tokens - 1)  # 减去首字
# decode 阶段平均每 token 耗时，影响生成流畅度
```

#### 2. **显存指标**

```python
# 显存峰值（GPU Memory Peak）
memory_peak = max(memory_allocated_during_inference)

# KV Cache 显存估算
kv_cache_memory = (
    2 *  # K + V
    num_layers *
    batch_size *
    seq_len *
    num_heads *
    head_dim *
    dtype_size  # fp16=2, bf16=2, int8=1
)

# Cache Hit Rate（前缀缓存命中率）
cache_hit_rate = num_requests_with_cache_hit / total_requests
```

#### 3. **调度指标**

```python
# Queue Latency（排队延迟）
queue_latency = request_start_time - request_arrival_time

# Batch Size Distribution（批大小分布）
avg_batch_size = total_requests_processed / total_batches

# GPU Utilization（GPU 利用率）
gpu_util = gpu_active_time / total_time
```

### 性能分解公式

```python
# 一轮生成的总延迟
total_latency = queue_latency + ttft + (num_tokens - 1) * tpot

# 吞吐分解
throughput = batch_size / (prefill_time + decode_time_per_token * avg_output_len)

# 瓶颈定位
if ttft > threshold:
    bottleneck = "prefill"  # prefill 太慢
elif tpot > threshold:
    bottleneck = "decode"   # decode 太慢或 KV cache 读写慢
elif queue_latency > threshold:
    bottleneck = "scheduling"  # 调度问题
```

### 代码示例：性能监控实现

```python
import time
import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InferenceMetrics:
    """推理性能指标收集器"""
    request_id: str
    arrival_time: float
    start_time: float
    first_token_time: float
    end_time: float
    num_tokens: int
    batch_size: int
    
    @property
    def ttft(self) -> float:
        """首字延迟（秒）"""
        return self.first_token_time - self.start_time
    
    @property
    def tpot(self) -> float:
        """平均每 token 耗时（秒）"""
        if self.num_tokens <= 1:
            return 0.0
        return (self.end_time - self.first_token_time) / (self.num_tokens - 1)
    
    @property
    def queue_latency(self) -> float:
        """排队延迟（秒）"""
        return self.start_time - self.arrival_time
    
    @property
    def total_latency(self) -> float:
        """总延迟（秒）"""
        return self.end_time - self.arrival_time


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: List[InferenceMetrics] = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(
        self,
        request_id: str,
        arrival_time: float,
        start_time: float,
        first_token_time: float,
        end_time: float,
        num_tokens: int,
        batch_size: int
    ):
        """记录单次请求的指标"""
        metric = InferenceMetrics(
            request_id=request_id,
            arrival_time=arrival_time,
            start_time=start_time,
            first_token_time=first_token_time,
            end_time=end_time,
            num_tokens=num_tokens,
            batch_size=batch_size
        )
        self.metrics.append(metric)
    
    def record_cache_hit(self, hit: bool):
        """记录缓存命中"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_summary(self) -> dict:
        """获取性能摘要"""
        if not self.metrics:
            return {}
        
        ttfts = [m.ttft for m in self.metrics]
        tpots = [m.tpot for m in self.metrics if m.tpot > 0]
        queue_latencies = [m.queue_latency for m in self.metrics]
        
        total_tokens = sum(m.num_tokens for m in self.metrics)
        total_time = self.metrics[-1].end_time - self.metrics[0].arrival_time
        
        return {
            "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
            "avg_ttft_ms": sum(ttfts) / len(ttfts) * 1000,
            "p99_ttft_ms": sorted(ttfts)[int(len(ttfts) * 0.99)] * 1000,
            "avg_tpot_ms": sum(tpots) / len(tpots) * 1000 if tpots else 0,
            "avg_queue_latency_ms": sum(queue_latencies) / len(queue_latencies) * 1000,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses),
            "total_requests": len(self.metrics)
        }


# 使用示例
monitor = PerformanceMonitor()

def inference_with_monitoring(prompt: str, model):
    """带监控的推理"""
    arrival_time = time.time()
    
    # 加入批次，等待调度
    batch = scheduler.wait_for_batch(prompt)
    start_time = time.time()
    
    # Prefill + Decode
    output = model.generate(prompt)
    first_token_time = time.time()
    
    # 流式生成
    for token in output:
        yield token
    
    end_time = time.time()
    
    # 记录指标
    monitor.record_request(
        request_id=prompt[:10],
        arrival_time=arrival_time,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        num_tokens=len(output),
        batch_size=len(batch)
    )
```

## 为什么（2-3 个因果链）

### 1. 为什么 decode 比 prefill 更慢？

**现象**：长文本生成时，decode 阶段每个 token 耗时反而比 prefill 更长

**因果链**：
```
Prefill 阶段：
  所有 token 并行计算 QK^T
    → 矩阵运算，GPU 利用率高
    → 算力密集型，受 GPU 算力限制
    → 时间 = O(seq_len^2 * d_model / GPU_FLOPS)

Decode 阶段：
  每次只生成 1 个 token
    → 矩阵乘法，访问整个 KV cache
    → 带宽密集型，受 HBM 带宽限制
    → 时间 = O(seq_len * d_model / HBM_BW)

随着 seq_len 增长：
  Prefill 时间 = O(seq_len^2)  # 但算力并行
  Decode 时间 = O(seq_len)     # 但要读整个 KV cache
  
当 seq_len 很长时：
  KV cache 很大
    → Decode 每次都要读整个 cache
    → HBM 带宽成为瓶颈
    → Decode 每步反而更慢
```

**关键洞察**：
- **Prefill**：算力密集型（compute-bound），GPU 算力瓶颈
- **Decode**：带宽密集型（memory-bound），HBM 带宽瓶颈
- 长上下文下，decode 的带宽瓶颈更明显

**公式解释**：
```python
# Prefill 时间（近似）
prefill_time = (seq_len^2 * d_model) / (GPU_FLOPS * efficiency)

# Decode 每步时间（近似）
decode_time_per_token = (2 * seq_len * d_model) / HBM_BW
# 2x 因为要读 K 和 V

# 当 seq_len 增大时
# prefill_time 增长快，但可以并行
# decode_time_per_token 线性增长，无法并行
```

### 2. 为什么 batch 一大就 OOM？

**现象**：小 batch 正常，batch 稍微增大就 OOM

**因果链**：
```
KV Cache 显存占用公式：
  memory = 2 * num_layers * batch_size * seq_len * num_heads * head_dim * dtype_size

当 batch_size 增大：
  KV cache 显存线性增长
    → 例如：batch_size 从 8 → 16
    → KV cache 翻倍
    → 加上激活、模型权重，总显存超限
    → OOM

关键点：
  1. 模型权重：固定，不随 batch 变化
  2. 激活值：随 batch 线性增长
  3. KV cache：随 batch * seq_len 增长（最大头）

例如（7B 模型，fp16）：
  模型权重：14 GB
  单个请求 KV cache（2048 tokens）：约 2 GB
  8 个请求：14 + 8 * 2 = 30 GB（A100 40GB 够用）
  16 个请求：14 + 16 * 2 = 46 GB（OOM！）
```

**显存分配示意**：
```python
# 显存占用分解
total_memory = (
    model_weights +           # 固定
    optimizer_states +        # 推理时为 0
    activations +             # 随 batch 增长
    kv_cache +                # 随 batch * seq_len 增长
    fragmentation +           # 碎片，不可控
    workspace                # CUDA 工作空间
)

# 为什么看着还够却 OOM？
# 1. 碎片化：显存不是连续分配，实际可用 < 理论可用
# 2. 峰值：某些中间步骤需要更多临时显存
# 3. 框架开销：PyTorch、vLLM 内部有额外占用
# 4. 其他进程：GPU 上可能有其他进程占用
```

### 3. 为什么会出现"偶发慢"？

**现象**：大部分请求正常，偶尔出现超长延迟

**因果链**：
```
偶发慢的常见原因：

1. GC（垃圾回收）暂停：
   Python GC 触发
     → 所有线程暂停
     → 请求延迟突增
   解决：定期主动 GC、减少对象创建

2. 长上下文请求：
   遇到超长 prompt
     → prefill 时间暴涨（O(seq_len^2)）
     → 占用整个 batch
     → 其他请求等待
   解决：分离长短请求、限制最大长度

3. KV cache 驱逐：
   显存不足，驱逐旧 cache
     → 重新计算 prefill
     → 延迟突增
   解决：增加显存、优化 cache 策略

4. 多卡通信抖动：
   TP/PP 跨卡通信
     → 网络抖动或负载不均
     → all-reduce 延迟突增
   解决：优化通信拓扑、监控网络

5. 调度饥饿：
   长请求占满 batch
     → 短请求一直等待
     → queue latency 突增
   解决：优先级调度、时间片轮转
```

**偶发慢定位方法**：
```python
# 1. 记录每次请求的详细指标
detailed_metrics = {
    "request_id": "xxx",
    "prompt_length": 512,
    "output_length": 128,
    "queue_latency": 0.01,
    "ttft": 0.5,
    "tpot": 0.02,
    "batch_size": 8,
    "cache_hit": False,
    "gc_triggered": True,  # 记录是否触发 GC
    "cross_node": False    # 是否跨机
}

# 2. 按延迟排序，找异常点
sorted_metrics = sorted(all_metrics, key=lambda x: x.total_latency, reverse=True)

# 3. 分析 Top 10 慢请求的共同特征
for metric in sorted_metrics[:10]:
    print(f"Request {metric.request_id}: {metric.total_latency:.2f}s")
    print(f"  Prompt len: {metric.prompt_length}")
    print(f"  TTFT: {metric.ttft:.2f}s, TPOT: {metric.tpot:.3f}s")
    print(f"  Queue: {metric.queue_latency:.2f}s")
```

## 怎么做（可落地步骤）

### 一、性能 Profiling 标准流程

#### 步骤 1：建立监控体系

```python
# 关键指标监控配置
monitoring_config = {
    "metrics": [
        "throughput_tokens_per_sec",
        "avg_ttft_ms",
        "p99_ttft_ms",
        "avg_tpot_ms",
        "avg_queue_latency_ms",
        "cache_hit_rate",
        "gpu_memory_used_gb",
        "gpu_utilization"
    ],
    "collection_interval": 1.0,  # 秒
    "aggregation_window": 60,    # 聚合窗口（秒）
    "alert_thresholds": {
        "p99_ttft_ms": 500,       # P99 TTFT > 500ms 告警
        "avg_tpot_ms": 100,       # 平均 TPOT > 100ms 告警
        "gpu_memory_used_gb": 75, # 显存使用 > 75GB 告警
        "cache_hit_rate": 0.3     # 缓存命中率 < 30% 告警
    }
}
```

#### 步骤 2：定位瓶颈阶段

```python
def diagnose_bottleneck(metrics: dict) -> str:
    """诊断性能瓶颈阶段"""
    
    # 1. 检查排队延迟
    if metrics["avg_queue_latency_ms"] > 100:
        return "scheduling: queue too long, check batch scheduler"
    
    # 2. 检查 prefill
    if metrics["p99_ttft_ms"] > 500:
        return "prefill: TTFT too high, check prompt length or batching"
    
    # 3. 检查 decode
    if metrics["avg_tpot_ms"] > 100:
        return "decode: TPOT too high, check KV cache or GPU memory BW"
    
    # 4. 检查 GPU 利用率
    if metrics["gpu_utilization"] < 0.5:
        return "underutilization: GPU not fully used, check batch size or IO"
    
    # 5. 检查显存
    if metrics["gpu_memory_used_gb"] > 75:
        return "memory: near OOM, check KV cache size or reduce batch"
    
    return "no obvious bottleneck"
```

#### 步骤 3：使用 Profiling 工具

```bash
# 1. NVIDIA Nsight Systems（nsys）
# 记录时间线，定位耗时分布
nsys profile -o inference_profile \
  --trace=cuda,nvtx,osrt \
  python inference.py

# 分析结果
nsys-ui inference_profile.nsys-rep

# 2. NVIDIA Nsight Compute（ncu）
# 分析单个 kernel 的性能
ncu --set full -o kernel_profile \
  python inference.py

# 3. PyTorch Profiler
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model.generate(prompt)

# 查看 CPU/CUDA 时间分布
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### 步骤 4：算子级 Profiling

```python
import torch
import time

def profile_attention_layer(model, input_ids, num_iterations=10):
    """剖析 attention 层性能"""
    
    # 预热
    with torch.no_grad():
        _ = model(input_ids)
    
    # 记录每个阶段的时间
    times = {
        "embedding": [],
        "attention": [],
        "ffn": [],
        "lm_head": []
    }
    
    with torch.no_grad():
        for _ in range(num_iterations):
            # Embedding
            torch.cuda.synchronize()
            start = time.time()
            hidden_states = model.model.embed_tokens(input_ids)
            torch.cuda.synchronize()
            times["embedding"].append(time.time() - start)
            
            # Attention layers
            for i, layer in enumerate(model.model.layers):
                torch.cuda.synchronize()
                start = time.time()
                hidden_states = layer(hidden_states)
                torch.cuda.synchronize()
                times["attention"].append(time.time() - start)
            
            # LM Head
            torch.cuda.synchronize()
            start = time.time()
            logits = model.lm_head(hidden_states)
            torch.cuda.synchronize()
            times["lm_head"].append(time.time() - start)
    
    # 汇总
    for stage, t_list in times.items():
        avg_time = sum(t_list) / len(t_list) * 1000
        print(f"{stage}: {avg_time:.2f} ms")
```

### 二、故障排查剧本

#### 故障 1：吞吐突然掉一半怎么查？

```python
# 排查脚本
def diagnose_throughput_drop(
    current_metrics: dict,
    baseline_metrics: dict
) -> List[str]:
    """诊断吞吐下降原因"""
    
    issues = []
    
    # 1. 检查 batch size
    if current_metrics["avg_batch_size"] < baseline_metrics["avg_batch_size"] * 0.8:
        issues.append(
            f"Batch size dropped: {current_metrics['avg_batch_size']:.1f} vs "
            f"{baseline_metrics['avg_batch_size']:.1f}\n"
            "Possible causes: request rate decreased, scheduler issue, "
            "or max_concurrent_requests limit"
        )
    
    # 2. 检查 TTFT（prefill 变慢）
    if current_metrics["avg_ttft_ms"] > baseline_metrics["avg_ttft_ms"] * 1.5:
        issues.append(
            f"TTFT increased: {current_metrics['avg_ttft_ms']:.1f}ms vs "
            f"{baseline_metrics['avg_ttft_ms']:.1f}ms\n"
            "Possible causes: longer prompts, cache miss, prefill kernel issue"
        )
    
    # 3. 检查 TPOT（decode 变慢）
    if current_metrics["avg_tpot_ms"] > baseline_metrics["avg_tpot_ms"] * 1.5:
        issues.append(
            f"TPOT increased: {current_metrics['avg_tpot_ms']:.1f}ms vs "
            f"{baseline_metrics['avg_tpot_ms']:.1f}ms\n"
            "Possible causes: KV cache growth, memory bandwidth saturation, "
            "longer output sequences"
        )
    
    # 4. 检查 cache hit rate
    if current_metrics["cache_hit_rate"] < baseline_metrics["cache_hit_rate"] * 0.5:
        issues.append(
            f"Cache hit rate dropped: {current_metrics['cache_hit_rate']:.2f} vs "
            f"{baseline_metrics['cache_hit_rate']:.2f}\n"
            "Possible causes: prompt pattern change, cache eviction, cache config issue"
        )
    
    # 5. 检查 GPU 利用率
    if current_metrics["gpu_utilization"] < baseline_metrics["gpu_utilization"] * 0.8:
        issues.append(
            f"GPU utilization dropped: {current_metrics['gpu_utilization']:.2f} vs "
            f"{baseline_metrics['gpu_utilization']:.2f}\n"
            "Possible causes: CPU bottleneck, IO bottleneck, communication overhead"
        )
    
    # 6. 检查显存使用
    if current_metrics["gpu_memory_used_gb"] > baseline_metrics["gpu_memory_used_gb"] * 1.2:
        issues.append(
            f"GPU memory increased: {current_metrics['gpu_memory_used_gb']:.1f}GB vs "
            f"{baseline_metrics['gpu_memory_used_gb']:.1f}GB\n"
            "Possible causes: longer sequences, memory leak, fragmentation"
        )
    
    if not issues:
        issues.append("No obvious bottleneck found, check:")
        issues.append("  1. Request pattern change (longer prompts/outputs)")
        issues.append("  2. Model change (new checkpoint, quantization)")
        issues.append("  3. Hardware change (GPU frequency, thermal throttling)")
    
    return issues
```

**排查步骤**：
```
1. 对比指标：当前 vs 基线
   - batch_size 是否下降？
   - TTFT/TPOT 是否增加？
   - cache_hit_rate 是否下降？
   - GPU utilization 是否下降？

2. 检查请求模式
   - prompt 长度分布是否变化？
   - output 长度分布是否变化？
   - 请求频率是否下降？

3. 检查系统状态
   - 是否有新进程占用 GPU？
   - GPU 温度、频率是否正常？
   - 网络是否有抖动（多卡场景）？

4. 检查配置变更
   - 是否有热更新？
   - batch size 限制是否变化？
   - cache 配置是否变化？

5. 使用 profiling 工具
   - nsys 记录时间线
   - 找到耗时突增的阶段
   - 定位到具体算子
```

#### 故障 2：OOM 但显存看着还够

```python
def diagnose_oom(memory_allocated: float, memory_reserved: float, total_memory: float):
    """诊断 OOM 原因"""
    
    print("=== Memory Diagnostics ===")
    print(f"Memory allocated: {memory_allocated:.2f} GB")
    print(f"Memory reserved: {memory_reserved:.2f} GB")
    print(f"Total memory: {total_memory:.2f} GB")
    print(f"Fragmentation: {memory_reserved - memory_allocated:.2f} GB")
    
    # 1. 碎片化严重
    fragmentation_ratio = (memory_reserved - memory_allocated) / memory_reserved
    if fragmentation_ratio > 0.3:
        print("⚠️  High fragmentation (>30%)")
        print("   Solution: Use paged attention, reduce block size, or restart server")
    
    # 2. 峰值显存
    print("\n=== Peak Memory Analysis ===")
    print("Possible causes:")
    print("  1. Temporary buffers during attention computation")
    print("  2. Activation checkpointing not enabled")
    print("  3. KV cache allocation burst")
    print("  4. Cross-layer communication buffers (TP/PP)")
    
    # 3. 显存泄漏
    print("\n=== Memory Leak Check ===")
    print("Check if memory grows over time:")
    print("  1. Monitor memory_allocated over multiple requests")
    print("  2. Check for dangling references to tensors")
    print("  3. Verify KV cache is properly freed after request")
    
    # 4. 显存分配策略
    print("\n=== Memory Allocation Strategy ===")
    print("Current PyTorch memory allocator:")
    print("  - PyTorch uses caching allocator")
    print("  - Reserved != Allocated (fragmentation)")
    print("  - Large contiguous blocks are kept for reuse")
    print("\nSolutions:")
    print("  1. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    print("  2. Use vLLM paged attention (eliminates fragmentation)")
    print("  3. Reduce max_batch_size or max_seq_len")
```

**OOM 常见原因**：
```
1. 碎片化（Fragmentation）
   现象：reserved > allocated，但分配失败
   原因：显存碎片化，无法找到足够大的连续块
   解决：
     - vLLM paged attention（消除碎片）
     - PYTORCH_CUDA_ALLOC_CONF 调整
     - 定期重启服务

2. 峰值显存
   现象：某些中间步骤需要额外显存
   原因：
     - Attention 计算的临时 buffer
     - TP/PP 的通信 buffer
     - Beam search 的候选序列
   解决：
     - Activation checkpointing
     - 限制 beam size
     - 优化 attention kernel

3. 显存泄漏
   现象：显存持续增长，不释放
   原因：
     - Python 对象引用未释放
     - KV cache 未正确释放
     - 日志/监控对象累积
   解决：
     - 定位泄漏点（torch.cuda.memory_summary()）
     - 显式释放（del + torch.cuda.empty_cache()）
     - 定期重启

4. 框架开销
   现象：框架本身占用显存
   原因：
     - PyTorch CUDA context
     - vLLM 内部管理结构
     - NCCL 通信 buffer
   解决：
     - 预估时留出 5-10% 余量
     - 监控框架开销
```

#### 故障 3：偶发慢定位

```python
def diagnose_occasional_slow(
    slow_requests: List[dict],
    normal_requests: List[dict]
) -> str:
    """诊断偶发慢请求"""
    
    # 统计特征
    def get_statistics(requests: List[dict]) -> dict:
        prompt_lens = [r["prompt_length"] for r in requests]
        output_lens = [r["output_length"] for r in requests]
        ttfts = [r["ttft"] for r in requests]
        tpots = [r["tpot"] for r in requests]
        queue_latencies = [r["queue_latency"] for r in requests]
        
        return {
            "avg_prompt_len": sum(prompt_lens) / len(prompt_lens),
            "avg_output_len": sum(output_lens) / len(output_lens),
            "avg_ttft": sum(ttfts) / len(ttfts),
            "avg_tpot": sum(tpots) / len(tpots),
            "avg_queue_latency": sum(queue_latencies) / len(queue_latencies),
        }
    
    slow_stats = get_statistics(slow_requests)
    normal_stats = get_statistics(normal_requests)
    
    # 对比分析
    print("=== Slow vs Normal Requests ===")
    
    if slow_stats["avg_prompt_len"] > normal_stats["avg_prompt_len"] * 2:
        return "Slow requests have much longer prompts (prefill bottleneck)"
    
    if slow_stats["avg_output_len"] > normal_stats["avg_output_len"] * 2:
        return "Slow requests have much longer outputs (decode bottleneck)"
    
    if slow_stats["avg_queue_latency"] > normal_stats["avg_queue_latency"] * 5:
        return "Slow requests have long queue latency (scheduling issue)"
    
    if slow_stats["avg_ttft"] > normal_stats["avg_ttft"] * 2:
        return "Slow requests have high TTFT (prefill or cache issue)"
    
    if slow_stats["avg_tpot"] > normal_stats["avg_tpot"] * 2:
        return "Slow requests have high TPOT (decode bottleneck)"
    
    # 检查其他特征
    gc_count_slow = sum(r.get("gc_triggered", False) for r in slow_requests)
    gc_count_normal = sum(r.get("gc_triggered", False) for r in normal_requests)
    
    if gc_count_slow > gc_count_normal * 5:
        return "Slow requests often triggered GC (Python GC issue)"
    
    return "No clear pattern, check:\n  1. GPU frequency throttling\n  2. Network jitter (multi-node)\n  3. Concurrent heavy requests"
```

#### 故障 4：热更新后效果变差

```python
def diagnose_hot_reload_issue(
    old_model_path: str,
    new_model_path: str,
    test_prompts: List[str]
) -> List[str]:
    """诊断热更新后效果变差"""
    
    issues = []
    
    # 1. 检查模型配置
    print("=== Checking Model Config ===")
    old_config = load_config(old_model_path)
    new_config = load_config(new_model_path)
    
    if old_config.vocab_size != new_config.vocab_size:
        issues.append(f"Vocab size changed: {old_config.vocab_size} -> {new_config.vocab_size}")
    
    if old_config.hidden_size != new_config.hidden_size:
        issues.append(f"Hidden size changed: {old_config.hidden_size} -> {new_config.hidden_size}")
    
    # 2. 检查 tokenizer
    print("\n=== Checking Tokenizer ===")
    old_tokenizer = load_tokenizer(old_model_path)
    new_tokenizer = load_tokenizer(new_model_path)
    
    for prompt in test_prompts[:5]:
        old_tokens = old_tokenizer.encode(prompt)
        new_tokens = new_tokenizer.encode(prompt)
        if old_tokens != new_tokens:
            issues.append(f"Tokenizer mismatch for: {prompt[:30]}...")
            break
    
    # 3. 检查 KV cache 兼容性
    print("\n=== Checking KV Cache Compatibility ===")
    issues.append("If using KV cache, ensure cache is cleared after hot reload")
    
    # 4. 检查动态库
    print("\n=== Checking Shared Libraries ===")
    issues.append("If using custom kernels (FlashAttention), ensure they are reloaded")
    
    # 5. 测试对比
    print("\n=== Testing Output Quality ===")
    old_model = load_model(old_model_path)
    new_model = load_model(new_model_path)
    
    for prompt in test_prompts[:3]:
        old_output = old_model.generate(prompt, max_tokens=50)
        new_output = new_model.generate(prompt, max_tokens=50)
        
        if old_output != new_output:
            print(f"Output difference for: {prompt[:30]}")
            print(f"  Old: {old_output[:50]}")
            print(f"  New: {new_output[:50]}")
    
    return issues
```

**热更新问题常见原因**：
```
1. Tokenizer 不匹配
   现象：输出乱码或格式错误
   原因：新模型用不同 tokenizer
   解决：热更新时同步 tokenizer

2. KV cache 未清空
   现象：输出混淆（"串台"）
   原因：旧 cache 被新模型使用
   解决：热更新时清空所有 cache

3. 配置不匹配
   现象：推理错误或效果下降
   原因：hidden_size、vocab_size 等配置变化
   解决：检查配置一致性

4. 动态库未重载
   现象：自定义算子错误
   原因：FlashAttention 等 kernel 未重新加载
   解决：重启进程而非热更新

5. 状态残留
   现象：行为不一致
   原因：旧模型状态（batch norm、统计量）未清除
   解决：完全重新初始化
```

#### 故障 5：输出重复/循环

```python
def diagnose_repetition(output: str, model, tokenizer) -> str:
    """诊断输出重复/循环"""
    
    # 检测重复模式
    tokens = tokenizer.encode(output)
    
    # 1. 检测重复片段
    for pattern_len in [2, 3, 4, 5]:
        patterns = {}
        for i in range(len(tokens) - pattern_len):
            pattern = tuple(tokens[i:i+pattern_len])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        repeated_patterns = {k: v for k, v in patterns.items() if v > 1}
        if repeated_patterns:
            print(f"Found repeated {pattern_len}-gram patterns: {len(repeated_patterns)}")
    
    # 2. 检测循环
    for cycle_len in range(1, min(20, len(tokens) // 2)):
        cycle = tokens[-cycle_len:]
        prev_cycle = tokens[-2*cycle_len:-cycle_len]
        if cycle == prev_cycle:
            print(f"Found exact cycle of length {cycle_len}")
            return "repetition_penalty too low or sampling temperature too low"
    
    # 3. 检查 sampling 参数
    issues = []
    issues.append("Check sampling parameters:")
    issues.append("  - temperature: too low (< 0.5) causes repetition")
    issues.append("  - repetition_penalty: too low (< 1.0) allows repetition")
    issues.append("  - top_k/top_p: too restrictive narrows choices")
    
    # 4. 检查 EOS token
    if tokenizer.eos_token_id not in tokens:
        issues.append("EOS token not generated, model stuck in loop")
    
    return "\n".join(issues)
```

**输出重复常见原因**：
```
1. Sampling 参数不当
   - temperature 太低（< 0.5）：模型过于贪婪
   - repetition_penalty 太低（< 1.0）：不惩罚重复
   - top_k/top_p 太严格：选择空间太小

2. EOS token 问题
   - EOS token 未正确设置
   - 模型未学到何时停止
   - 遇到训练时未见过的模式

3. 模型问题
   - 训练数据有大量重复
   - SFT 数据质量差
   - 过度微调导致模式崩溃

4. Prompt 问题
   - Prompt 引导重复
   - Prompt 与训练数据分布偏差大

解决方案：
  - 提高 repetition_penalty（1.1~1.3）
  - 适当提高 temperature（0.7~0.9）
  - 添加 n-gram blocking
  - 检查 EOS token 设置
  - 过滤训练数据中的重复
```

#### 故障 6：线上"串台"

```python
def diagnose_crosstalk(request_id: str, all_requests: List[dict]) -> str:
    """诊断串台问题（不同请求的输出混淆）"""
    
    # 找到相关请求
    target_request = None
    related_requests = []
    
    for req in all_requests:
        if req["request_id"] == request_id:
            target_request = req
        elif abs(req["start_time"] - target_request["start_time"]) < 1.0:
            related_requests.append(req)
    
    if not target_request:
        return "Request not found"
    
    # 检查输出是否包含其他请求的内容
    issues = []
    
    # 1. 检查 KV cache 混淆
    if target_request["cache_hit"] and target_request["cache_shared"]:
        issues.append("Request used shared KV cache")
        issues.append("  Possible cause: cache not properly isolated")
    
    # 2. 检查 position_id 错误
    if target_request.get("position_id_mismatch", False):
        issues.append("Position ID mismatch detected")
        issues.append("  Possible cause: multi-turn conversation position not reset")
    
    # 3. 检查 batch 处理
    if target_request["batch_size"] > 1:
        issues.append(f"Request processed in batch of size {target_request['batch_size']}")
        issues.append("  Possible cause: batch dimension confusion in custom code")
    
    # 4. 检查多轮对话
    if target_request.get("conversation_turn", 1) > 1:
        issues.append(f"Multi-turn conversation (turn {target_request['conversation_turn']})")
        issues.append("  Possible cause: previous turn's cache not cleared or position ID not updated")
    
    return "\n".join(issues) if issues else "No obvious crosstalk cause found"
```

**串台常见原因**：
```
1. KV cache 混淆
   现象：A 请求的输出包含 B 请求的内容
   原因：
     - Prefix cache 命中后，cache 未隔离
     - Multi-turn 对话时，不同用户的 cache 混淆
   解决：
     - 每个请求独立的 cache block
     - cache key 包含 user_id 或 session_id

2. Position ID 错误
   现象：输出错位或内容混乱
   原因：
     - 多轮对话时 position_id 未正确累加
     - 不同请求共享 position_id 状态
   解决：
     - 每个请求独立的 position_id 起始点
     - 清空或重置 position_id 状态

3. Batch 维度混淆
   现象：batch 中不同请求输出互换
   原因：
     - 自定义代码中 batch 维度索引错误
     - vLLM/TensorRT-LLM 的 batch 维度处理不当
   解决：
     - 严格检查 batch 维度索引
     - 使用框架提供的 batching 机制

4. 热更新后状态残留
   现象：新请求包含旧模型的状态
   原因：
     - 热更新未清空 KV cache
     - 旧请求的 cache 被新请求复用
   解决：
     - 热更新时清空所有 cache
     - 重启服务而非热更新
```

#### 故障 7：多轮对话越聊越飘

```python
def diagnose_conversation_drift(conversation_history: List[dict]) -> str:
    """诊断多轮对话越聊越飘"""
    
    issues = []
    
    # 1. 检查上下文长度
    total_tokens = sum(
        len(tokenizer.encode(turn["content"]))
        for turn in conversation_history
    )
    
    if total_tokens > model_max_length * 0.8:
        issues.append(f"Context length ({total_tokens}) approaching limit ({model_max_length})")
        issues.append("  Solution: Implement sliding window or conversation summary")
    
    # 2. 检查 position_id 累积
    position_ids = [turn.get("position_id", 0) for turn in conversation_history]
    if max(position_ids) > model_max_length * 0.8:
        issues.append("Position ID accumulated too much")
        issues.append("  Solution: Reset position IDs after sliding window")
    
    # 3. 检查 KV cache 质量
    if len(conversation_history) > 10:
        issues.append("Too many turns, early KV cache may be outdated")
        issues.append("  Solution: Implement cache eviction or recomputation")
    
    # 4. 检查话题漂移
    topics = [extract_topic(turn["content"]) for turn in conversation_history]
    if len(set(topics)) > 5:
        issues.append("Conversation topic drifted significantly")
        issues.append("  Solution: Use conversation manager to detect and handle drift")
    
    # 5. 检查指令遗忘
    if "system" in conversation_history[0]:
        system_instruction = conversation_history[0]["content"]
        later_outputs = [turn["content"] for turn in conversation_history[5:]]
        if not any(is_consistent(output, system_instruction) for output in later_outputs):
            issues.append("Model forgot initial system instruction")
            issues.append("  Solution: Periodically repeat system instruction")
    
    return "\n".join(issues) if issues else "No obvious drift cause"
```

**多轮对话飘的原因**：
```
1. 上下文超长
   现象：模型输出不符合早期指令
   原因：
     - 上下文超出窗口，早期内容被截断
     - Position ID 累积超过训练长度
   解决：
     - 滑动窗口：只保留最近 N 轮
     - 对话摘要：定期压缩早期对话
     - 重置 position ID

2. KV cache 质量下降
   现象：早期 token 的 cache 影响当前生成
   原因：
     - 早期对话的 cache 包含过时信息
     - Cache 累积误差
   解决：
     - Cache eviction：移除早期的 cache
     - 定期重算：重新计算重要部分的 cache

3. 指令遗忘
   现象：模型忘记初始 system instruction
   原因：
     - 长对话中，早期指令权重被稀释
     - 注意力机制偏向近期 token
   解决：
     - 周期性重复 system instruction
     - 使用 special token 标记重要指令

4. 话题漂移
   现象：对话偏离原始目标
   原因：
     - 用户问题逐渐偏离
     - 模型跟随对话流，忘记初始目标
   解决：
     - 对话管理器：检测话题漂移
     - 定期提醒原始目标

5. 数据分布问题
   现象：模型在长对话中表现不稳定
   原因：
     - 训练数据中长对话少
     - 多轮对话数据分布偏差
   解决：
     - 增加多轮对话训练数据
     - 使用多轮对话特定的 SFT 数据
```

### 三、关键配置/参数

#### vLLM 性能配置

```python
# vLLM 配置示例
vllm_config = {
    # 模型配置
    "model": "Qwen/Qwen2-7B",
    "tokenizer": "Qwen/Qwen2-7B",
    "dtype": "float16",  # 或 bfloat16
    
    # 批处理配置
    "max_num_batched_tokens": 32768,  # 最大批处理 token 数
    "max_num_seqs": 256,               # 最大并发序列数
    "max_model_len": 8192,             # 最大上下文长度
    
    # KV Cache 配置
    "block_size": 16,                  # Paged attention block 大小
    "gpu_memory_utilization": 0.9,     # GPU 显存利用率目标
    "swap_space": 4,                   # CPU swap 空间（GB）
    
    # 调度配置
    "scheduler_delay_factor": 0.0,     # 调度延迟因子
    "enable_chunked_prefill": True,    # 启用分块 prefill
    
    # Prefix Cache 配置
    "enable_prefix_caching": True,     # 启用前缀缓存
    "max_num_prefix_seqs": 128,        # 最大缓存前缀数
    
    # 并行配置
    "tensor_parallel_size": 2,         # TP 并行度
    "pipeline_parallel_size": 1,       # PP 并行度
    
    # Speculative Decoding 配置（可选）
    "speculative_model": None,         # Draft model
    "num_speculative_tokens": 0,       # Speculative token 数
}
```

#### TensorRT-LLM 性能配置

```python
# TensorRT-LLM 配置示例
trtllm_config = {
    # 模型配置
    "model_dir": "./models/qwen-7b-trtllm",
    "max_batch_size": 128,
    "max_input_len": 2048,
    "max_output_len": 2048,
    "max_beam_width": 1,
    
    # KV Cache 配置
    "max_tokens_in_paged_kv_cache": 25600,
    "kv_cache_enable_block_reuse": True,
    
    # Quantization 配置
    "quantization": {
        "enabled": True,
        "type": "int8",  # int8, int4, fp8
        "calibration_data": "./calibration_data.jsonl"
    },
    
    # Batching 配置
    "batch_scheduler_policy": "max_utilization",  # max_utilization, guaranteed_no_evict
    
    # Parallelism 配置
    "tensor_parallel": {
        "world_size": 2,
        "gpus": [0, 1]
    }
}
```

## 权衡分析

| 优化方案 | 收益 | 代价 | 适用边界 |
|---------|------|------|----------|
| **增大 batch size** | 吞吐提升 2-5x | 延迟增加、显存需求大 | 吞吐优先、用户延迟容忍度高 |
| **KV Cache 量化（int8/int4）** | 显存减少 50-75% | 精度下降 1-5% | 长上下文场景、显存紧张 |
| **Paged Attention** | 消除显存碎片、支持更大 batch | 实现复杂、小块可能影响 kernel 效率 | 长上下文、动态长度请求 |
| **Continuous Batching** | 吞吐提升 1.5-2x | 实现复杂、调度开销 | 多请求并发、延迟要求不高 |
| **Prefix Cache** | TTFT 降低 50-90%（命中时） | 显存占用增加、cache 管理 | Prompt 重复率高（RAG、模板对话） |
| **Speculative Decoding** | 延迟降低 2-3x（接受率高时） | 额外 draft model 显存、可能变慢 | 接受率 > 0.5、延迟敏感场景 |
| **TP 并行** | 支持 larger model、加速推理 | 通信开销、多卡占用 | 大模型（>70B）、有足够 GPU |
| **PP 并行** | 支持 larger model、减少显存/卡 | Bubble 开销、延迟增加 | 大模型、显存紧张、TP 受限 |
| **Chunked Prefill** | 降低长 prompt 的 TTFT | 实现复杂、调度复杂 | 长上下文场景、TTFT 敏感 |

### 详细权衡：Batch Size vs 延迟

```python
# Batch size 对性能的影响（示例数据）
performance_data = [
    {"batch_size": 1, "throughput": 100, "avg_latency": 0.5, "p99_latency": 0.6},
    {"batch_size": 8, "throughput": 500, "avg_latency": 1.2, "p99_latency": 2.0},
    {"batch_size": 16, "throughput": 800, "avg_latency": 2.5, "p99_latency": 4.5},
    {"batch_size": 32, "throughput": 1200, "avg_latency": 5.0, "p99_latency": 10.0},
    {"batch_size": 64, "throughput": 1500, "avg_latency": 10.0, "p99_latency": 20.0},
]

# 权衡：
# - 吞吐：随 batch size 增大，但增速递减
# - 延迟：随 batch size 线性增长
# - 最优点：根据业务需求选择
#   - 实时对话：batch=8~16，延迟 < 2s
#   - 批量处理：batch=32~64，吞吐优先
```

## 高频追问（至少 5 个）

### 1. Q: 怎么做性能 profiling？（大模型-推理infra）

**A**: 

**性能 Profiling 三步法**：

**第一步：建立监控**
- 关键指标：throughput、TTFT、TPOT、queue_latency、GPU utilization、memory usage
- 采集频率：每秒采集，每分钟聚合
- 存储与可视化：Prometheus + Grafana

**第二步：定位瓶颈阶段**
```python
# 根据指标定位
if queue_latency > threshold:
    bottleneck = "scheduling"  # 调度问题
elif ttft > threshold:
    bottleneck = "prefill"     # prefill 慢
elif tpot > threshold:
    bottleneck = "decode"      # decode 慢或 KV cache 读写慢
```

**第三步：算子级剖析**
- **nsys**：记录时间线，看哪个阶段耗时最多
- **PyTorch Profiler**：记录每个算子的 CPU/CUDA 时间
- **ncu**：深入分析单个 kernel 的性能

**工具使用**：
```bash
# nsys profiling
nsys profile -o profile python inference.py

# PyTorch Profiler
with torch.profiler.profile() as prof:
    model.generate(prompt)
print(prof.key_averages().table())
```

### 2. Q: 说一个你会抓的关键指标？（大模型-推理infra）

**A**: 

**P99 TTFT（Time To First Token）**

**为什么重要**：
- 直接影响用户感知：用户等待多久看到第一个字
- 反映 prefill 性能和调度效率
- P99 比平均值更能暴露长尾问题

**如何抓取**：
```python
# 记录每次请求的 TTFT
ttfts = []
for request in requests:
    start_time = time.time()
    first_token_time = model.generate_first_token(request)
    ttft = first_token_time - start_time
    ttfts.append(ttft)

# 计算 P99
p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)]
```

**正常范围**：
- 短 prompt（< 512 tokens）：50-100ms
- 中等 prompt（512-2048 tokens）：100-500ms
- 长 prompt（> 2048 tokens）：500ms-2s

**异常原因**：
- P99 > 500ms 且均值低：偶发慢，排查长 prompt 或 GC
- 均值 > 500ms：整体 prefill 慢，排查 batch size 或 kernel

### 3. Q: 吞吐突然掉一半怎么查？（大模型-推理infra）

**A**: 

**排查路径**：

```
1. 对比关键指标
   当前 vs 基线：
   - batch_size 是否下降？
   - TTFT/TPOT 是否增加？
   - GPU utilization 是否下降？
   - cache_hit_rate 是否下降？

2. 检查请求模式
   - prompt 长度分布是否变化？
   - 请求频率是否下降？
   - 是否有大量长上下文请求？

3. 检查系统状态
   - GPU 温度、频率是否正常（是否降频）？
   - 显存是否接近 OOM？
   - 是否有其他进程占用 GPU？

4. 检查配置变更
   - 是否有热更新？
   - batch size 限制是否变化？
   - cache 配置是否变化？

5. 使用 profiling 工具
   - nsys 记录时间线，找耗时突增的阶段
```

**常见原因**：
- 请求模式变化：长 prompt 增多
- 热更新后配置错误
- GPU 降频（温度过高）
- 显存接近 OOM，频繁驱逐 cache

### 4. Q: OOM 但显存看着还够？（大模型-推理infra）

**A**: 

**原因分析**：

**1. 碎片化（Fragmentation）**
```python
# PyTorch 的显存分配
memory_allocated = torch.cuda.memory_allocated() / 1e9  # 实际使用
memory_reserved = torch.cuda.memory_reserved() / 1e9    # 预留（包含碎片）

# 碎片率
fragmentation = (memory_reserved - memory_allocated) / memory_reserved
# 如果 > 30%，碎片严重
```

**解决**：
- 使用 vLLM paged attention（消除碎片）
- 调整 PyTorch 分配器：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
- 定期重启服务

**2. 峰值显存**
- 某些中间步骤需要额外显存
- 例如：attention 计算的临时 buffer、TP/PP 通信 buffer

**解决**：
- Activation checkpointing
- 监控峰值显存，预估时留余量

**3. 显存泄漏**
- 显存持续增长，不释放
- Python 对象引用未释放、KV cache 未清理

**解决**：
- `torch.cuda.memory_summary()` 定位泄漏点
- 显式释放 + 定期重启

### 5. Q: 输出重复/循环常见原因？（大模型-推理infra）

**A**: 

**常见原因**：

**1. Sampling 参数不当**
- **temperature 太低**（< 0.5）：模型过于贪婪，选择概率最高的 token
- **repetition_penalty 太低**（< 1.0）：不惩罚重复的 token
- **top_k/top_p 太严格**：选择空间太小

**解决方案**：
```python
# 调整 sampling 参数
sampling_params = {
    "temperature": 0.7,           # 提高温度
    "repetition_penalty": 1.1,    # 增加重复惩罚
    "top_k": 50,
    "top_p": 0.95
}
```

**2. EOS token 问题**
- 模型未生成 EOS token，陷入循环
- EOS token 设置错误

**解决方案**：
- 检查 EOS token id 设置
- 添加 max_tokens 限制
- 训练时加强 EOS token 的学习

**3. 模型训练问题**
- 训练数据有大量重复
- 过度微调导致模式崩溃

**解决方案**：
- 过滤训练数据中的重复
- 检查微调数据质量
- 添加 n-gram blocking

**4. Prompt 引导**
- Prompt 本身包含重复模式
- Prompt 引导模型重复

**解决方案**：
- 优化 Prompt 设计
- 添加示例展示非重复输出

### 6. Q: 多轮对话越聊越飘？（大模型-推理infra）

**A**: 

**原因与解决方案**：

**1. 上下文超长**
- 上下文超出窗口，早期内容被截断
- Position ID 累积超过训练长度

**解决方案**：
- 滑动窗口：只保留最近 N 轮
- 对话摘要：定期压缩早期对话
- 重置 position ID

**2. 指令遗忘**
- 模型忘记初始 system instruction
- 长对话中早期指令权重被稀释

**解决方案**：
- 周期性重复 system instruction
- 使用 special token 标记重要指令

**3. KV cache 质量下降**
- 早期 token 的 cache 包含过时信息
- Cache 累积误差

**解决方案**：
- Cache eviction：移除早期的 cache
- 定期重算重要部分的 cache

**4. 话题漂移**
- 对话偏离原始目标

**解决方案**：
- 对话管理器：检测话题漂移
- 定期提醒原始目标

**5. 数据分布问题**
- 训练数据中长对话少

**解决方案**：
- 增加多轮对话训练数据
- 使用多轮对话特定的 SFT 数据

### 7. Q: 热更新后效果变差？（大模型-推理infra）

**A**: 

**常见原因**：

**1. Tokenizer 不匹配**
- 新模型使用不同 tokenizer
- 导致输出乱码或格式错误

**解决**：热更新时同步 tokenizer

**2. KV cache 未清空**
- 旧 cache 被新模型使用
- 导致输出混淆（"串台"）

**解决**：热更新时清空所有 cache

**3. 配置不匹配**
- hidden_size、vocab_size 等配置变化
- 导致推理错误

**解决**：检查配置一致性

**4. 动态库未重载**
- FlashAttention 等 kernel 未重新加载
- 导致自定义算子错误

**解决**：重启进程而非热更新

**最佳实践**：
- 热更新前测试新模型
- 热更新时清空所有状态（cache、tokenizer、配置）
- 监控热更新后的性能指标
- 准备回滚方案

### 8. Q: 线上"串台"通常怎么来的？（大模型-推理infra）

**A**: 

**串台定义**：不同请求的输出混淆，A 请求的输出包含 B 请求的内容

**常见原因**：

**1. KV cache 混淆**
```python
# 错误示例：共享 cache
shared_cache = {}  # 所有请求共享

# 正确做法：隔离 cache
def process_request(request):
    # 每个请求独立的 cache key
    cache_key = f"{request.user_id}_{request.session_id}"
    cache = get_or_create_cache(cache_key)
```

**原因**：
- Prefix cache 命中后，cache 未隔离
- Multi-turn 对话时，不同用户的 cache 混淆

**解决**：
- 每个请求独立的 cache block
- cache key 包含 user_id 或 session_id

**2. Position ID 错误**
```python
# 错误示例：全局 position_id
global_position_id = 0

def process_request(request):
    global global_position_id
    # 使用全局 position_id，不同请求会混淆
    output = model.generate(request, position_id=global_position_id)
    global_position_id += len(output)
```

**解决**：
- 每个请求独立的 position_id 起始点
- 清空或重置 position_id 状态

**3. Batch 维度混淆**
```python
# 错误示例：batch 维度索引错误
outputs = model.generate_batch(prompts)  # shape: [batch, seq_len]
# 错误：第一个请求拿到第二个请求的输出
output_0 = outputs[1]  # 应该是 outputs[0]
```

**解决**：
- 严格检查 batch 维度索引
- 使用框架提供的 batching 机制

**4. 热更新后状态残留**
- 新请求包含旧模型的 cache

**解决**：
- 热更新时清空所有 cache
- 重启服务而非热更新

## 常见错误（至少 3 个）

### 1. 错误：只看平均延迟，忽略 P99

**现象**：平均延迟正常，但用户经常投诉慢

**错误做法**：
```python
# 只看平均延迟
avg_latency = sum(latencies) / len(latencies)
if avg_latency < 1.0:
    print("Performance OK")  # 忽略长尾
```

**正确做法**：
```python
# 同时关注 P99
avg_latency = sum(latencies) / len(latencies)
p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

print(f"Avg: {avg_latency:.2f}s, P99: {p99_latency:.2f}s")

if p99_latency > 5.0:
    print("⚠️  High P99 latency, investigate long tail")
    # 分析长尾原因
    slow_requests = [r for r in requests if r.latency > p99_latency]
    analyze_slow_requests(slow_requests)
```

**原因**：
- 平均值掩盖长尾问题
- P99 更能反映用户体验
- 长尾请求可能是系统问题的信号

### 2. 错误：固定 batch size，不根据负载调整

**现象**：低负载时延迟高，高负载时吞吐低

**错误做法**：
```python
# 固定 batch size
BATCH_SIZE = 32

def inference(requests):
    batch = requests[:BATCH_SIZE]  # 不管请求多少，都等够 32 个
    output = model.generate(batch)
    return output
```

**正确做法**：
```python
# 动态 batch size
MAX_BATCH_SIZE = 64
MAX_WAIT_TIME = 0.1  # 最多等待 100ms

def inference(requests):
    start_time = time.time()
    batch = []
    
    # 动态凑 batch
    while len(batch) < MAX_BATCH_SIZE:
        if time.time() - start_time > MAX_WAIT_TIME:
            break  # 超时，不等了
        if request_queue:
            batch.append(request_queue.pop())
        else:
            time.sleep(0.01)
    
    if not batch:
        return []
    
    output = model.generate(batch)
    return output
```

**原因**：
- 低负载时：固定 batch size 导致等待时间长
- 高负载时：固定 batch size 无法充分利用 GPU
- 动态 batch size 可以平衡吞吐和延迟

### 3. 错误：忽略 KV cache 清理，导致显存泄漏

**现象**：运行一段时间后显存持续增长，最终 OOM

**错误做法**：
```python
# 不清理 cache
global_cache = {}

def inference(request):
    # 生成 cache
    cache = generate_kv_cache(request)
    global_cache[request.id] = cache
    
    # 生成输出
    output = model.generate(request, cache)
    
    # 忘记删除 cache
    return output
```

**正确做法**：
```python
# 及时清理 cache
def inference(request):
    try:
        # 生成 cache
        cache = generate_kv_cache(request)
        
        # 生成输出
        output = model.generate(request, cache)
        
        return output
    finally:
        # 清理 cache
        del cache
        torch.cuda.empty_cache()
```

**原因**：
- KV cache 是显存占用的大头
- Python GC 不一定及时回收
- 显式清理可以避免显存泄漏

### 4. 错误：Profiling 只看 GPU，忽略 CPU 和 IO

**现象**：GPU 利用率低，但找不到原因

**错误做法**：
```python
# 只看 GPU 时间
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    model.generate(prompt)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# 结论：GPU 时间短，但整体延迟高，不知道为什么
```

**正确做法**：
```python
# 同时看 CPU 和 CUDA
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model.generate(prompt)

print(prof.key_averages().table(sort_by="cpu_time_total"))
# 可能发现：CPU 预处理或后处理占大头
```

**原因**：
- CPU 预处理（tokenize、后处理）可能占大头
- IO（读取 prompt、写入输出）可能是瓶颈
- 多卡场景下通信开销也是瓶颈

## 反问面试官的问题

### 技术深度类

1. **"团队在推理侧主要关注哪个指标？TTFT、TPOT 还是 throughput？如何权衡？"**
   - 了解团队的关注点：实时性 vs 吞吐
   - 了解权衡策略：batch size、调度策略

2. **"线上遇到过最棘手的性能问题是什么？怎么定位和解决的？"**
   - 了解团队的故障排查能力
   - 了解实际问题的复杂度

3. **"团队有做过 speculative decoding 或 prefix cache 吗？效果如何？"**
   - 了解技术深度
   - 了解优化方案的落地经验

4. **"GPU 显存不够时，优先考虑哪种优化？量化、TP 并行、还是 KV cache 优化？"**
   - 了解优化思路
   - 了解技术决策过程

### 业务场景类

1. **"线上推理的最大延迟要求是多少？如何保证 P99 延迟在阈值内？"**
   - 了解业务约束
   - 了解 SLA 保障机制

2. **"团队有遇到过热更新后效果变差或串台的问题吗？怎么避免的？"**
   - 了解实际运维经验
   - 了解线上稳定性保障

3. **"多轮对话场景下，如何处理上下文超长和指令遗忘问题？"**
   - 了解业务场景
   - 了解技术方案

## 自测题

### 口述（能流畅讲清楚）

1. **性能 Profiling 的三步法是什么？每步做什么？**
   - 第一步：建立监控（关键指标）
   - 第二步：定位瓶颈阶段（prefill/decode/scheduling）
   - 第三步：算子级剖析（nsys/PyTorch Profiler）

2. **关键指标有哪些？每个指标反映什么问题？**
   - throughput：整体吞吐
   - TTFT：prefill 性能
   - TPOT：decode 性能
   - queue_latency：调度效率
   - cache_hit_rate：缓存效率
   - GPU utilization：GPU 利用率

3. **OOM 但显存看着还够，可能的原因是什么？**
   - 碎片化
   - 峰值显存
   - 显存泄漏
   - 框架开销

4. **输出重复/循环的常见原因和解决方案？**
   - sampling 参数不当：调整 temperature、repetition_penalty
   - EOS token 问题：检查 EOS token id
   - 模型训练问题：过滤重复数据、检查微调质量

5. **多轮对话越聊越飘的原因和解决方案？**
   - 上下文超长：滑动窗口、摘要
   - 指令遗忘：周期性重复 system instruction
   - KV cache 质量下降：cache eviction
   - 话题漂移：对话管理器

### 手写（5 分钟能写出）

1. **计算 P99 TTFT**
```python
def calculate_p99_ttft(ttfts: List[float]) -> float:
    """
    计算P99 TTFT
    
    Args:
        ttfts: TTFT列表（秒）
    
    Returns:
        P99 TTFT（毫秒）
    """
    if not ttfts:
        return 0.0
    
    sorted_ttfts = sorted(ttfts)
    p99_index = int(len(sorted_ttfts) * 0.99)
    p99_ttft_sec = sorted_ttfts[p99_index]
    
    return p99_ttft_sec * 1000  # 转换为毫秒
```

2. **诊断吞吐下降**
```python
def diagnose_throughput_drop(current: dict, baseline: dict) -> str:
    """
    诊断吞吐下降原因
    
    Args:
        current: 当前指标
        baseline: 基线指标
    
    Returns:
        诊断结果
    """
    if current["avg_batch_size"] < baseline["avg_batch_size"] * 0.8:
        return "Batch size dropped, check scheduler or request rate"
    
    if current["avg_ttft_ms"] > baseline["avg_ttft_ms"] * 1.5:
        return "TTFT increased, check prefill or cache"
    
    if current["avg_tpot_ms"] > baseline["avg_tpot_ms"] * 1.5:
        return "TPOT increased, check decode or KV cache"
    
    if current["cache_hit_rate"] < baseline["cache_hit_rate"] * 0.5:
        return "Cache hit rate dropped, check cache config or prompt pattern"
    
    return "No obvious bottleneck, check request pattern or hardware"
```

3. **KV cache 显存估算**
```python
def estimate_kv_cache_memory(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "fp16"
) -> float:
    """
    估算 KV cache 显存占用
    
    Args:
        num_layers: 层数
        batch_size: 批大小
        seq_len: 序列长度
        num_heads: 注意力头数
        head_dim: 头维度
        dtype: 数据类型（fp16, bf16, int8）
    
    Returns:
        显存占用（GB）
    """
    dtype_size = {"fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 2)
    
    # KV cache = 2 (K+V) * layers * batch * seq * heads * head_dim * dtype_size
    memory_bytes = (
        2 * num_layers * batch_size * seq_len * num_heads * head_dim * dtype_size
    )
    
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb
```

## 标签
#推理 #Profiling #工程 #性能优化 #故障排查 #KV_Cache #vLLM #TensorRT-LLM #GPU优化 #线上稳定性 #字节 #阿里 #腾讯 #美团 #百度

## 相关文档
- [[02-KV Cache核心]]：KV cache 的原理和优化
- [[04-Batching与调度]]：动态 batching 和调度策略
- [[06-量化推理]]：INT8/INT4 量化的性能权衡
- [[../08-数值精度/01-训练精度选择]]：FP16/BF16 的精度问题
