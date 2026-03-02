# DeepSpeed ZeRO

## 一句话结论
ZeRO（Zero Redundancy Optimizer）通过切分优化器状态、梯度、参数，将单卡显存从 O(ψ) 降至 O(ψ/N)，ZeRO-1/2/3 分别切分 optimizer state / gradients / parameters，使超大模型分布式训练成为可能。

## 核心定义/公式

### ZeRO-1/2/3 切分策略

| 阶段 | 切分对象 | 显存占用（每卡） | 通信开销 |
|------|---------|-----------------|---------|
| ZeRO-1 | Optimizer States | ψ/N (OS) + ψ + ψ | 1.5× |
| ZeRO-2 | Optimizer States + Gradients | ψ/N (OS) + ψ/N (G) + ψ | 1.5× |
| ZeRO-3 | Optimizer States + Gradients + Parameters | ψ/N (OS) + ψ/N (G) + ψ/N (P) | 3.0× |

其中 ψ = 模型参数量，N = GPU 数量。

### 显存估算公式（混合精度训练）

**基础公式**：
```
总显存 = 参数 + 梯度 + 优化器状态 + 激活 + 碎片/峰值开销
```

**各部分详细计算**：

1. **模型参数（FP16）**：
   ```
   Memory_params = 2 × ψ bytes
   ```
   - 7B 模型：14 GB
   - 72B 模型：144 GB

2. **梯度（FP16）**：
   ```
   Memory_grads = 2 × ψ bytes
   ```

3. **优化器状态（Adam，FP32）**：
   ```
   Memory_optimizer = 4 × ψ (参数 FP32) + 8 × ψ (momentum + variance)
                    = 12 × ψ bytes
   ```
   - 7B 模型：84 GB
   - 72B 模型：864 GB

4. **激活（Activation）**：
   ```
   Memory_activation = batch_size × seq_len × hidden_dim × num_layers × factor
   ```
   - factor ≈ 10-20（含中间激活、dropout 等）

5. **峰值显存（含碎片）**：
   ```
   Memory_peak = (Memory_params + Memory_grads + Memory_optimizer + Memory_activation) × 1.1-1.3
   ```

### LoRA 显存计算

**LoRA 参数量**：
```python
# LoRA: A (rank × d_in), B (d_out × rank)
params_lora = rank × (d_in + d_out)
```

**显存占用**：
```python
# 全参微调（不使用 ZeRO）
Memory_full = 2ψ + 2ψ + 12ψ + activation  # 16ψ + activation

# LoRA 微调（rank=64, target_modules=['q_proj', 'v_proj'])
# 假设 attention 维度 d_model/h = hidden_dim
params_lora_per_layer = 64 × (hidden_dim + hidden_dim) = 128 × hidden_dim
total_lora_params = num_layers × 2 × 128 × hidden_dim

Memory_lora = 2ψ (冻结权重 FP16) + 2 × params_lora (梯度) + 12 × params_lora (优化器) + activation
```

**实例：Qwen2-72B LoRA 微调**
- 冻结权重：144 GB
- LoRA 参数（rank=64, 全部 attention）：约 0.5 GB
- LoRA 梯度 + 优化器：约 7 GB
- 激活（batch=1, seq=2048）：约 15-20 GB
- **总计**：约 160-170 GB（A100 80GB × 2 可承载）

### ZeRO-3 配置示例

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",  # CPU offload
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,  # 通信计算重叠
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 4,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

## 为什么（2-3 个因果链）

### 1. 为什么需要 ZeRO：大模型训练的显存瓶颈

**现象**：72B 模型全参训练需要 1152 GB 显存（16ψ），单卡 A100 80GB 远远不够

**根因**：
- **数据并行冗余**：每张卡都保存完整模型副本（参数、梯度、优化器状态）
- **Adam 优化器膨胀**：momentum + variance = 8× 参数量（FP32），是参数本身的 4 倍
- **混合精度矛盾**：训练用 FP16，但优化器需要 FP32 累积精度

**结果**：ZeRO 通过状态分片消除冗余，ZeRO-3 可将显存降至 O(ψ/N)

### 2. 为什么 ZeRO-3 显存还是大：显存占用不止于模型状态

**现象**：ZeRO-3 微调 72B 模型，单卡仍需 20-30 GB

**根因**：
- **激活占用**：activation ∝ batch_size × seq_len × hidden_dim × num_layers，不参与 ZeRO 分片
  - 72B, batch=1, seq=2048: 约 15-20 GB
- **通信缓冲区**：all-gather/all-reduce 需要临时缓冲
  - 参数收集缓冲：约 1-2 GB
- **显存碎片**：频繁分配/释放导致碎片，实际利用率 70-80%
- **CUDA 内核开销**：约 1-2 GB 固定开销

**结果**：ZeRO-3 只优化了模型状态（参数/梯度/优化器），激活和系统开销仍需额外显存

### 3. 为什么 LoRA 能大幅降低显存：参数效率的本质

**现象**：LoRA 微调 72B 仅需 2 张 A100，全参需要 16 张

**根因**：
- **低秩分解**：权重更新 ΔW = BA，参数量从 d² 降至 2×r×d
  - 72B attention 权重：每层 Q/K/V/O 约 512 MB
  - LoRA (rank=64)：每层仅 0.003 GB
- **冻结主干**：模型主体参数冻结，不存储梯度和优化器状态
  - 冻结权重：144 GB（FP16，只读）
  - LoRA 参数：0.5 GB + 梯度优化器约 7 GB

**结果**：LoRA 将优化开销从 O(ψ) 降至 O(params_lora)，降低 2-3 个数量级

## 怎么做（可落地步骤）

### 标准做法

**1. ZeRO 阶段选择**

```python
# 决策流程
def choose_zero_stage(model_size, gpu_memory, num_gpus):
    """
    model_size: 参数量（B）
    gpu_memory: 单卡显存（GB）
    num_gpus: GPU 数量
    """
    # 估算基础需求（不含 ZeRO）
    params_memory = model_size * 2  # FP16 权重
    grad_memory = model_size * 2     # 梯度
    optimizer_memory = model_size * 12  # Adam 状态
    
    total = params_memory + grad_memory + optimizer_memory
    
    if total < gpu_memory * num_gpus * 0.7:
        return "ZeRO-1 或 DDP"  # 显存充足
    elif total < gpu_memory * num_gpus * 0.8:
        return "ZeRO-2"  # 切分优化器+梯度
    else:
        return "ZeRO-3"  # 必须切分参数
```

**2. 显存估算模板（面试答题公式）**

```
步骤 1：计算参数显存
  - FP16 权重：2ψ GB
  - 72B → 144 GB

步骤 2：计算优化器状态（Adam）
  - FP32 参数副本：4ψ GB
  - Momentum：4ψ GB
  - Variance：4ψ GB
  - 总计：12ψ GB
  - 72B → 864 GB

步骤 3：计算 ZeRO 分片后显存
  - ZeRO-3: (2ψ + 2ψ + 12ψ) / N
  - 72B, 8 卡: 144 / 8 = 18 GB

步骤 4：加激活显存
  - Activation ≈ batch × seq × hidden × layers × factor
  - 72B, batch=1, seq=2048: 约 15-20 GB

步骤 5：加峰值开销（碎片+通信缓冲）
  - 峰值系数：1.1-1.3
  - 总计：(18 + 20) × 1.2 ≈ 45 GB

答案：72B 全参训练，ZeRO-3, 8 卡，单卡约 40-50 GB
```

**3. LoRA vs 全参显存计算**

```python
def estimate_memory_lora_vs_full(
    model_size_b,  # 模型参数量（B）
    lora_rank=64,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    batch_size=1,
    seq_len=2048,
    hidden_dim=8192,  # 72B 级别
    num_layers=80
):
    """
    估算 LoRA 和全参微调的显存需求
    """
    psi = model_size_b * 1e9  # 参数量
    
    # 全参微调（ZeRO-1, 8卡）
    params_full = 2 * psi  # FP16
    grads_full = 2 * psi
    optimizer_full = 12 * psi  # Adam
    activation_full = batch_size * seq_len * hidden_dim * num_layers * 15  # factor≈15
    
    memory_full_per_gpu = (params_full + grads_full + optimizer_full) / 8 / 1e9 + activation_full / 1e9
    memory_full_per_gpu *= 1.2  # 碎片
    
    # LoRA 微调
    params_lora = 0
    for module in target_modules:
        # 假设每个 module 维度为 hidden_dim
        params_lora += lora_rank * hidden_dim * 2  # A+B
    
    params_lora *= num_layers
    params_lora_bytes = params_lora * 2  # FP16
    
    memory_lora = {
        'frozen_weights': params_full / 8 / 1e9,  # ZeRO-3 分片
        'lora_params': params_lora_bytes / 1e9,
        'lora_grads': params_lora_bytes / 1e9,
        'lora_optimizer': params_lora_bytes * 6 / 1e9,  # Adam FP32
        'activation': activation_full / 1e9
    }
    
    total_lora = sum(memory_lora.values()) * 1.2
    
    return {
        'full_finetune': f"{memory_full_per_gpu:.1f} GB per GPU (8 cards)",
        'lora_finetune': f"{total_lora:.1f} GB per GPU (2 cards)",
        'lora_breakdown': memory_lora
    }

# 实例：Qwen2-72B
result = estimate_memory_lora_vs_full(72, lora_rank=64, batch_size=1, seq_len=2048)
# 输出：
# full_finetune: "135.0 GB per GPU (8 cards)" → 需要多卡
# lora_finetune: "78.5 GB per GPU (2 cards)" → 单卡 A100 80GB 可承载
```

### 关键配置/参数

**DeepSpeed ZeRO-3 关键参数**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `stage` | 3 | 大模型必须用 3 |
| `offload_optimizer.device` | `cpu` | 优化器状态卸载到 CPU，节省 GPU 显存 |
| `offload_param.device` | `cpu` | 参数卸载，但会降低速度 |
| `overlap_comm` | `true` | 通信计算重叠，提升速度 |
| `contiguous_gradients` | `true` | 连续梯度存储，减少碎片 |
| `reduce_bucket_size` | `1e6` | 通信桶大小，越大越省显存但越慢 |
| `stage3_prefetch_bucket_size` | `1e7` | 预取桶大小，平衡速度与显存 |

**LoRA 关键参数**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `r` (rank) | 8-64 | 越大表达能力越强但显存越多 |
| `lora_alpha` | 16-64 | 通常设为 rank 的 2 倍 |
| `target_modules` | `['q_proj', 'v_proj']` | 全量微调用所有线性层 |
| `lora_dropout` | 0.05-0.1 | 防止过拟合 |

### 代码示例

**启动 ZeRO-3 训练**：

```python
# launch_deepspeed.sh
deepspeed \
    --num_gpus=8 \
    --num_nodes=1 \
    train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Qwen/Qwen2-72B \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --bf16  # 使用 BF16 替代 FP16，更稳定
```

**LoRA 微调配置**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-72B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 301,989,888 || all params: 72,702,742,528 || trainable%: 0.42%
```

**显存监控**：

```python
import torch
import deepspeed

def log_memory_usage(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# 训练循环中插入
for step, batch in enumerate(dataloader):
    if step % 10 == 0:
        log_memory_usage(f"Step {step}")
    
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **ZeRO-1** | 切分优化器状态，显存降至 4ψ+activation | 通信量 1.5×，速度略慢 | 参数量中等，显存略不足 |
| **ZeRO-2** | 切分优化器+梯度，显存降至 2ψ+activation | 通信量 1.5×，梯度同步开销 | 参数量大，单卡显存紧张 |
| **ZeRO-3** | 切分参数+梯度+优化器，显存降至 ψ/N+activation | 通信量 3×，速度显著下降 | 超大模型（>70B），必须切分 |
| **ZeRO-3 + CPU Offload** | 进一步降低 GPU 显存至 10-20 GB | CPU-GPU 传输瓶颈，速度降低 2-5× | 单卡训练超大模型，时间不敏感 |
| **LoRA** | 显存降低 90%+，速度快，易部署 | 参数量小，表达能力有限 | 微调任务，保持原模型能力 |
| **全参微调** | 表达能力最强，充分适应新任务 | 显存需求极大，训练成本高 | 领域适应、能力迁移、资源充足 |

## 高频追问（至少 5 个）

### 1. Q: ZeRO-3 为什么通信量是 DDP 的 3 倍？

**A**: ZeRO-3 需要在前向和反向时收集分片的参数（all-gather），反向时同步梯度（all-reduce），更新优化器时再次收集（all-gather），总共 3 轮通信。DDP 只需梯度同步（1 轮 all-reduce）。

**细节**：
- 前向：all-gather 参数 → 计算 → 释放
- 反向：all-gather 参数 → 计算 → all-gather 梯度 → all-reduce → 释放
- 优化器：all-gather 状态 → 更新 → 释放

### 2. Q: ZeRO-3 微调 72B 模型，每张卡显存多少？如何估算？

**A**: 
- **参数分片**：(2ψ + 2ψ + 12ψ) / N = 16×72 / 8 = 144 GB（假设 ZeRO-1，未切分）
- **ZeRO-3 分片**：(16×72) / 8 = 18 GB
- **激活**：batch×seq×hidden×layers×factor = 1×2048×8192×80×15 ≈ 20 GB
- **峰值系数**：×1.2 = 45 GB
- **实际测试**：Qwen2-72B, ZeRO-3, 8×A100 80GB, 单卡约 40-50 GB

### 3. Q: LoRA 微调 72B 需要多少显存？为什么比全参少这么多？

**A**:
- **冻结权重**：144 GB（FP16，只读，可分片）
- **LoRA 参数**：rank=64, target_modules=all_linear → 约 0.5 GB
- **LoRA 梯度+优化器**：约 7 GB
- **激活**：约 20 GB
- **总计**：约 170 GB（2×A100 80GB 可承载）

**为什么少**：
1. LoRA 参数量极小（0.5 GB vs 72B 主干），优化器状态少
2. 冻结权重不参与梯度计算，无反向传播开销
3. 可用 ZeRO-3 切分冻结权重，进一步降低单卡显存

### 4. Q: ZeRO-Offload 什么时候用？有什么坑？

**A**:
**使用场景**：
- 单卡训练大模型（如 7B 在 24GB 显卡）
- 显存极度紧张，可容忍速度下降

**坑**：
1. **速度下降严重**：CPU-GPU 传输带宽限制，速度降低 2-5×
2. **CPU 内存需求**：需要大内存（72B 模型需要 150+ GB CPU 内存）
3. **不适合多卡**：多卡场景优先用 ZeRO-3 分片，速度更快
4. **梯度累积不适用**：Offload 与 gradient accumulation 配合有问题

**推荐**：
- 多卡优先用 ZeRO-3
- 单卡且显存不足才用 Offload
- 使用 NVMe Offload（DeepSpeed-Training）替代 CPU Offload

### 5. Q: 激活检查点（Activation Checkpointing）如何配合 ZeRO？

**A**:
**原理**：前向时不存储中间激活，反向时重新计算，用计算换显存。

**显存收益**：
- 激活显存从 O(batch×seq×hidden×layers) 降至 O(batch×seq×hidden)
- 72B 模型，batch=1, seq=2048：从 20 GB 降至 1-2 GB

**代价**：
- 速度降低 20-30%（需重新计算）

**配置**：
```python
# PyTorch
model.gradient_checkpointing_enable()

# DeepSpeed
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,  # 不推荐，太慢
        "contiguous_memory_optimization": true
    }
}
```

**最佳实践**：
- ZeRO-3 + Activation Checkpointing + gradient accumulation
- 可将 72B 模型单卡显存降至 30-40 GB

### 6. Q: 项目里微调实际用到了多少显存？如何答？

**A**:
**答题模板**：
1. **模型规模**：Qwen2-72B，参数量 72B
2. **训练配置**：
   - LoRA rank=64, target_modules=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']
   - batch_size=1, gradient_accumulation_steps=4
   - seq_len=2048, BF16
3. **显存估算**：
   - 冻结权重（FP16）：144 GB（ZeRO-3 分片到 18 GB/卡）
   - LoRA 参数：0.3 GB
   - LoRA 梯度+优化器：3 GB
   - 激活：20 GB
   - 碎片与峰值：×1.2 = 50 GB
4. **实测**：
   - 使用 8×A100 80GB，ZeRO-3, 单卡峰值 48 GB
   - 使用 2×A100 80GB, ZeRO-3, 单卡峰值 75 GB

### 7. Q: ZeRO-3 和模型并行（TP/PP）的区别？如何选择？

**A**:
| 维度 | ZeRO-3 | TP（张量并行） | PP（流水线并行） |
|------|--------|---------------|-----------------|
| **切分对象** | 优化器状态/梯度/参数 | 模型层内矩阵 | 模型层间 |
| **通信模式** | All-gather/All-reduce | All-reduce（高频） | 点对点（低频） |
| **显存收益** | 线性降低至 O(ψ/N) | 线性降低 | 线性降低 |
| **速度影响** | 通信开销 3× | 通信开销中等 | bubble 开销 |
| **适用场景** | 数据并行，大模型训练 | 超大模型，单机多卡 | 跨机训练，大 batch |

**选择策略**：
- **单机多卡（8×A100）**：ZeRO-3 或 TP=8
- **跨机训练**：ZeRO-3 + PP（或 3D 并行：DP×TP×PP）
- **显存极度紧张**：ZeRO-3 + CPU Offload

### 8. Q: 为什么有的层不适合量化？INT4 微调有什么坑？

**A**:
**不适合量化的层**：
1. **LayerNorm/RMSNorm**：归一化层对精度敏感，量化会导致分布偏移
2. **输出层（lm_head）**：词表维度大，量化误差放大
3. **Attention QKV 投影**：累积误差影响注意力分布

**INT4 微调的坑**：
1. **精度损失**：长上下文场景下幻觉增加，推理质量下降 5-10%
2. **outlier 问题**：部分权重/激活值超出 INT4 表示范围（-8 到 7），需分组量化
3. **微调困难**：QLoRA 虽然能训，但低比特下梯度精度不足，收敛变慢

**缓解方法**：
- 分组量化（Group Quantization）：每 64-128 个元素一组
- 混合精度：关键层保持 FP16/BF16
- QLoRA：用 NF4（NormalFloat4）替代 INT4，保留正态分布特性

## 常见错误（至少 3 个）

### 1. 错误：认为 ZeRO-3 只切分参数

**描述**：误以为 ZeRO-3 只切分模型参数，忽略优化器状态和梯度。

**正确做法**：
- ZeRO-1：切分优化器状态（momentum + variance + FP32 参数副本）
- ZeRO-2：ZeRO-1 + 梯度切分
- ZeRO-3：ZeRO-2 + 参数切分

**记忆口诀**：1-优化器，2-加梯度，3-全切分

### 2. 错误：忽略激活显存，导致 OOM

**描述**：估算显存只算参数和优化器状态，忽略激活占用，实际训练 OOM。

**正确做法**：
- 激活显存 ≈ batch_size × seq_len × hidden_dim × num_layers × factor
- factor 取值：无 checkpointing 时 10-20，有 checkpointing 时 1-2
- 72B, batch=1, seq=2048：无 checkpointing 约 20 GB

**排查步骤**：
1. 监控显存曲线，看峰值在哪里
2. 若峰值在前向传播 → 激活占用大
3. 启用 `gradient_checkpointing_enable()`

### 3. 错误：ZeRO-3 配置参数不当，导致通信爆炸或 OOM

**描述**：
- `reduce_bucket_size` 太小 → 通信次数增加，速度下降
- `stage3_prefetch_bucket_size` 太大 → 显存溢出
- 忘记设置 `overlap_comm=true` → 通信与计算串行

**正确做法**：
```python
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,  # 必须开启
        "reduce_bucket_size": 1e6,  # 平衡通信次数与显存
        "stage3_prefetch_bucket_size": 1e7,  # 根据显存调整
        "contiguous_gradients": true  # 减少碎片
    }
}
```

**调试经验**：
- 先用小 bucket size（1e5），确认能跑
- 逐步增大 bucket size，观察速度和显存
- 监控通信时间占比（应 < 30%）

### 4. 错误：LoRA rank 设得太大，失去参数效率

**描述**：将 LoRA rank 设为 256 或 512，以为越大越好，结果显存接近全参，效果没提升。

**正确做法**：
- rank 建议：8-64（常见 16-32）
- rank 太大：显存增加，过拟合风险
- rank 太小：表达能力不足

**实验经验**：
- Qwen-72B 微调：rank=64 效果最佳，rank=128 无显著提升
- 小模型（7B）：rank=8-16 足够

### 5. 错误：混淆 ZeRO 和模型并行

**描述**：认为 ZeRO-3 就是把模型层切分到不同卡上。

**正确做法**：
- **ZeRO**：数据并行框架，每张卡仍是完整模型（逻辑上），通过分片优化器状态/梯度/参数节省显存
- **模型并行（TP/PP）**：将模型层切分到不同卡，每张卡只有部分层

**区分关键**：
- ZeRO：前向/反向时动态收集参数，计算后释放
- TP：层内矩阵切分，前向时多卡协同计算一层
- PP：层间切分，数据流过不同卡的层

## 反问面试官的问题

### 技术深度类

1. **ZeRO-3 的通信开销 3× 具体来自哪里？有没有优化方法？**
   - 追问：overlap_comm 在你们实际项目中提速多少？瓶颈在哪？

2. **你们团队在实际训练中遇到过 ZeRO 的哪些坑？如何排查和解决的？**
   - 追问：显存 OOM 时，如何定位是参数、优化器、激活还是碎片问题？

3. **LoRA 和全参微调在你们业务场景中的选择标准是什么？效果差距有多大？**
   - 追问：有没有做过 LoRA rank/alpha 的调优实验？最佳配置是什么？

### 业务场景类

1. **你们团队训练的大模型参数量级？训练和推理的成本占比？**
   - 追问：显存优化在整体工程优化中的优先级？

2. **实际项目中，ZeRO-3 和模型并行如何配合使用？有没有 3D 并行的经验？**
   - 追问：跨机训练时，通信瓶颈如何优化？

3. **显存优化和训练速度如何权衡？有没有遇到过为了省显存牺牲太多速度的情况？**
   - 追问：CPU Offload 在生产环境中用过吗？实际效果如何？

## 自测题

### 口述（能流畅讲清楚的知识点）

1. **ZeRO-1/2/3 分别切分什么？为什么显存能从 O(ψ) 降到 O(ψ/N)？**
   - 要点：优化器状态 12ψ → 梯度 2ψ → 参数 2ψ，分片后每卡承担 1/N

2. **估算 Qwen2-72B 全参训练显存需求（8 卡 A100，ZeRO-3，batch=1，seq=2048）**
   - 要点：参数 144 GB → ZeRO-3 分片 18 GB + 激活 20 GB + 峰值 ×1.2 = 45 GB

3. **LoRA 为什么能大幅降低显存？显存占用公式是什么？**
   - 要点：参数效率（rank≪d），冻结主干，只优化 LoRA 参数（0.5 GB vs 72B）

### 手写（5 分钟能写出的代码/公式）

1. **显存估算公式（参数 + 梯度 + 优化器 + 激活）**
   ```python
   def estimate_memory(params_b, batch, seq, hidden, layers, num_gpus, zero_stage):
       psi = params_b * 1e9
       params = 2 * psi  # FP16
       grads = 2 * psi
       optimizer = 12 * psi  # Adam FP32
       activation = batch * seq * hidden * layers * 15  # factor
       
       if zero_stage == 3:
           model_memory = (params + grads + optimizer) / num_gpus
       else:
           model_memory = params + grads + optimizer
       
       total = model_memory + activation
       return total * 1.2 / 1e9  # GB, 峰值系数
   ```

2. **DeepSpeed ZeRO-3 配置（关键参数）**
   ```json
   {
       "zero_optimization": {
           "stage": 3,
           "offload_optimizer": {"device": "cpu"},
           "overlap_comm": true,
           "contiguous_gradients": true,
           "reduce_bucket_size": 1e6
       },
       "fp16": {"enabled": true}
   }
   ```

3. **LoRA 参数量计算**
   ```python
   def lora_params(hidden_dim, rank, num_layers, num_modules):
       # 每个模块：A(rank × hidden) + B(hidden × rank)
       params_per_module = 2 * rank * hidden_dim
       total = params_per_module * num_modules * num_layers
       return total
   ```

## 标签
#训练 #工程 #ZeRO #DeepSpeed #显存优化 #分布式训练 #LoRA

## 相关文档
- [[01-并行策略总览]] - DP/DDP/TP/PP 基础概念与通信模式
- [[../08-数值精度量化/01-训练精度选择]] - FP16/BF16 混合精度策略与数值稳定性
- [[../03-SFT与LoRA/02-LoRA原理]] - LoRA 原理、初始化策略与效果对比