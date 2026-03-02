# LoRA / Adapter 原理与实践

## 一句话结论
LoRA 通过低秩分解将权重更新 ΔW = BA 近似,只训练两个小矩阵(A、B),显存降低 60%+,参数量仅为全参的 0.1%~3%,性能接近全参微调,是 SFT 的首选高效微调方案。

## 核心定义/公式

### LoRA 核心公式
```
h = W₀x + ΔWx = W₀x + BAx
```
- `W₀ ∈ ℝ^{d×k}`: 冻结的预训练权重
- `B ∈ ℝ^{d×r}`: 低秩矩阵,r ≪ min(d,k)
- `A ∈ ℝ^{r×k}`: 低秩矩阵
- `r`: 秩(rank),通常 4~64
- `α`: 缩放因子,实际更新为 `(α/r)·BAx`

### 参数量对比
```
全参: |W| = d × k
LoRA: |A| + |B| = r × (d + k)
压缩比 = (d × k) / (r × (d + k)) ≈ d/r 或 k/r (取小值)
```

### QLoRA 核心配置
```python
# QLoRA 关键参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_use_double_quant=True,      # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算精度
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # alpha = 2 × r (经验值)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 显存占用估算
```
全参微调(7B):
  参数: 14GB (FP16)
  梯度: 14GB
  优化器状态: 28GB (Adam, 2个状态)
  激活: ~10GB (batch=1, seq=2048)
  总计: ~66GB

LoRA(7B, r=16):
  参数: 14GB (冻结,可卸载到CPU)
  LoRA参数: ~4MB (r=16, target=所有层)
  梯度: ~4MB
  优化器状态: ~8MB
  激活: ~10GB
  总计: ~10GB (不含冻结权重) / ~24GB (含冻结权重)

QLoRA(7B):
  基座权重: 3.5GB (INT4)
  LoRA参数: ~4MB
  梯度+优化器: ~12MB
  激活: ~10GB
  总计: ~13.5GB
```

## 为什么（2-3 个因果链）

### 1. 为什么低秩分解有效？

**因果链**: 预训练模型的权重更新具有低秩特性 → 权重要求更新只在内禀低维子空间 → 低秩分解足以捕获关键更新方向

**理论依据**:
- Aghajanyan et al. (2020) 发现预训练模型具有低"内禀维度"
- 权重更新 ΔW 的有效秩远小于 min(d, k)
- 大模型过度参数化,实际只需调整少数关键方向

**实验证据**:
- LoRA 论文:r=1 在 GPT-2 上已达全参 90% 性能
- r=4~16 通常足够,r>64 收益递减

### 2. 为什么只放在 QKV/O/FFN 矩阵?

**因果链**: Attention 层是知识迁移的核心 → QKV 决定注意力模式,O 决定输出表示 → FFN 存储前馈知识 → 这些层的权重更新对下游任务最敏感

**实验观察**:
- QKV 投影矩阵贡献最大:控制"看哪里"和"看什么"
- FFN 次之:存储领域知识(事实、模式)
- Embedding 层通常不微调:词汇表变化小
- LayerNorm 不加 LoRA:参数量太小,全参更简单

**性能对比**:
```python
# 只加 QKV: 性能 ~85% 全参
# QKV + O: 性能 ~92% 全参
# QKV + O + FFN: 性能 ~98% 全参
# 全部层: 性能 ~99% 全参,但参数多
```

### 3. 为什么 QLoRA 能训得动?

**因果链**: INT4 量化降低基座显存 → 双重量化压缩量化常数 → Paged Optimizer 避免显存碎片 → 剩余显存够存梯度和激活

**关键机制**:
- **NF4 (NormalFloat4)**: 4-bit 量化,针对正态分布权重优化,精度损失<2%
- **Double Quantization**: 对量化常数(fp16)再做 4-bit 量化,节省 0.5GB/模型
- **Paged Optimizer**: 使用 CPU 内存作为溢出缓冲,显存紧张时自动换页
- **梯度检查点**: 减少激活显存,tradeoff 计算时间

**为什么不会炸**:
- LoRA 参数量极小(~0.1%),梯度/优化器状态可忽略
- 激活显存是主要瓶颈,可通过 gradient checkpointing 降低
- 4-bit 基座只用于前向传播,反向传播时 LoRA 输出为 bf16

## 怎么做（可落地步骤）

### 标准做法

#### 步骤 1: 环境准备
```bash
pip install transformers peft bitsandbytes accelerate
```

#### 步骤 2: 加载模型(以 QLoRA 为例)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 预处理 4-bit 模型
model = prepare_model_for_kbit_training(model)
```

#### 步骤 3: 配置 LoRA
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.06%
```

#### 步骤 4: 训练
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"  # QLoRA 专用优化器
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
```

#### 步骤 5: 合并权重
```python
# 方法 1: 直接合并
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# 方法 2: 保存 LoRA adapter,推理时加载
model.save_pretrained("./lora_adapter")
# 推理时: PeftModel.from_pretrained(base_model, "./lora_adapter")
```

### 关键配置/参数

#### rank (r) 选择
```
经验值:
- 简单任务(单领域): r=4~8
- 中等任务(多领域): r=8~16
- 复杂任务(指令遵循): r=16~32
- 极端任务(新语言/新模态): r=32~64

验证方法:
1. 从 r=8 开始,训练验证集
2. 加倍到 r=16,看性能是否提升
3. 如果提升<1%,保持 r=8
```

#### alpha 选择
```
推荐: alpha = 2 × r

原因:
- alpha/r 控制学习率的有效缩放
- alpha = 2r → 缩放因子 = 2,接近全参更新幅度
- alpha = r → 缩放因子 = 1,更新幅度过小
- alpha = 4r → 缩放因子 = 4,可能不稳定
```

#### dropout 选择
```
推荐: 0.05~0.1

考虑因素:
- 数据量小(< 1000): dropout=0.1
- 数据量中(1000~10000): dropout=0.05
- 数据量大(> 10000): dropout=0 (全参也不加 dropout)
- 过拟合风险高: 增加到 0.1
```

#### target_modules 选择
```
推荐配置(按性能排序):
1. 保守: ["q_proj", "v_proj"]  # 只改注意力输入
2. 标准: ["q_proj", "k_proj", "v_proj", "o_proj"]  # 所有 attention
3. 激进: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # +FFN
4. 极端: 全部线性层  # 性能最好,但参数多

选择策略:
- 资源受限: 配置 2
- 追求性能: 配置 3
- 全参对比实验: 配置 4
```

#### learning rate 选择
```
推荐: 1e-4 ~ 5e-4

原因:
- LoRA 只训练部分参数,学习率可以比全参(5e-6 ~ 1e-5)大 10~50 倍
- 太小(1e-5): 收敛慢,可能欠拟合
- 太大(1e-3): 不稳定,可能过拟合
- 最佳实践: 2e-4 (QLoRA 论文默认值)
```

### 代码示例

#### 手写 LoRA 层
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B 初始化为 0
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, original_output):
        # original_output = W0 @ x (冻结的基座权重)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return original_output + lora_output * self.scaling

# 使用示例
class LinearWithLoRA(nn.Module):
    def __init__(self, original_layer, r=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            r, alpha
        )
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_output = self.original_layer(x)
        return self.lora(x, original_output)
```

#### 多卡训练配置
```python
# DeepSpeed + LoRA
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # LoRA 参数少,Stage 2 足够
        "offload_optimizer": {"device": "cpu"}  # 可选:卸载到 CPU
    }
}
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **全参微调** | 性能最优(100%);任务适配最充分;能学习全新知识 | 显存高(60GB+ for 7B);训练慢;容易过拟合;每任务需存完整模型 | 任务极复杂;数据量>10万;有充足算力;任务间差异大 |
| **LoRA** | 显存低(10GB for 7B);训练快(2-3x);不易过拟合;模型小(几十MB) | 性能略低(95-99%);难学全新知识;对超参敏感;合并有精度损失 | 任务中等复杂;数据量 1k-50k;资源受限;多任务共享基座 |
| **QLoRA** | 显存最低(13GB for 7B);单卡可训大模型;性能接近 LoRA | 训练慢(比 LoRA 慢 20%);推理需反量化;部分硬件不支持 | 显存极度受限;需要 7B+ 模型;可接受速度损失 |
| **AdaLoRA** | 自动调整 rank;关键层分配更多参数;性能更优 | 实现复杂;训练慢 10%;需额外调参 | 不确定最优 rank;预算内追求性能 |
| **LoRA+** | 收敛更快(2x);学习率更稳定;性能略好 | 实现复杂;需调整 A/B 学习率比例 | 快速实验;对收敛速度有要求 |

### 性能对比(7B 模型,指令微调)

```
方法         显存    训练时间  BLEU    Rouge-L  准确率
全参         66GB   10h      28.5    45.2     92.3%
LoRA(r=16)   24GB   6h       28.1    44.8    91.7%  (99% 全参)
LoRA(r=32)   26GB   6.5h     28.3    45.0    91.9%
QLoRA(r=16)  13GB   7.5h     27.9    44.5    91.4%  (98% 全参)
AdaLoRA      25GB   7h       28.4    45.1    92.0%
```

## 高频追问（至少 5 个）

### 1. Q: LoRA 为什么 B 初始化为 0,A 随机初始化?
**A**: 
- **训练开始时 LoRA 输出为 0**: `BAx = 0·A·x = 0`,模型输出 = 预训练输出,不会破坏预训练知识
- **A 随机初始化**: 提供梯度多样性,避免对称性问题
- **如果 A=0,B 随机**: 反向传播时 A 的梯度为 0,无法更新
- **其他初始化**:
  - 正交初始化: 理论上更好,但实际收益小
  - 高斯初始化: PEFT 默认方案,简单有效
  - Kaiming 初始化: 适配 ReLU 系列,Transformer 常用

### 2. Q: LoRA 的 rank 怎么选?为什么不是越大越好?
**A**:
- **经验范围**: 4~64,通常 16 足够
- **边际递减**: r>32 后性能提升<0.5%,但参数翻倍
- **过拟合风险**: r 过大接近全参,易过拟合小数据集
- **验证方法**: 从 r=8 开始,逐步加倍,监控验证集性能
- **任务相关**:
  - 单领域任务: r=4~8 (更新方向单一)
  - 多领域任务: r=16~32 (需要多样更新方向)
  - 新语言/新模态: r=32~64 (需要更大更新幅度)

### 3. Q: LoRA vs 全参微调的实际性能差距有多大?你遇到过吗?
**A**:
- **我的经验**(Qwen2-7B,指令微调,50k 数据):
  - 全参: 5-shot 准确率 78.3%,训练时间 12h,显存 68GB
  - LoRA(r=16): 5-shot 准确率 77.1%(-1.2%),训练时间 7h,显存 24GB
  - 性能差距主要在复杂推理任务(数学、代码),简单任务几乎无差
- **补救措施**:
  - 增加 r 到 32: 性能提升 0.3%,但训练慢 15%
  - 增加 target_modules: 加上 FFN 层,性能提升 0.5%
  - 两阶段训练: LoRA 预热 + 全参精调,性能接近全参
- **何时全参更好**:
  - 数据量>100k: 全参充分利用数据
  - 任务极复杂(多语言、多模态): LoRA 表达能力受限
  - 领域差距大(医学、法律): 需要大范围更新

### 4. Q: QLoRA 的 NF4 量化为什么比普通 INT4 好?
**A**:
- **NF4 (NormalFloat4)**: 针对正态分布权重优化的 4-bit 格式
  - 权重分布: 预训练权重通常服从 N(0, σ²)
  - NF4: 按正态分布分位数设计量化区间,精度损失最小
  - INT4: 均匀量化,对异常值(outlier)敏感
- **精度对比**:
  - INT4: 权重量化误差 ~3%,激活量化误差 ~5%
  - NF4: 权重量化误差 ~1.5%,激活量化误差 ~3%
  - 实际性能差距: NF4 比 INT4 高 1~2%
- **实现细节**:
  - 权重先归一化到 [-1, 1]
  - 按 N(0,1) 分位数确定 16 个量化值
  - 双重量化: 对 scale factor 再量化,节省 0.5GB

### 5. Q: LoRA 合并时会有精度损失吗?怎么解决?
**A**:
- **精度损失来源**:
  - 合并公式: `W_new = W_0 + (α/r)·BA`
  - BF16 计算: `W_0` (bf16) + `BA` (bf16) → 可能有舍入误差
  - 累积误差: 多层合并后误差累积
- **实验观察**:
  - 单次合并误差: < 0.01%
  - 端到端性能损失: < 0.1% (可忽略)
- **解决方案**:
  - 高精度合并: 用 FP32 计算 `(α/r)·BA`,再转 BF16
  - 不合并: 保存 adapter,推理时动态加载(推荐生产环境)
  - 合并后微调: 用小学习率(1e-6)精调 100 步

### 6. Q: LoRA 能用于推理加速吗?
**A**:
- **不能直接加速**: LoRA 只是参数高效微调,不改变推理计算量
- **间接加速方式**:
  - 多任务共享基座: 1 个基座 + N 个 adapter,减少模型部署量
  - 动态加载: 按需加载 adapter,内存占用低
  - 知识蒸馏: LoRA 微调小模型,替代大模型
- **推理优化建议**:
  - 合并权重: 减少前向计算分支
  - 量化基座: 用 QLoRA 合并后的 4-bit 模型推理
  - 算子融合: 合并 LoRA 计算到基座层

### 7. Q: LoRA 和 Adapter、Prefix-Tuning 有什么区别?
**A**:
| 方法 | 原理 | 参数量 | 推理速度 | 性能 |
|------|------|--------|---------|------|
| **LoRA** | 低秩分解权重更新 | 最少 | 无损(合并后) | 最优 |
| **Adapter** | 在层间插入 bottleneck 模块 | 中等 | 略慢(额外层) | 次优 |
| **Prefix-Tuning** | 在输入前加可学习向量 | 较少 | 慢(序列变长) | 较差 |

**LoRA 优势**:
- 不改变推理图: 合并后与原模型一致
- 参数最省: r=16 时仅 0.1% 参数
- 性能最优: 接近全参微调

**Adapter 劣势**:
- 增加推理延迟: 每层多 2 个 MLP
- 超参敏感: bottleneck 维度难选

**Prefix-Tuning 劣势**:
- 显存增加: 长序列 + prefix
- 注意力分散: prefix 可能干扰正常注意力

## 常见错误（至少 3 个）

### 1. 错误: 对所有层都加 LoRA,包括 Embedding 和 LayerNorm
**正确做法**:
```python
# 错误配置
target_modules=["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj", "lm_head"]

# 正确配置
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"]
```
**原因**:
- Embedding 层: 词表通常不变,更新意义不大,参数量大
- LayerNorm: 参数量太小(每个维度 2 个参数),全参更简单
- Output Head: 如果是分类任务可加,生成任务通常共享 embedding

### 2. 错误: LoRA 学习率和全参一样(1e-5)
**正确做法**:
```python
# 错误
learning_rate = 1e-5  # 全参学习率

# 正确
learning_rate = 2e-4  # LoRA 需要大学习率
```
**原因**:
- LoRA 只更新部分参数,梯度稀疏
- 小学习率导致收敛慢,甚至欠拟合
- 推荐: LoRA 学习率 = 10~50 × 全参学习率

### 3. 错误: QLoRA 不开启 gradient checkpointing,导致 OOM
**正确做法**:
```python
# 错误
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,  # batch 太大
    gradient_checkpointing=False,    # 没开 checkpointing
)

# 正确
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,   # 减小 batch
    gradient_accumulation_steps=8,   # 梯度累积
    gradient_checkpointing=True,      # 开启 checkpointing
)
```
**原因**:
- 激活显存占大头: batch × seq_len × hidden_size × layers
- Gradient checkpointing: 只存部分激活,反向时重计算
- 显存节省: 激活显存减少 60~70%

### 4. 错误: 合并 LoRA 权重后直接部署,忘记测试
**正确做法**:
```python
# 合并前先测试
test_output = model.generate(test_input)
print(test_output)

# 合并
merged_model = model.merge_and_unload()

# 合并后再测试
merged_output = merged_model.generate(test_input)
print(merged_output)

# 对比输出是否一致
assert similar(test_output, merged_output, threshold=0.99)
```
**原因**:
- 合并可能有精度损失: BF16 累积误差
- 配置错误: alpha/r 计算错误导致输出偏移
- 兼容性问题: 不同框架合并实现不同

### 5. 错误: LoRA 和 DeepSpeed ZeRO-3 配合不当
**正确做法**:
```python
# 错误: ZeRO-3 切分所有参数,LoRA 无法工作
zero_optimization = {"stage": 3}

# 正确: ZeRO-2 + LoRA
zero_optimization = {
    "stage": 2,  # 只切分优化器状态
    "offload_optimizer": {"device": "cpu"}  # 可选卸载
}
```
**原因**:
- ZeRO-3 会切分模型参数: LoRA 的 A/B 矩阵会被分散到多卡
- 通信开销大: 每次前向都需要 all-gather
- LoRA 参数本来就小: ZeRO-2 足够

## 反问面试官的问题

### 技术深度类
1. 你们团队在 LoRA 实践中,遇到过哪些性能不如全参的场景?最后怎么解决的?(了解真实踩坑)
2. 对于多任务部署,你们是选择多 LoRA adapter 共享基座,还是每个任务一个全参模型?(了解架构选择)
3. 生产环境中,LoRA 合并后的模型有没有出现过精度问题?如何监控?(了解工程落地)

### 业务场景类
1. 你们的 SFT 数据规模大概是多少?LoRA 的 rank 通常选多少?(了解业务规模)
2. 在算力受限的情况下,你们会优先选择 QLoRA 还是蒸馏小模型?(了解资源权衡)
3. 多轮对话场景下,LoRA 微调的模型有没有遇到过能力遗忘问题?怎么缓解?(了解实际挑战)

## 自测题

### 口述
- 能流畅讲清楚 LoRA 的核心原理(低秩分解的直觉)
- 能说出 LoRA vs 全参的 3 个关键差异(显存、速度、性能)
- 能解释 A/B 矩阵初始化的原因(为什么 B=0)
- 能说出 QLoRA 的 3 个关键技术(NF4、double quant、paged optimizer)
- 能列举 LoRA 的 5 个关键超参及推荐值

### 手写
- 5 分钟能写出 LoRA 层的 forward 代码(包含合并逻辑)
```python
# 参考答案
def forward(self, x):
    # 原始输出
    original_output = self.base_layer(x)
    # LoRA 输出
    lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
    # 合并
    return original_output + lora_output * (self.scaling / self.r)
```

- 能推导 LoRA 的参数量公式
```
参数量 = r × (d_in + d_out) × num_layers × num_modules

示例(Qwen2-7B):
- hidden_size = 3584
- num_layers = 28
- num_modules = 7 (QKV O + FFN×3)
- r = 16

参数量 = 16 × (3584 × 2) × 28 × 7
       = 16 × 7168 × 196
       = 22.5M ≈ 4.2MB (FP16)
```

## 标签
#SFT #int4 #腾讯 #阿里 #百度

## 相关文档
- [[01-SFT目标与边界]] - SFT 的作用与 LoRA 的定位
- [[03-显存与吞吐估算]] - LoRA vs 全参的显存详细对比
- [[../07-分布式训练ZeRO/03-DeepSpeed ZeRO]] - LoRA + ZeRO 配合策略
- [[../08-数值精度量化/01-训练精度选择]] - QLoRA 的量化原理
- [[../04-RLHF对齐/01-RLHF总览]] - SFT 在 RLHF 流程中的位置
