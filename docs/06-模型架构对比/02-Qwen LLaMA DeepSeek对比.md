# Qwen / LLaMA / DeepSeek 对比框架

## 一句话结论
Qwen 与 LLaMA 架构相似（Dense + GQA + SwiGLU + RoPE），差异在中文数据与 tokenizer 优化；DeepSeek 核心创新在 MoE + MLA + GRPO，用架构创新换取推理效率与训练稳定性。

## 核心定义/公式

### 三大模型架构对比总览

| 维度 | LLaMA 系列 | Qwen 系列 | DeepSeek 系列 |
|------|-----------|----------|--------------|
| **架构类型** | Dense | Dense（Qwen-MoE 有 MoE 版本） | MoE（DeepSeek-V2/V3） + Dense（V1） |
| **Attention** | GQA（LLaMA2/3）<br>MHA（LLaMA1） | GQA | MLA（Multi-Head Latent Attention） |
| **位置编码** | RoPE | RoPE + YaRN（长上下文） | RoPE |
| **归一化** | RMSNorm | RMSNorm | RMSNorm |
| **激活函数** | SwiGLU | SwiGLU | SwiGLU |
| **对齐算法** | PPO/DPO | PPO/DPO | GRPO |
| **训练数据** | 英文为主（2T tokens） | 多语言（中英为主，3T+ tokens） | 多语言（中英为主，14.8T tokens V3） |
| **Tokenizer** | SentencePiece BPE | 基于 BPE 的多语言 tokenizer | 自研 tokenizer |

### 关键架构创新

#### 1. GQA (Grouped-Query Attention) - LLaMA/Qwen

```python
# GQA: 头分组，组内共享 KV
# LLaMA2/3: 32 头 → 8 组 KV（节省 4 倍显存）
# Qwen: 类似配置

class GQA:
    """
    num_heads = 32
    num_groups = 8  # 每组 4 个头共享 KV
    KV cache 显存: batch × seq_len × 2 × num_groups × head_dim
    相比 MHA 节省: num_heads / num_groups 倍
    """
    def __init__(self, d_model, num_heads, num_groups):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        
        # Q: 每个头独立
        self.W_Q = nn.Linear(d_model, num_heads * head_dim)
        # K/V: 按组共享
        self.W_K = nn.Linear(d_model, num_groups * head_dim)
        self.W_V = nn.Linear(d_model, num_groups * head_dim)
```

#### 2. MLA (Multi-Head Latent Attention) - DeepSeek

```python
# MLA: KV 压缩到潜在空间，推理时解压
# 核心思想：K/V 先压缩到低维潜在向量，再解压回高维

class MLA:
    """
    DeepSeek-V2/V3 核心创新
    - KV 压缩率: ~90%（相比 MHA）
    - 推理加速: 显著降低 KV cache 访存
    
    公式：
    K_compressed = x @ W_DK  # 压缩到低维 d_c
    K = K_compressed @ W_UK  # 解压回高维 d_h
    V_compressed = x @ W_DV
    V = V_compressed @ W_UV
    """
    def __init__(self, d_model, d_c=512, d_h=2048):
        # 压缩矩阵（下投影）
        self.W_DK = nn.Linear(d_model, d_c)  # d_c << d_h
        self.W_DV = nn.Linear(d_model, d_c)
        
        # 解压矩阵（上投影）
        self.W_UK = nn.Linear(d_c, d_h)
        self.W_UV = nn.Linear(d_c, d_h)
        
    def forward(self, x):
        # 推理时只缓存压缩后的 KV
        K_c = self.W_DK(x)  # [batch, seq, d_c]
        V_c = self.W_DV(x)
        
        # 计算时解压
        K = self.W_UK(K_c)  # [batch, seq, d_h]
        V = self.W_UV(V_c)
        
        # Attention 计算...
        return K, V
```

**MLA 显存收益**：
```python
# MHA: batch × seq_len × 2 × num_heads × head_dim
# MLA: batch × seq_len × 2 × d_c (压缩维度)

# DeepSeek-V2 示例
# MHA: 128 layers × 32 heads × 128 dim = 524,288 per token per layer
# MLA: 128 layers × 512 dim = 65,536 per token per layer
# 压缩率: ~87.5%
```

#### 3. MoE (Mixture of Experts) - DeepSeek

```python
# DeepSeekMoE: 细粒度专家 + 共享专家

class DeepSeekMoE:
    """
    DeepSeek-V3:
    - 总参数: 671B
    - 激活参数: 37B（top-6 routing）
    - 专家数: 256（细粒度）
    - 共享专家: 1（始终激活）
    - 路由专家: top-6 from 255
    
    核心创新:
    1. 细粒度专家: 每个专家更小，路由更灵活
    2. 共享专家: 捕捉通用知识，稳定训练
    """
    def __init__(self, d_model, num_experts=256, top_k=6, 
                 shared_experts=1):
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        
        # 路由专家
        self.experts = nn.ModuleList([
            FFN(d_model, d_ff) for _ in range(num_experts - shared_experts)
        ])
        
        # 共享专家（始终激活）
        self.shared = nn.ModuleList([
            FFN(d_model, d_ff) for _ in range(shared_experts)
        ])
        
        # 路由网络
        self.router = nn.Linear(d_model, num_experts - shared_experts)
        
    def forward(self, x):
        batch, seq, d = x.shape
        x_flat = x.view(-1, d)
        
        # 路由得分
        router_logits = self.router(x_flat)  # [batch*seq, num_experts]
        
        # Top-k 选择
        topk_values, topk_indices = torch.topk(router_logits, self.top_k)
        topk_probs = F.softmax(topk_values, dim=-1)
        
        # 路由专家输出
        routed_output = torch.zeros_like(x_flat)
        for i, (probs, indices) in enumerate(zip(topk_probs, topk_indices)):
            for prob, idx in zip(probs, indices):
                expert_out = self.experts[idx](x_flat[i:i+1])
                routed_output[i] += prob * expert_out.squeeze(0)
        
        # 共享专家输出（始终激活）
        shared_output = sum(expert(x_flat) for expert in self.shared)
        
        # 合并
        output = routed_output + shared_output
        return output.view(batch, seq, d)
```

### Tokenizer 对比

```python
# LLaMA Tokenizer: SentencePiece BPE
# - 词表大小: 32K (LLaMA2)
# - 中文支持: 较差，中文被切分成多个字节
# - 例: "你好" → ["你", "好"] 或更细粒度

# Qwen Tokenizer: 基于 BPE 的多语言优化
# - 词表大小: 151,936 (Qwen2)
# - 中文支持: 好，中文词直接编码
# - 例: "你好世界" → ["你好", "世界"] 或更优切分

# DeepSeek Tokenizer: 自研
# - 词表大小: ~100K
# - 中文支持: 优化
# - 设计哲学: 平衡压缩率与词表大小
```

## 为什么（2-3 个因果链）

### 1. **为什么 DeepSeek 选择 MoE + MLA 架构**

**因果链**：
- **现象**：大模型推理成本高，长上下文显存压力大
  - Dense 671B 模型推理：显存占用巨大，batch size 受限
  - 长上下文（128K）KV cache 显存爆炸
  - 商业部署成本高，影响可及性

- **根因**：Dense 架构的"计算-参数"强耦合
  - Dense 模型：参数量 ∝ 计算量 ∝ 推理成本
  - 无法在保持能力的同时降低推理成本
  - KV cache 显存与序列长度线性相关，长文本场景受限

- **结果**：MoE + MLA 解耦参数量与推理成本
  - **MoE**：参数量大（671B），激活参数小（37B），推理快
  - **MLA**：KV cache 压缩 87.5%，长上下文显存友好
  - **GRPO**：训练稳定，适合 MoE 这种复杂架构

**量化收益**：
```python
# DeepSeek-V3 vs Dense 671B

# 推理成本对比
dense_cost = {
    "params": "671B",
    "active_params": "671B",  # 全部激活
    "kv_cache_per_token": "524K per layer",  # MHA
}

moe_cost = {
    "params": "671B",
    "active_params": "37B",  # top-6 routing
    "kv_cache_per_token": "65K per layer",  # MLA
}

# 推理加速: ~18x (671B / 37B)
# KV cache 节省: ~87.5%
```

### 2. **为什么 Qwen 架构与 LLaMA 相似但效果更好**

**因果链**：
- **现象**：Qwen 在中文和多语言任务上表现优于 LLaMA
  - Qwen 中文 benchmark 领先
  - Qwen 多语言能力强
  - 架构差异不大，但效果差异明显

- **根因**：数据与 tokenizer 的"隐性优势"
  - **训练数据**：Qwen 中英数据配比优化，中文数据质量高
  - **Tokenizer**：词表更大（151K vs 32K），中文编码效率高
  - **长上下文**：YaRN + 窗口注意力，支持 128K 上下文

- **结果**：相同架构下，数据质量决定天花板
  - 架构相似（GQA + SwiGLU + RoPE）
  - 数据差异带来效果差异
  - Tokenizer 优化减少序列长度，提升训练效率

**Tokenizer 效率对比**：
```python
# 同一段中文文本
text = "人工智能正在改变世界，深度学习技术发展迅速。"

# LLaMA tokenizer (32K 词表)
llama_tokens = ["人工", "智能", "正在", "改变", "世", "界", ...]  # 20+ tokens
# 问题: 中文被过度切分，语义信息丢失

# Qwen tokenizer (151K 词表)
qwen_tokens = ["人工智能", "正在", "改变世界", ...]  # 12 tokens
# 优势: 中文词直接编码，压缩率高，语义完整

# 压缩率差异
# Qwen 比 LLaMA 少 ~40% tokens
# 训练效率提升，推理成本降低
```

### 3. **为什么 GRPO 比 PPO 更适合 DeepSeek**

**因果链**：
- **现象**：DeepSeek-V3 用 GRPO 替代 PPO 进行对齐
  - 训练稳定性更好
  - 无需独立奖励模型
  - 适合 MoE 这种复杂架构

- **根因**：MoE 架构的训练敏感性
  - MoE 有路由网络，梯度流动复杂
  - PPO 的 RM 偏差 + MoE 路由偏差 → 双重不稳定
  - GRPO 的组内相对比较天然抗偏差

- **结果**：GRPO 简化流程，提升稳定性
  - 移除 RM 训练阶段
  - 组内相对比较消除系统性偏差
  - 更适合大规模 MoE 训练

**DeepSeek 选择 GRPO 的权衡**：
```python
# PPO 流程（传统）
# Stage 1: SFT
# Stage 2: Train Reward Model (需要偏好数据)
# Stage 3: PPO with RM + KL constraint

# GRPO 流程（DeepSeek）
# Stage 1: SFT
# Stage 2: GRPO (组内相对比较，无需独立 RM)

# 优势
# 1. 少一个模型训练（RM）
# 2. 少一份偏好数据标注
# 3. 训练更稳定（无 RM 偏差累积）
# 4. 天然适合 MoE（组内比较抗路由偏差）

# 代价
# 1. 每步采样 G 个响应（通常 G=4）
# 2. 需要可靠评分器
```

## 怎么做（可落地步骤）

### 标准 Transformer 模型选择流程

#### Step 1: 场景分析与模型选型

```python
def select_model(requirements):
    """
    根据需求选择合适的模型架构
    """
    # 1. 语言支持
    if requirements["primary_language"] == "chinese":
        # 中文为主：优先 Qwen
        primary_choice = "Qwen"
        alternative = "DeepSeek"
    elif requirements["primary_language"] == "english":
        # 英文为主：优先 LLaMA
        primary_choice = "LLaMA"
        alternative = "Qwen"
    else:
        # 多语言：Qwen 或 DeepSeek
        primary_choice = "Qwen"
    
    # 2. 上下文长度
    if requirements["context_length"] >= 128000:
        # 超长上下文：Qwen (YaRN) 或 DeepSeek (MLA)
        candidates = ["Qwen-128K", "DeepSeek-V3"]
    elif requirements["context_length"] >= 32000:
        candidates = ["LLaMA-3-32K", "Qwen-32K", "DeepSeek-V2"]
    else:
        candidates = ["LLaMA-3", "Qwen", "DeepSeek"]
    
    # 3. 推理成本约束
    if requirements["inference_cost"] == "low":
        # 低成本推理：优先 MoE
        primary_choice = "DeepSeek-V3 (MoE, 37B active)"
        # 或者用小模型
        if requirements["model_size"] == "small":
            primary_choice = "Qwen-7B / LLaMA-3-8B"
    
    # 4. 微调需求
    if requirements["finetune"] == "full":
        # 全参微调：考虑 Dense 模型
        primary_choice = "LLaMA-3 or Qwen (Dense)"
    elif requirements["finetune"] == "lora":
        # LoRA 微调：都可以
        pass
    
    return primary_choice

# 使用示例
requirements = {
    "primary_language": "chinese",
    "context_length": 32000,
    "inference_cost": "medium",
    "finetune": "lora"
}
model = select_model(requirements)
# 输出: Qwen-32K
```

#### Step 2: 推理部署配置

```python
# 不同模型的推理配置

# LLaMA-3-70B
llama_config = {
    "model": "meta-llama/Meta-Llama-3-70B",
    "tensor_parallel_size": 4,  # 4 卡 TP
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,  # 默认 8K
    "trust_remote_code": True,
    
    # GQA 配置
    "num_heads": 64,
    "num_key_value_heads": 8,  # GQA: 8 组 KV
    
    # vLLM 部署
    "backend": "vLLM",
    "dtype": "auto",  # 自动选择 bf16
}

# Qwen2-72B
qwen_config = {
    "model": "Qwen/Qwen2-72B",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,  # 支持 32K（YaRN）
    
    # 长上下文扩展
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,  # 从 8K 扩展到 32K
    },
    
    # GQA
    "num_heads": 64,
    "num_key_value_heads": 8,
}

# DeepSeek-V3 (671B MoE)
deepseek_config = {
    "model": "deepseek-ai/DeepSeek-V3",
    "tensor_parallel_size": 8,  # 8 卡 TP
    "expert_parallel_size": 8,  # EP 并行
    "gpu_memory_utilization": 0.95,
    "max_model_len": 128000,  # 128K 上下文
    
    # MoE 配置
    "num_experts": 256,
    "num_experts_per_tok": 6,  # top-6 routing
    "num_shared_experts": 1,
    
    # MLA 配置
    "kv_lora_rank": 512,  # KV 压缩维度
    "qk_rope_head_dim": 64,
    
    # 推理优化
    "enable_prefix_caching": True,  # MLA 适合前缀缓存
    "enforce_eager": False,  # 使用 CUDA graph
}
```

#### Step 3: 微调配置（以 Qwen 为例）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Qwen LoRA 微调
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B",
    trust_remote_code=True
)

# LoRA 配置
lora_config = LoraConfig(
    r=64,  # LoRA rank
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Qwen 特定优化
# 1. 中文数据质量高，可直接微调
# 2. Tokenizer 对中文友好，序列长度可控
# 3. 支持 YaRN 扩展，可训练长上下文任务
```

#### Step 4: vLLM / TensorRT-LLM 适配

```python
# vLLM 部署不同模型

from vllm import LLM, SamplingParams

# 1. LLaMA 部署
llama_llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)

# 2. Qwen 部署（长上下文）
qwen_llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,
    max_model_len=32768,  # 32K 上下文
    rope_scaling={"type": "yarn", "factor": 4.0},
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)

# 3. DeepSeek 部署（MoE + MLA）
# 注意：DeepSeek-V3 需要 vLLM >= 0.6.0 支持 MLA
deepseek_llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    max_model_len=128000,  # 128K
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    # MLA 特定配置
    enable_prefix_caching=True,
)

# 推理示例
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

prompts = ["请解释什么是深度学习"]
outputs = qwen_llm.generate(prompts, sampling_params)
```

### 关键配置/参数

| 参数 | LLaMA-3 | Qwen2 | DeepSeek-V3 | 说明 |
|------|---------|-------|-------------|------|
| `max_model_len` | 8192 (默认) | 32768 (YaRN) | 128000 | 上下文长度 |
| `num_heads` | 64 (70B) | 64 (72B) | 128 | 注意力头数 |
| `num_kv_heads` | 8 (GQA) | 8 (GQA) | N/A (MLA) | KV 头数（GQA） |
| `kv_lora_rank` | N/A | N/A | 512 | MLA 压缩维度 |
| `num_experts` | N/A | N/A | 256 | MoE 专家数 |
| `num_experts_per_tok` | N/A | N/A | 6 | Top-k routing |
| `tensor_parallel_size` | 4 (70B) | 4 (72B) | 8 (671B) | TP 并行度 |
| `rope_scaling` | 无 | yarn, factor=4.0 | 无 | 位置编码扩展 |

### 代码示例：模型对比实验

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compare_models(prompts, models=["Qwen/Qwen2-7B", "meta-llama/Meta-Llama-3-8B"]):
    """
    对比不同模型在同一任务上的表现
    """
    results = {}
    
    for model_name in models:
        print(f"\n=== Testing {model_name} ===")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Tokenizer 效率对比
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            print(f"Prompt: {prompt[:30]}...")
            print(f"  Tokens: {len(tokens)}")
            print(f"  Token IDs: {tokens[:10]}...")
        
        # 生成对比
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results[model_name] = {
                "prompt": prompt,
                "response": response,
                "num_tokens": len(tokens)
            }
    
    return results

# 测试中文 tokenizer 效率
test_prompts = [
    "人工智能正在改变世界",  # 中文
    "Deep learning is transforming AI",  # 英文
    "深度学习技术发展迅速，应用广泛",  # 长中文
]

results = compare_models(test_prompts)

# 分析
# Qwen: 中文 tokens 少，编码效率高
# LLaMA: 中文 tokens 多，被切分
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **LLaMA (Dense + GQA)** | • 生态完善，开源工具链丰富<br>• 英文能力强<br>• 微调资源多 | • 中文支持较弱<br>• 上下文长度受限（默认 8K）<br>• Dense 架构推理成本高 | • 英文为主的应用<br>• 已有 LLaMA 微调经验<br>• 需要丰富生态支持 |
| **Qwen (Dense + GQA + YaRN)** | • 中文和多语言能力强<br>• 长上下文支持（32K-128K）<br>• Tokenizer 高效<br>• 数据质量高 | • 生态不如 LLaMA 完善<br>• 推理成本与 Dense 相同<br>• 长上下文需特殊配置 | • 中文为主的应用<br>• 需要长上下文<br>• 多语言场景 |
| **DeepSeek (MoE + MLA + GRPO)** | • 推理效率高（37B active）<br>• 显存占用低（MLA 压缩）<br>• 长上下文支持（128K）<br>• 训练稳定（GRPO） | • MoE 训练复杂度高<br>• 需要更多 GPU（671B 总参数）<br>• 生态较新，工具链不完善<br>• MLA 需要特定优化 | • 追求推理效率<br>• 长上下文场景<br>• 大规模部署<br>• 有充足算力 |

### 模型规模与推理成本对比

```python
# 假设：7B Dense 作为基准

models_comparison = {
    "LLaMA-3-8B": {
        "params": "8B",
        "active_params": "8B",
        "relative_cost": 1.14,  # 相对 7B
        "max_context": "8K",
    },
    "LLaMA-3-70B": {
        "params": "70B",
        "active_params": "70B",
        "relative_cost": 10.0,
        "max_context": "8K",
    },
    "Qwen2-7B": {
        "params": "7B",
        "active_params": "7B",
        "relative_cost": 1.0,  # 基准
        "max_context": "32K (YaRN)",
    },
    "Qwen2-72B": {
        "params": "72B",
        "active_params": "72B",
        "relative_cost": 10.3,
        "max_context": "32K (YaRN)",
    },
    "DeepSeek-V3": {
        "params": "671B",
        "active_params": "37B",  # MoE
        "relative_cost": 5.3,  # 37B / 7B
        "max_context": "128K (MLA)",
    },
}

# DeepSeek-V3 的优势
# - 参数量最大（671B），能力最强
# - 激活参数适中（37B），推理成本可控
# - MLA 支持 128K 上下文，显存友好
```

### 长上下文方案对比

| 方案 | LLaMA | Qwen | DeepSeek |
|------|-------|------|----------|
| **默认上下文** | 8K | 32K | 128K |
| **扩展方法** | RoPE scaling (不推荐) | YaRN | MLA |
| **显存占用** | 高（MHA/GQA） | 高（GQA） | 低（MLA 压缩） |
| **精度影响** | 扩展后下降明显 | YaRN 保持较好 | MLA 保持较好 |
| **实现复杂度** | 低 | 中 | 高 |

## 高频追问（至少 5 个）

### 1. Q: Qwen 和 LLaMA 架构基本一样，为什么 Qwen 中文效果更好？

**A**: 核心差异在数据与 tokenizer，而非架构：

**训练数据**：
- Qwen：中英数据配比优化，中文数据质量高，清洗细致
- LLaMA：英文为主（>80%），中文数据少且质量一般

**Tokenizer**：
- Qwen 词表 151K，中文词直接编码，压缩率高（中文 tokens 少 40%）
- LLaMA 词表 32K，中文被过度切分，语义信息丢失

**数据效率**：
```python
# 同样 3T tokens 训练
# Qwen: 中文数据占比高，中文能力强
# LLaMA: 英文数据占比高，英文能力强

# Tokenizer 效率差异
# Qwen 中文压缩率高 → 实际见到的中文数据更多
# LLaMA 中文切分碎 → 序列长度长，有效信息密度低
```

**结论**：架构相似度 90%，数据与 tokenizer 决定了语言能力差异。

### 2. Q: DeepSeek 的 MLA 和 GQA 有什么区别？为什么更优？

**A**: 

**GQA（LLaMA/Qwen）**：
- 组内共享 KV，减少存储
- 压缩率：`num_groups / num_heads`（如 8/32 = 25%）
- KV 维度不变，只是头数减少

**MLA（DeepSeek）**：
- KV 压缩到潜在空间，推理时解压
- 压缩率：`d_c / (num_heads × head_dim)`（如 512 / 16384 = 3%）
- KV 维度降低，通过低秩分解实现

**对比**：
```python
# GQA (LLaMA-3-70B)
num_heads = 64
num_kv_heads = 8
head_dim = 128
kv_per_token = 8 * 128 = 1024  # 维度

# MLA (DeepSeek-V3)
kv_lora_rank = 512
kv_per_token = 512  # 压缩后的维度

# 压缩率对比
gqa_compression = 1024 / (64 * 128) = 12.5%
mla_compression = 512 / (64 * 128) = 6.25%

# MLA 压缩率更高，且表达能力更强（低秩分解保留信息）
```

**为什么 MLA 更优**：
1. **压缩率更高**：MLA 可达 90%+，GQA 通常 75%
2. **表达能力更强**：低秩分解比直接共享更灵活
3. **长上下文友好**：压缩后显存占用低，支持 128K

### 3. Q: DeepSeek 为什么用 GRPO 不用 PPO？

**A**: 

**核心原因**：MoE 架构 + 训练稳定性需求

**MoE 的敏感性**：
```python
# MoE 有路由网络，梯度流动复杂
# PPO: RM 偏差 + 路由偏差 → 双重不稳定
# GRPO: 组内相对比较 → 消除系统性偏差
```

**GRPO 优势**：
1. **无需独立 RM**：省去 RM 训练，简化流程
2. **抗偏差**：组内相对比较消除评分器偏差
3. **训练稳定**：适合 MoE 这种复杂架构
4. **工程简洁**：少一个模型，少一套流程

**PPO 劣势**：
1. **RM 偏差累积**：Policy 学会欺骗 RM
2. **训练不稳定**：需要大量调参
3. **流程复杂**：SFT → RM → PPO 三阶段

**DeepSeek 的权衡**：
- 采样成本增加（G=4），但换来稳定性
- 适合大规模 MoE 训练
- 671B 模型，稳定性 > 采样效率

详见 [[04-GRPO算法]]

### 4. Q: Qwen 的长上下文是怎么做的？和 LLaMA 有什么区别？

**A**: 

**Qwen 长上下文方案**：
1. **YaRN (Yet another RoPE extension)**：频率缩放 + 温度调整
2. **窗口注意力**：限制注意力范围，减少显存
3. **训练策略**：长短文本混合训练

**LLaMA 长上下文**：
- 默认 8K，官方不支持扩展
- 社区方案：RoPE linear scaling（效果一般）

**YaRN vs Linear Scaling**：
```python
# Linear Scaling（简单）
# 直接缩放 RoPE 频率
theta_new = theta / scale_factor
# 问题：短上下文能力下降，长上下文精度损失

# YaRN（Qwen）
# 频率缩放 + 温度调整 + 动态插值
# 优势：保持短上下文能力，长上下文效果好
```

**Qwen 的实践**：
```python
# Qwen2-72B 长上下文配置
rope_scaling = {
    "type": "yarn",
    "factor": 4.0,  # 从 8K 扩展到 32K
    "original_max_position_embeddings": 8192,
}
```

### 5. Q: MoE 训练有哪些难点？DeepSeek 怎么解决？

**A**: 

**MoE 训练难点**：
1. **负载不均**：某些专家过载，某些专家闲置
2. **训练不稳定**：路由梯度 + 专家梯度交织
3. **通信开销**：专家并行时 all-to-all 通信
4. **路由坍缩**：路由网络只选少数专家

**DeepSeek 解决方案**：

**1. 负载均衡 Loss**：
```python
# DeepSeekMoE 负载均衡
def auxiliary_loss(router_probs, expert_indices):
    """
    router_probs: [batch, seq, num_experts]
    expert_indices: [batch, seq, top_k]
    """
    # 专家使用频率
    expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]
    
    # 路由熵
    route_entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum()
    
    # 负载均衡 loss
    balance_loss = (expert_usage ** 2).sum() * num_experts
    
    return balance_loss - 0.1 * route_entropy
```

**2. 共享专家**：
```python
# DeepSeek-V3: 1 个共享专家 + 255 个路由专家
# 共享专家始终激活，捕捉通用知识
# 稳定训练，避免路由坍缩
```

**3. 细粒度专家**：
```python
# 传统 MoE: 8-16 个大专家
# DeepSeekMoE: 256 个小专家（细粒度）
# 优势：路由更灵活，负载更均衡
```

**4. GRPO 稳定训练**：
```python
# GRPO 组内相对比较
# 消除评分器偏差，稳定 MoE 训练
# 详见 [[04-GRPO算法]]
```

### 6. Q: 实际项目中如何选择这三个模型？

**A**: 

**选择决策树**：
```python
def choose_model(scenario):
    # 1. 语言
    if scenario["language"] == "chinese":
        # 中文为主
        if scenario["need_long_context"] and scenario["context"] >= 64000:
            # 超长上下文（64K+）
            if scenario["inference_cost_sensitive"]:
                return "DeepSeek-V3 (MLA 显存友好)"
            else:
                return "Qwen-72B-128K (YaRN 支持好)"
        else:
            # 常规上下文
            if scenario["budget"] == "low":
                return "Qwen-7B / Qwen-14B"
            else:
                return "Qwen-72B"
    
    elif scenario["language"] == "english":
        # 英文为主
        if scenario["ecosystem_importance"] == "high":
            # 需要丰富生态
            return "LLaMA-3"
        else:
            return "Qwen (多语言备用)"
    
    else:
        # 多语言
        if scenario["inference_cost_sensitive"]:
            return "DeepSeek-V3 (MoE 高效)"
        else:
            return "Qwen (多语言平衡)"
    
    # 2. 特殊场景
    if scenario["task"] == "code":
        # 代码任务
        return "DeepSeek-Coder or Qwen-Coder"
    
    if scenario["task"] == "math":
        # 数学任务
        return "DeepSeek-V3 (MoE 推理强)"
```

**实践建议**：
- **快速验证**：Qwen-7B / LLaMA-3-8B
- **中文生产**：Qwen-72B
- **英文生产**：LLaMA-3-70B
- **追求效率**：DeepSeek-V3（MoE + MLA）
- **长上下文**：Qwen-128K 或 DeepSeek-V3

### 7. Q: 这三个模型的 tokenizer 有什么差异？

**A**: 

**词表大小对比**：
- LLaMA-3: 128K（LLaMA-2 是 32K）
- Qwen2: 151,936
- DeepSeek: ~100K

**中文编码效率对比**：
```python
# 测试文本
text = "深度学习是人工智能的核心技术"

# LLaMA-3 (128K 词表)
# 中文支持改善，但仍不如 Qwen
llama_tokens = tokenizer_llama.encode(text)
# 结果: ["深", "度", "学", "习", ...] (约 15 tokens)

# Qwen2 (151K 词表)
# 中文直接编码
qwen_tokens = tokenizer_qwen.encode(text)
# 结果: ["深度", "学习", ...] (约 8 tokens)

# DeepSeek (100K 词表)
# 平衡压缩率与词表大小
deepseek_tokens = tokenizer_deepseek.encode(text)
# 结果: ["深度", "学习", ...] (约 9 tokens)
```

**选择建议**：
- **中文为主**：Qwen tokenizer 最优
- **英文为主**：LLaMA tokenizer 足够
- **多语言**：Qwen 或 DeepSeek
- **追求压缩率**：Qwen（词表大，压缩率高）

## 常见错误（至少 3 个）

### 1. **错误：认为 Qwen 架构与 LLaMA 完全相同**

**正确理解**：
- 架构层面：90% 相似（GQA + SwiGLU + RoPE + RMSNorm）
- 数据层面：完全不同（中英配比、数据质量、清洗策略）
- Tokenizer 层面：差异大（词表大小、中文编码效率）

**关键差异**：
```python
# 架构相似，效果差异来自：
# 1. 训练数据质量与配比
# 2. Tokenizer 设计（中文编码效率）
# 3. 长上下文策略（YaRN）

# 不要低估数据与 tokenizer 的重要性
# "数据决定天花板，架构决定下限"
```

### 2. **错误：DeepSeek 是 MoE，所以推理慢**

**正确理解**：
- MoE 的"参数量"大，但"激活参数"小
- DeepSeek-V3: 671B 总参数，37B 激活参数
- 推理速度取决于激活参数，而非总参数

**推理成本对比**：
```python
# Dense 70B vs MoE 671B (37B active)

# Dense 70B
# - 每个前向传播：所有 70B 参数都参与计算
# - 推理成本：70B FLOPs per token

# MoE 671B (DeepSeek-V3)
# - 每个前向传播：只有 37B 参数激活（top-6 routing）
# - 推理成本：37B FLOPs per token

# 结论：DeepSeek-V3 推理比 Dense 70B 更快
# 且能力更强（总参数 671B）
```

### 3. **错误：直接用 LLaMA 的长上下文扩展方法**

**正确做法**：
- LLaMA 官方不支持长上下文扩展
- 社区方案（Linear Scaling）效果一般，短上下文能力下降明显
- 如需长上下文，选择 Qwen（YaRN）或 DeepSeek（MLA）

**Qwen 的 YaRN 优势**：
```python
# 错误：对 LLaMA 用简单 linear scaling
rope_scaling = {"type": "linear", "factor": 4.0}
# 问题：短上下文能力下降，长上下文精度损失

# 正确：使用 Qwen 的 YaRN
rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
}
# 优势：保持短上下文能力，长上下文效果好
```

### 4. **错误：认为 MLA 只是简单的 KV 压缩**

**正确理解**：
- MLA 是低秩分解，而非简单压缩
- 通过潜在空间投影，保留表达能力
- 需要特定的推理优化（vLLM >= 0.6.0）

**MLA 原理**：
```python
# MLA 不是简单压缩
# 而是"压缩-解压"的低秩分解

# 错误理解：直接降维
# K_compressed = K[:, :512]  # 截断
# 这会丢失信息

# 正确理解：低秩分解
# K_compressed = x @ W_DK  # 压缩到低维
# K = K_compressed @ W_UK  # 解压回高维
# 通过低秩分解保留关键信息

# 类比：PCA 降维后再重建，而非简单截断
```

### 5. **错误：忽略 tokenizer 对训练效率的影响**

**正确做法**：
- Tokenizer 决定序列长度，影响训练效率
- 中文场景：Qwen tokenizer 更高效（tokens 少 40%）
- Tokenizer 选择需与训练数据语言匹配

**训练效率影响**：
```python
# 同样的中文数据
# LLaMA tokenizer: 1000 tokens
# Qwen tokenizer: 600 tokens (压缩 40%)

# 训练效率影响
# - 批次大小：tokens 少 → 可增大 batch size
# - 训练速度：序列短 → 计算快
# - 显存占用：tokens 少 → 显存低

# 结论：中文场景选 Qwen tokenizer 不仅效果好，训练也快
```

## 反问面试官的问题

### 技术深度类
1. **您团队在实际生产中，更倾向于选择 Dense 架构（如 Qwen）还是 MoE 架构（如 DeepSeek）？主要考量是什么？**
   - 了解实际工程决策，评估不同架构的适用性

2. **对于长上下文场景（如 64K+），您团队是否尝试过 MLA 或其他压缩方案？显存优化效果如何？**
   - 了解长上下文技术的实践情况

3. **DeepSeek 的 GRPO 在您的对齐实践中是否有应用？与 PPO/DPO 的效果对比如何？**
   - 了解对齐算法的实际选择

### 业务场景类
1. **在您的业务中，中文能力和英文能力哪个更重要？Qwen 和 LLaMA 的选择依据是什么？**
   - 了解业务语言需求，评估模型选择

2. **您团队是否关注推理成本？MoE 架构的推理加速在实际部署中收益有多大？**
   - 了解成本敏感性，评估架构选择

3. **对于多语言场景，您倾向于用统一的多语言模型（如 Qwen）还是分别部署不同语言模型？**
   - 了解多语言策略，评估部署方案

## 自测题

### 口述（能流畅讲清楚的知识点）
1. **三大模型核心差异**：
   - LLaMA：Dense + GQA，英文强，生态好
   - Qwen：Dense + GQA + YaRN，中文强，长上下文
   - DeepSeek：MoE + MLA + GRPO，推理高效，训练稳定

2. **MLA vs GQA**：
   - GQA：组内共享 KV，压缩率 ~75%
   - MLA：低秩分解，压缩率 ~90%+，表达能力更强

3. **为什么 Qwen 中文效果好**：
   - 数据质量高，中英配比优化
   - Tokenizer 词表大，中文编码效率高
   - 长上下文支持好（YaRN）

### 手写（5 分钟能写出的代码/公式）
1. **GQA 配置**：
```python
# LLaMA-3-70B GQA 配置
num_heads = 64
num_kv_heads = 8  # 8 组 KV
head_dim = 128

# KV cache 显存（单层，FP16）
kv_memory = batch_size * seq_len * 2 * num_kv_heads * head_dim * 2  # bytes

# 相比 MHA 节省
savings = num_heads / num_kv_heads  # 8 倍
```

2. **MLA 压缩率计算**：
```python
# DeepSeek-V3 MLA 配置
num_heads = 128
head_dim = 128
kv_lora_rank = 512

# MHA KV 维度
mha_dim = num_heads * head_dim  # 16384

# MLA KV 维度
mla_dim = kv_lora_rank  # 512

# 压缩率
compression_ratio = mla_dim / mha_dim  # 3.1%
```

3. **MoE 推理成本**：
```python
# DeepSeek-V3
total_params = 671  # B
active_params = 37  # B (top-6 routing)
num_experts = 256
num_experts_per_tok = 6

# 相比 Dense 671B 的加速
speedup = total_params / active_params  # 18.1x

# 相比 Dense 70B 的成本
relative_cost = active_params / 70  # 0.53 (节省 47%)
```

## 标签

#架构 #Qwen #LLaMA #DeepSeek #MoE #MLA #GQA #GRPO #RoPE #YaRN #长上下文 #推理效率 #Tokenizer #中文优化 #阿里 #字节 #百度 #腾讯

## 相关文档

- [[01-主流架构共同点]] - Decoder-only + RMSNorm + SwiGLU + RoPE 的统一趋势
- [[03-MoE专题]] - MoE 原理、训练难点、DeepSeekMoE 创新点
- [[../01-Transformer基础/02-Attention机制]] - MHA / MQA / GQA / MLA 详解
- [[../04-RLHF对齐/04-GRPO算法]] - DeepSeek 选择 GRPO 的原因
- [[../05-长上下文/03-YaRN]] - Qwen 长上下文扩展方案
- [[../01-Transformer基础/04-Tokenizer与Embedding]] - Tokenizer 对比与选择

---

## 参考资源

1. **LLaMA 系列**
   - LLaMA: Open and Efficient Foundation Language Models (Meta, 2023)
   - LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Meta, 2023)
   - LLaMA 3 Technical Report (Meta, 2024)

2. **Qwen 系列**
   - Qwen Technical Report (Alibaba, 2023)
   - Qwen2 Technical Report (Alibaba, 2024)
   - YaRN: Efficient Context Window Extension (Peng et al., 2023)

3. **DeepSeek 系列**
   - DeepSeek LLM: Scaling Open-Source Language Models (DeepSeek, 2024)
   - DeepSeek-V2: A Strong, Economical, and Efficient MoE Model (DeepSeek, 2024)
   - DeepSeek-V3 Technical Report (DeepSeek, 2024)
   - DeepSeekMoE: Towards Ultimate Expert Specialization (DeepSeek, 2024)

4. **架构创新**
   - GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)
   - MLA: Multi-Head Latent Attention (DeepSeek, 2024)
   - MoE: Mixture of Experts 综述

---

**文档版本**: v1.0  
**最后更新**: 2025-03-01  
**适用场景**: LLM 面试 - 架构对比 / 模型选择 / 推理优化  
**难度级别**: ⭐⭐⭐⭐ (高级，需要 Transformer + MoE + 推理基础)
