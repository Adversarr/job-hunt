# YaRN（Yet another RoPE extensioN）

## 一句话结论
YaRN 通过温度缩放和 NTK-aware 插值扩展 RoPE，在不重新训练的情况下让模型支持 2-8 倍训练长度，同时保持短上下文性能不退化，是当前长上下文外推的 SOTA 方案之一。

## 核心定义/公式

### RoPE 基础回顾
RoPE（Rotary Position Embedding）通过旋转矩阵编码相对位置信息：

```
RoPE(x, m) = x * e^(i*m*θ)
```

其中：
- `m`：位置索引（0, 1, 2, ...）
- `θ_i = 10000^(-2i/d)`：基频，`d` 为 head_dim
- 不同频率对应不同"波长"，低频捕捉长程依赖，高频捕捉短程依赖

### YaRN 核心机制

**1. 温度缩放（Temperature Scaling）**

```python
# 标准注意力
attention_scores = QK^T / sqrt(d)

# YaRN 温度缩放
attention_scores = QK^T / (sqrt(d) * t)
t = 1 + 0.1 * log(scale_factor)  # scale_factor = 新长度/训练长度
```

**作用**：降低注意力分布的熵，让模型在更长上下文中更关注局部信息，缓解"注意力分散"问题。

**2. NTK-aware 插值**

```python
# 原始频率
theta_i = 10000^(-2i/d)

# NTK-aware 插值
def yarn_freq(scale_factor, dim, beta_fast=32, beta_slow=1):
    """YaRN 频率调整"""
    # 低频部分（长波长）保持不变
    # 高频部分（短波长）进行插值
    
    base_freq = 10000 ** (-torch.arange(0, dim, 2) / dim)
    
    # 计算波长
    wavelength = 2 * math.pi / base_freq
    
    # 定义高频/低频边界
    high_freq = wavelength < beta_fast
    low_freq = wavelength > beta_slow
    
    # 三角函数插值
    freq = base_freq.clone()
    
    # 中间频率：线性插值
    mid_freq = ~high_freq & ~low_freq
    freq[mid_freq] = base_freq[mid_freq] / scale_factor
    
    # 高频保持不变
    # 低频可以进一步调整
    
    return freq
```

**核心思想**：
- **高频分量**（短波长）：直接插值，类似线性扩展
- **低频分量**（长波长）：保持原频率，避免位置信息混淆
- **中间频率**：平滑过渡

**3. 完整 YaRN 实现**

```python
import torch
import math

def yarn_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    scale_factor: float = 4.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
):
    """
    YaRN 旋转位置编码
    
    Args:
        x: [batch, seq_len, num_heads, head_dim]
        position_ids: [batch, seq_len]
        scale_factor: 扩展倍数 (新长度/训练长度)
        beta_fast: 高频边界阈值
        beta_slow: 低频边界阈值
    
    Returns:
        应用 YaRN 后的 tensor
    """
    batch, seq_len, num_heads, head_dim = x.shape
    
    # 计算频率
    dim = head_dim
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
    # NTK-aware 频率调整
    inv_freq = yarn_adjust_freq(inv_freq, scale_factor, beta_fast, beta_slow)
    
    # 计算旋转角度
    position_ids = position_ids.unsqueeze(-1).float()  # [batch, seq_len, 1]
    angles = position_ids * inv_freq.unsqueeze(0)  # [batch, seq_len, dim//2]
    
    # 计算旋转矩阵
    cos = torch.cos(angles).unsqueeze(2)  # [batch, seq_len, 1, dim//2]
    sin = torch.sin(angles).unsqueeze(2)
    
    # 应用旋转
    x_rot = x.reshape(batch, seq_len, num_heads, -1, 2)
    x1, x2 = x_rot[..., 0], x_rot[..., 1]
    
    # 旋转操作
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    return x_rotated.reshape(x.shape)


def yarn_adjust_freq(
    inv_freq: torch.Tensor,
    scale_factor: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
):
    """调整频率：NTK-aware 插值"""
    # 计算波长
    wavelength = 2 * math.pi / inv_freq
    
    # 高频/低频边界
    high_freq_mask = wavelength < beta_fast
    low_freq_mask = wavelength > beta_slow
    
    # 中间频率
    mid_freq_mask = ~high_freq_mask & ~low_freq_mask
    
    # 高频：线性插值
    # 低频：保持不变
    # 中间：平滑过渡
    adjusted_freq = inv_freq.clone()
    
    if mid_freq_mask.any():
        adjusted_freq[mid_freq_mask] = inv_freq[mid_freq_mask] / scale_factor
    
    # 高频部分也插值（可选）
    if high_freq_mask.any():
        adjusted_freq[high_freq_mask] = inv_freq[high_freq_mask] / scale_factor
    
    return adjusted_freq


def yarn_attention_temperature(
    attention_scores: torch.Tensor,
    scale_factor: float = 4.0,
):
    """
    YaRN 温度缩放
    
    Args:
        attention_scores: [batch, num_heads, seq_len, seq_len]
        scale_factor: 扩展倍数
    """
    # 温度系数
    temperature = 1.0 + 0.1 * math.log(scale_factor)
    
    # 缩放注意力分数
    return attention_scores / temperature
```

### 关键参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `scale_factor` | 2.0 - 8.0 | 扩展倍数，越大外推越难 |
| `beta_fast` | 32.0 | 高频边界，控制哪些频率保持原样 |
| `beta_slow` | 1.0 | 低频边界，控制低频插值程度 |
| `temperature_coeff` | 0.1 | 温度系数，默认 0.1 * log(scale_factor) |

## 为什么（2-3 个因果链）

### 1. 为什么需要 YaRN？线性扩展的问题

**现象**：直接线性扩展 RoPE（位置索引除以 scale_factor）会导致短上下文性能严重下降。

**根因**：
- RoPE 的不同频率分量承担不同角色
  - **高频分量**：捕捉局部、精细的位置关系（相邻 token）
  - **低频分量**：捕捉全局、粗粒度的位置关系（长程依赖）
- 线性扩展统一缩放所有频率：
  - 高频部分被过度压缩 → 局部位置区分能力丧失 → 短上下文性能下降
  - 低频部分也被压缩 → 长程信息混淆 → 外推效果不佳

**结果**：
- 线性扩展：短上下文性能下降 10-30%，长上下文勉强能用
- YaRN：短上下文性能保持，长上下文外推能力强

### 2. 为什么 YaRN 有效？频率分量的差异化处理

**因果链**：

**问题**：训练时模型学到的频率分布对应特定长度范围
- 训练长度 4096：模型学习到位置 0-4096 的频率表示
- 推理时 16384：超出训练范围，位置信息失效

**解决方案**：YaRN 核心洞察
- 高频分量（波长 < 32）：
  - 在训练长度内已经充分学习
  - 直接插值不影响区分能力
  - 类似"把 100 个格子拉伸到 400 个"
  
- 低频分量（波长 > 1）：
  - 对应全局结构，训练时样本少
  - 保持原频率避免混淆
  - 让模型用已有知识泛化

**效果**：
- 高频：保留精细位置信息 → 短文本不退化
- 低频：保持全局结构 → 长文本能外推
- 中间：平滑过渡 → 整体性能最优

### 3. 为什么需要温度缩放？注意力熵的问题

**现象**：上下文变长后，注意力分数分布趋于均匀，"关注点分散"。

**根因分析**：
```
训练时：注意力分数 = softmax(QK^T / sqrt(d))
         seq_len = 4096，每个位置平均关注 4096 个 token

推理时：注意力分数 = softmax(QK^T / sqrt(d))
         seq_len = 16384，每个位置平均关注 16384 个 token
         
结果：softmax 输出熵增大，注意力变得"平均"
```

**YaRN 解决方案**：
```python
temperature = 1.0 + 0.1 * log(scale_factor)
attention_scores = attention_scores / temperature
```

**效果**：
- 降低温度 → softmax 分布更尖锐 → 保留局部关注能力
- 对长上下文建模更有效，避免"注意力稀释"

## 怎么做（可落地步骤）

### 标准做法

**步骤 1：确定扩展倍数**
```python
# 训练长度
train_length = 4096

# 目标推理长度
target_length = 16384

# 扩展倍数
scale_factor = target_length / train_length  # 4.0
```

**步骤 2：修改 RoPE 频率计算**
```python
# 标准 RoPE
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))

# YaRN 调整
inv_freq = yarn_adjust_freq(
    inv_freq,
    scale_factor=scale_factor,
    beta_fast=32.0,
    beta_slow=1.0
)
```

**步骤 3：应用温度缩放**
```python
# 在 attention 计算中
attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
attention_scores = attention_scores / (1.0 + 0.1 * math.log(scale_factor))
attention_probs = torch.softmax(attention_scores, dim=-1)
```

**步骤 4：处理 position_ids**
```python
# 方案 A：直接使用绝对位置（适合单轮对话）
position_ids = torch.arange(seq_len).unsqueeze(0)

# 方案 B：滑动窗口重映射（适合多轮对话）
def remap_positions(position_ids, window_size=4096):
    """滑动窗口内的位置重映射"""
    if position_ids.max() < window_size:
        return position_ids  # 未超窗口，直接返回
    
    # 超出窗口：只保留最近 window_size 个位置
    start_pos = position_ids.max() - window_size + 1
    remapped = torch.clamp(position_ids - start_pos, 0, window_size - 1)
    return remapped
```

### 关键配置/参数

**Hugging Face Transformers 配置**

```python
from transformers import AutoConfig, AutoModelForCausalLM

# 方案 1：修改 config（适合预训练模型）
config = AutoConfig.from_pretrained("Qwen/Qwen-7B")
config.rope_scaling = {
    "rope_type": "yarn",
    "factor": 4.0,  # 扩展倍数
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B",
    config=config,
)

# 方案 2：推理时动态调整
def apply_yarn_during_inference(model, scale_factor=4.0):
    """推理时应用 YaRN"""
    for name, module in model.named_modules():
        if 'rotary' in name.lower() or 'rope' in name.lower():
            # 修改频率
            module.inv_freq = yarn_adjust_freq(
                module.inv_freq,
                scale_factor=scale_factor
            )
            # 记录温度系数
            module.yarn_scale = scale_factor
```

**DeepSpeed / vLLM 配置**

```python
# vLLM 配置（支持 YaRN）
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen-72B",
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
    },
    max_model_len=32768,  # 扩展到 32K
)
```

### 代码示例：完整推理流程

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class YaRNModel:
    """支持 YaRN 的推理封装"""
    
    def __init__(
        self,
        model_path: str,
        train_length: int = 4096,
        target_length: int = 16384,
    ):
        self.train_length = train_length
        self.target_length = target_length
        self.scale_factor = target_length / train_length
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # 应用 YaRN
        self._apply_yarn()
    
    def _apply_yarn(self):
        """修改模型内部 RoPE 实现"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'rotary_emb'):
                # 修改频率
                original_inv_freq = module.rotary_emb.inv_freq
                module.rotary_emb.inv_freq = yarn_adjust_freq(
                    original_inv_freq,
                    self.scale_factor
                )
                
                # 添加温度缩放属性
                module.yarn_temperature = 1.0 + 0.1 * torch.log(
                    torch.tensor(self.scale_factor)
                )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 处理长序列
        if inputs['input_ids'].shape[1] > self.train_length:
            print(f"Warning: Input length {inputs['input_ids'].shape[1]} "
                  f"exceeds training length {self.train_length}")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# 使用示例
model = YaRNModel(
    model_path="Qwen/Qwen-7B",
    train_length=4096,
    target_length=16384,
)

long_prompt = "..." * 10000  # 超长上下文
response = model.generate(long_prompt)
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **YaRN** | • 无需重新训练<br>• 短上下文性能保持<br>• 外推能力强（2-8x）<br>• 实现简单 | • 超长（>8x）性能下降<br>• 参数需调优<br>• 依赖 RoPE 架构 | • 推理阶段外推<br>• 已有 RoPE 模型<br>• 2-8 倍扩展 |
| **线性扩展** | • 实现最简单<br>• 无额外参数 | • 短上下文严重退化<br>• 长上下文效果一般<br>• 无法精细控制 | • 快速原型验证<br>• 对短文本性能不敏感 |
| **NTK-aware** | • 比线性扩展好<br>• 频率感知 | • 缺少温度缩放<br>• 长上下文注意力分散<br>• 短文本仍有退化 | • 中等扩展需求（2-4x）<br>• 计算资源受限 |
| **滑动窗口注意力** | • 显存可控<br>• 可处理任意长度 | • 丢失长程依赖<br>• 需要特殊实现<br>• 不适合全局理解任务 | • 流式生成<br>• 显存受限场景<br>• 局部任务 |
| **稀疏注意力** | • 计算复杂度低<br>• 支持超长上下文 | • 需要训练<br>• 实现复杂<br>• 可能丢失关键信息 | • 预训练阶段<br>• 超长文档（>100K） |
| **长上下文微调** | • 效果最好<br>• 模型完全适应 | • 需要大量数据<br>• 训练成本高<br>• 可能灾难性遗忘 | • 生产环境<br>• 有训练资源<br>• 质量优先 |

### 方案选择决策树

```
是否可以重新训练？
├─ 是 → 长上下文微调（效果最优）
└─ 否 → 扩展倍数？
         ├─ 2-4x → YaRN（推荐）或 NTK-aware
         ├─ 4-8x → YaRN + 温度缩放
         └─ >8x → 滑动窗口 / 稀疏注意力 / 分段处理
```

## 高频追问（至少 5 个）

### 1. Q: YaRN 和线性扩展 RoPE 的本质区别是什么？

**A**: 本质区别在于**频率分量的差异化处理**：
- **线性扩展**：统一将所有位置索引除以 scale_factor，等价于所有频率分量都被压缩
  - 高频部分被过度压缩 → 短程位置区分能力丧失
  - 短上下文性能下降 10-30%
  
- **YaRN**：根据频率"波长"差异化处理
  - 高频（短波长）：线性插值，保留精细位置信息
  - 低频（长波长）：保持原频率，避免全局结构混淆
  - 中间频率：平滑过渡
  - 短上下文性能保持，长上下文外推能力强

**关键洞察**：RoPE 的不同频率分量承担不同任务，不能"一刀切"。

### 2. Q: YaRN 的温度缩放系数为什么是 `0.1 * log(scale_factor)`？

**A**: 这个公式来自经验观察和理论推导：

**理论依据**：
1. **注意力熵问题**：序列变长，注意力分布趋于均匀（熵增大）
2. **温度作用**：降低温度 → softmax 分布更尖锐 → 保留局部关注
3. **对数关系**：熵增大致与 log(length) 成正比

**实验验证**：
- Llama-2 上测试 2x、4x、8x 扩展
- 系数 0.1 是在多个 scale_factor 上调优的结果
- 太小（< 0.05）：注意力仍然分散
- 太大（> 0.2）：过度聚焦局部，丢失长程依赖

**变体**：
```python
# 原始 YaRN
temperature = 1.0 + 0.1 * log(scale_factor)

# 动态温度（按 head 调整）
temperature = 1.0 + alpha * log(scale_factor) * head_importance_weights
```

### 3. Q: Qwen 和 DeepSeek 是如何使用 YaRN 的？

**A**: 

**Qwen 系列**：
- **Qwen-7B/14B/72B**：训练长度 8192，通过 YaRN 扩展到 32K
- **实现细节**：
  ```python
  # Qwen 的 rope_scaling 配置
  rope_scaling = {
      "type": "yarn",
      "factor": 4.0,  # 8192 → 32768
      "original_max_position_embeddings": 8192,
  }
  ```
- **特点**：
  - 使用 GQA 减少长上下文的 KV cache 压力
  - 配合滑动窗口（局部注意力）处理超长文档
  - 通过长文本 SFT 数据进一步提升效果

**DeepSeek 系列**：
- **DeepSeek-V2/V3**：支持 128K-256K 上下文
- **策略组合**：
  1. **训练阶段**：使用长上下文数据 continued pretrain
  2. **推理阶段**：YaRN + 窗口注意力混合
  3. **架构优化**：MLA（Multi-Head Latent Attention）降低 KV cache
  4. **工程优化**：FlashAttention + PagedAttention
- **实现细节**：
  ```python
  # DeepSeek 的多层策略
  # 1. 前 32K：完整注意力 + YaRN
  # 2. 32K-128K：滑动窗口注意力
  # 3. 超过 128K：分段处理
  ```

**关键差异**：
- Qwen：纯推理时 YaRN，无需重新训练
- DeepSeek：训练 + 推理组合，更长上下文（128K+）

### 4. Q: YaRN 能支持多长的上下文扩展？有什么限制？

**A**: 

**理论极限**：
- **2-4x**：性能基本不退化，是推荐范围
- **4-8x**：性能轻微下降（PPL 增加 5-10%），仍可用
- **8-16x**：性能明显下降，需要结合其他技术
- **>16x**：单独 YaRN 效果不佳，需要训练

**限制因素**：

1. **训练分布偏移**：
   - 模型只在短文本上训练，长文本分布外推困难
   - 注意力模式、位置感知能力受限

2. **注意力机制限制**：
   - Softmax 熵随长度增长，注意力分数趋于均匀
   - 温度缩放只能缓解，无法完全解决

3. **KV cache 显存**：
   ```python
   # 70B 模型，16K 上下文，FP16
   KV_cache_size = 2 * num_layers * batch_size * seq_len * num_heads * head_dim * 2 bytes
                 = 2 * 80 * 1 * 16384 * 64 * 128 * 2
                 = 40 GB  # 单个请求！
   ```

4. **位置信息泛化**：
   - YaRN 假设低频分量可以外推
   - 但模型未见过超长距离的位置关系

**突破方案**：
```python
# 组合策略
def ultra_long_context(model, seq_len):
    if seq_len < 16K:
        return yarn_attention(model, scale_factor=4.0)
    elif seq_len < 64K:
        # YaRN + 滑动窗口
        return yarn_sliding_window(model, window=16K)
    else:
        # 分段 + RAG
        return chunked_attention_with_rag(model)
```

### 5. Q: YaRN 和滑动窗口注意力可以结合吗？如何结合？

**A**: 可以结合，且是处理超长上下文的有效方案。

**结合策略**：

**策略 1：分段处理**
```python
def yarn_with_sliding_window(
    attention_scores,
    position_ids,
    window_size=4096,
    scale_factor=4.0,
):
    """
    YaRN + 滑动窗口
    
    Args:
        attention_scores: [batch, heads, seq_len, seq_len]
        window_size: 窗口大小
        scale_factor: YaRN 扩展倍数
    """
    batch, heads, seq_len, _ = attention_scores.shape
    
    # 1. 应用 YaRN 温度缩放
    temperature = 1.0 + 0.1 * math.log(scale_factor)
    attention_scores = attention_scores / temperature
    
    # 2. 创建滑动窗口 mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, :start] = False  # 窗口外置 False
    
    # 3. 应用 mask
    attention_scores = attention_scores.masked_fill(
        ~mask.unsqueeze(0).unsqueeze(0),
        float('-inf')
    )
    
    return attention_scores
```

**策略 2：分层注意力**
```python
def hierarchical_yarn(model, seq_len):
    """
    分层注意力：
    - 局部：滑动窗口 + YaRN
    - 全局：稀疏采样（每隔 N 个 token）
    """
    # 局部窗口
    local_attn = yarn_sliding_window(window=4096)
    
    # 全局 token（如 [CLS]、段落首 token）
    global_tokens = [0, 512, 1024, ...]  # 关键位置
    
    # 组合
    # attention = local_attn + global_attn
```

**策略 3：动态窗口**
```python
def dynamic_yarn_window(
    position_ids,
    base_window=4096,
    scale_factor=4.0,
):
    """
    动态调整窗口大小：
    - 短文本：完整注意力
    - 长文本：缩小窗口 + YaRN
    """
    seq_len = position_ids.max() + 1
    
    if seq_len < base_window:
        # 无需窗口
        return seq_len
    else:
        # 动态窗口
        window = base_window * scale_factor / (seq_len / base_window)
        return min(int(window), base_window)
```

**应用场景**：
- **文档问答**：全局检索 + 局部精细理解
- **代码补全**：局部上下文 + 函数定义等全局信息
- **多轮对话**：最近 N 轮完整 + 早期对话压缩

### 6. Q: YaRN 对不同规模的模型效果一样吗？70B 和 7B 有什么差异？

**A**: 效果不同，规模越大 YaRN 的效果越好。

**实验观察**：

| 模型规模 | 训练长度 | YaRN 扩展 4x | PPL 变化 | 短文本性能 |
|---------|---------|-------------|---------|----------|
| 7B | 4096 | 16K | +15% | -5% |
| 13B | 4096 | 16K | +10% | -3% |
| 70B | 4096 | 16K | +5% | -1% |

**原因分析**：

1. **大模型泛化能力更强**：
   - 更多参数 → 更强的位置表示学习能力
   - 训练时见过更丰富的位置关系模式

2. **注意力机制更稳定**：
   - 大模型 head 数更多，分工更明确
   - 部分头专门负责长程依赖，受外推影响小

3. **RoPE 频率利用更充分**：
   - 大模型能更好利用低频分量（长波长）
   - 小模型低频分量训练不充分

**实践建议**：
- **<10B 模型**：优先考虑长上下文微调
- **10B-70B**：YaRN 是高性价比方案
- **>70B**：YaRN 效果接近重新训练

### 7. Q: 如何评估 YaRN 的效果？有哪些关键指标？

**A**: 

**核心指标**：

1. **困惑度（Perplexity）**
```python
def evaluate_ppl(model, tokenizer, long_texts):
    """评估长文本困惑度"""
    model.eval()
    total_ppl = 0
    
    for text in long_texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            ppl = torch.exp(outputs.loss)
            total_ppl += ppl.item()
    
    return total_ppl / len(long_texts)

# 对比
short_ppl = evaluate_ppl(model, tokenizer, short_texts)  # <4K
long_ppl_yarn = evaluate_ppl(model_yarn, tokenizer, long_texts)  # 4K-16K
long_ppl_baseline = evaluate_ppl(model_baseline, tokenizer, long_texts)

print(f"短文本 PPL: {short_ppl}")
print(f"长文本 YaRN PPL: {long_ppl_yarn}")
print(f"长文本 baseline PPL: {long_ppl_baseline}")
```

2. **长上下文检索任务（Passkey Retrieval）**
```python
def passkey_retrieval_test(model, tokenizer, length):
    """
    Passkey 检索：在长文本中插入一个密钥，测试模型能否检索
    
    Example:
    "There is a special token [KEY-12345] in this text..."
    要求模型输出 "12345"
    """
    key = random.randint(10000, 99999)
    text = generate_long_text(length, key=key)
    
    prompt = f"{text}\n\nWhat is the key number?"
    response = model.generate(prompt)
    
    return key in response
```

3. **长文本问答（LongBench）**
- Retrieve-based QA：需要从长文档检索信息
- Summarization：长文档摘要
- Few-shot learning：长上下文示例学习

**评估框架**：
```python
# 使用 LongBench 评估
from longbench import LongBenchEvaluator

evaluator = LongBenchEvaluator(model, tokenizer)
results = evaluator.evaluate(
    tasks=["longqa", "summ", "passkey"],
    max_length=16384,
    scale_factor=4.0,
)

print(f"长文本问答: {results['longqa']}")
print(f"摘要: {results['summ']}")
print(f"密钥检索: {results['passkey']}")
```

**对比基准**：
- **短文本性能保持率**：< 5% 下降算优秀
- **长文本 PPL**：< 20% 上升算可接受
- **密钥检索准确率**：>90% 为良好

## 常见错误（至少 3 个）

### 1. 错误：直接修改训练好的模型位置编码参数

**错误做法**：
```python
# ❌ 错误：试图修改模型的位置嵌入层
model.model.embed_positions.weight.data = interpolate_positions(
    model.model.embed_positions.weight.data,
    scale_factor=4.0
)
```

**问题**：
- RoPE 是**动态计算**的，没有固定的 embedding 矩阵
- 这样改的是绝对位置编码（如 BERT 的 position_embedding）
- 对 RoPE 模型无效

**正确做法**：
```python
# ✅ 正确：修改 RoPE 的频率计算
for name, module in model.named_modules():
    if 'rotary' in name.lower():
        # 调整 inv_freq
        module.inv_freq = yarn_adjust_freq(
            module.inv_freq,
            scale_factor=4.0
        )
```

### 2. 错误：温度缩放应用于所有层或所有 head

**错误做法**：
```python
# ❌ 错误：全局温度缩放
temperature = 1.0 + 0.1 * log(scale_factor)
for name, module in model.named_modules():
    if 'attention' in name:
        module.temperature = temperature
```

**问题**：
- 不同层、不同 head 对位置信息的敏感度不同
- 全局温度可能过度抑制某些层的长程依赖能力

**正确做法**：
```python
# ✅ 正确：按层或按 head 调整温度
# 方案 1：浅层弱缩放，深层强缩放
def get_layer_temperature(layer_idx, num_layers, scale_factor):
    base_temp = 1.0 + 0.1 * log(scale_factor)
    # 浅层保持更强长程能力
    return base_temp * (0.8 + 0.2 * layer_idx / num_layers)

# 方案 2：根据 attention head 的位置角色调整
def get_head_temperature(head_idx, num_heads):
    # 部分 head 专门负责长程，温度更小
    if head_idx < num_heads // 4:  # 前 1/4 head 负责长程
        return 1.0  # 不缩放
    else:
        return 1.0 + 0.1 * log(scale_factor)
```

### 3. 错误：忽略多轮对话的 position_id 处理

**错误做法**：
```python
# ❌ 错误：直接累加 position_id
def chat_with_yarn(model, messages):
    # 每轮对话的 position_id 不断累加
    # 最终超过训练长度，但未做窗口处理
    position_ids = torch.arange(total_length)
    # ...
```

**问题**：
- 多轮对话累积，position_id 很快超过训练长度
- 即使应用 YaRN，累积位置会让模型混淆
- 未考虑用户偏好局部上下文

**正确做法**：
```python
# ✅ 正确：滑动窗口 + 位置重映射
def chat_with_yarn(model, messages, max_window=4096):
    """
    多轮对话处理：
    1. 累积对话历史
    2. 超过窗口时，保留最近 N 个 token
    3. 重映射 position_id 到 [0, window_size)
    """
    # 1. 拼接历史
    full_text = "\n".join(messages)
    tokens = tokenizer(full_text, return_tensors="pt")
    
    # 2. 滑动窗口
    if tokens['input_ids'].shape[1] > max_window:
        tokens['input_ids'] = tokens['input_ids'][:, -max_window:]
        # 关键：重映射 position_id
        tokens['position_ids'] = torch.arange(max_window).unsqueeze(0)
    else:
        tokens['position_ids'] = torch.arange(tokens['input_ids'].shape[1]).unsqueeze(0)
    
    # 3. 应用 YaRN
    outputs = model.generate(**tokens)
    return outputs
```

### 4. 错误：YaRN 参数设置不当导致性能退化

**错误做法**：
```python
# ❌ 错误：参数未调优
scale_factor = 16.0  # 扩展 16 倍
beta_fast = 32.0     # 使用默认值
beta_slow = 1.0      # 使用默认值
# 结果：性能严重退化
```

**问题**：
- 不同 scale_factor 需要不同的 beta_fast/beta_slow
- 扩展倍数越大，越需要精细调参

**正确做法**：
```python
# ✅ 正确：根据 scale_factor 调整参数
def get_yarn_params(scale_factor):
    """根据扩展倍数自动调整参数"""
    if scale_factor <= 2.0:
        return {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "temperature_coeff": 0.1,
        }
    elif scale_factor <= 4.0:
        return {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "temperature_coeff": 0.15,
        }
    elif scale_factor <= 8.0:
        return {
            "beta_fast": 48.0,  # 增大高频边界
            "beta_slow": 2.0,   # 调整低频边界
            "temperature_coeff": 0.2,
        }
    else:
        # 扩展 >8x 需要训练，仅 YaRN 不够
        print(f"Warning: scale_factor={scale_factor} > 8.0, YaRN may not work well")
        return {
            "beta_fast": 64.0,
            "beta_slow": 4.0,
            "temperature_coeff": 0.25,
        }
```

### 5. 错误：混淆 YaRN 与其他位置编码方法

**错误做法**：
```python
# ❌ 错误：在非 RoPE 模型上应用 YaRN
# 如 BERT（绝对位置编码）、ALiBi（线性偏置）
model = BertModel.from_pretrained("bert-base")
apply_yarn(model)  # 无效！
```

**问题**：
- YaRN 是专门针对 RoPE 的扩展方案
- 不同位置编码方法的外推策略完全不同：
  - **ALiBi**：天然支持外推，无需 YaRN
  - **绝对位置编码**：需要插值 embedding 矩阵
  - **相对位置编码**：需要调整相对位置偏置

**正确做法**：
```python
# ✅ 正确：识别位置编码类型
def get_position_encoding_type(model):
    """判断模型的位置编码类型"""
    if hasattr(model, 'rope_embedding'):
        return 'rope'
    elif 'alibi' in str(type(model)).lower():
        return 'alibi'
    elif hasattr(model, 'position_embeddings'):
        return 'absolute'
    else:
        return 'unknown'

# 根据类型选择外推策略
pos_type = get_position_encoding_type(model)
if pos_type == 'rope':
    apply_yarn(model, scale_factor=4.0)
elif pos_type == 'alibi':
    # ALiBi 天然支持外推
    print("ALiBi supports extrapolation natively")
elif pos_type == 'absolute':
    # 绝对位置编码：插值 embedding
    interpolate_position_embeddings(model, scale_factor=4.0)
```

## 反问面试官的问题

### 技术深度类问题

1. **YaRN 的局限性**：在您的实际项目中，YaRN 在多长的上下文扩展时会遇到性能瓶颈？是优先选择继续训练还是结合其他技术（如滑动窗口）？

2. **工程实践**：Qwen 和 DeepSeek 在长上下文上的方案差异较大（Qwen 偏重推理时外推，DeepSeek 偏重训练时优化），您认为哪种更适合实际生产环境？有哪些权衡？

3. **性能评估**：除了困惑度和密钥检索，还有哪些更贴近业务的长上下文评估指标？如何评估 YaRN 在实际任务（如 RAG、多轮对话）中的效果？

### 业务场景类问题

1. **成本权衡**：在显存和计算成本受限的情况下，您会优先选择 YaRN（推理时扩展）还是长上下文微调（训练成本高但效果好）？

2. **多技术组合**：在处理超长文档（如 128K+）时，您倾向于哪种技术组合？YaRN + 滑动窗口、分段 + RAG，还是完全依赖训练？

3. **未来方向**：随着上下文长度需求不断增长（从 4K 到 128K 再到 1M+），您认为 YaRN 这类外推方案的适用性如何？会不会被新的架构（如 Linear Attention、Mamba）取代？

## 自测题

### 口述题（能流畅讲清楚的知识点）

1. **核心机制**：YaRN 如何通过差异化处理不同频率分量来平衡短文本性能和长文本外推能力？

2. **温度缩放**：为什么需要在注意力计算中加入温度缩放？公式 `t = 1 + 0.1 * log(scale_factor)` 的来源是什么？

3. **方案对比**：对比 YaRN、线性扩展、NTK-aware 三种 RoPE 外推方案的优缺点和适用场景。

4. **工程落地**：如果让你在生产环境中部署一个支持 32K 上下文的模型（原始训练长度 8K），你会如何选择技术方案？

5. **失败案例**：在什么情况下 YaRN 会失效或效果不佳？如何诊断和解决？

### 手写题（5 分钟能写出的代码/公式）

1. **YaRN 频率调整**：手写 `yarn_adjust_freq` 函数，实现 NTK-aware 插值逻辑。

2. **温度缩放**：写出 YaRN 的注意力计算公式，包含温度缩放项。

3. **参数配置**：给定 scale_factor=4.0，手写 Hugging Face 的 rope_scaling 配置。

```python
# 参考答案
rope_scaling = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}
```

4. **显存估算**：手写 70B 模型在 16K 上下文下的 KV cache 显存占用公式。

```python
# 参考答案
def estimate_kv_cache_memory(
    num_layers=80,
    batch_size=1,
    seq_len=16384,
    num_heads=64,
    head_dim=128,
    dtype_bytes=2,  # FP16
):
    return 2 * num_layers * batch_size * seq_len * num_heads * head_dim * dtype_bytes

# 计算
memory = estimate_kv_cache_memory()
print(f"KV cache 显存: {memory / 1e9:.2f} GB")  # 约 40 GB
```

5. **滑动窗口 + YaRN**：手写结合滑动窗口和 YaRN 的注意力 mask 生成代码。

## 标签

#YaRN #RoPE #kv_cache #paged_attention #FlashAttention #字节 #阿里 #腾讯 #美团

## 相关文档

- [[01-上下文窗口与外推]]：上下文窗口基础概念、外推问题的定义
- [[02-RoPE体系]]：RoPE 基础原理、频率分量分析、与其他位置编码对比
- [[../09-推理Infra/02-KV Cache核心]]：长上下文下的显存优化、Paged Attention
- [[../09-推理Infra/03-Paged Attention]]：超长上下文的显存管理方案
- [[../10-FlashAttention/01-FlashAttention原理]]：长上下文下的注意力计算优化