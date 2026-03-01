# 训练 Loop（面试可口述到"算子级"）

## 一句话结论

训练Loop是PyTorch中从数据到参数更新的完整流程：forward计算logits并shift对齐→cross-entropy loss计算→backward梯度累加与裁剪→optimizer.step()参数更新，每个环节都有数值稳定性和工程细节坑点。

## 核心定义/公式

### Forward Pass（前向传播）
```python
# 输入：input_ids [batch_size, seq_len]
# 输出：logits [batch_size, seq_len, vocab_size]

logits = model(input_ids)  # [B, L, V]

# Shift logits and labels for causal LM
# 因果语言模型需要错位：当前位置预测下一个token
shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
shift_labels = input_ids[..., 1:].contiguous()    # [B, L-1]

# Label masking（只计算非padding和target位置的loss）
loss_mask = (shift_labels != -100)  # -100是PyTorch CrossEntropyLoss的ignore_index
```

### Cross-Entropy Loss（交叉熵损失）
```python
# 标准交叉熵损失
# Loss = -∑ y_true * log(y_pred)
# 对于多分类：Loss = -log(y_pred[target_class])

# PyTorch实现
loss_fct = nn.CrossEntropyLoss(
    ignore_index=-100,           # 忽略padding和prompt位置的token
    label_smoothing=0.0,         # 可选：label smoothing (通常0.1)
    reduction='mean'             # 或 'sum' / 'none'
)

# logits: [B*L-1, V], labels: [B*L-1]
loss = loss_fct(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)

# 手动实现（理解原理用）
def manual_cross_entropy(logits, labels, ignore_index=-100, label_smoothing=0.0):
    """
    logits: [N, C] 未经过softmax的原始输出
    labels: [N] 类别索引
    """
    # Log softmax with numerical stability
    max_logits = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - max_logits
    log_probs = logits_stable - torch.log(torch.exp(logits_stable).sum(dim=-1, keepdim=True))
    
    # Create one-hot labels
    valid_mask = labels != ignore_index
    num_classes = logits.size(-1)
    one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
    
    # Label smoothing
    if label_smoothing > 0:
        one_hot = one_hot * (1 - label_smoothing) + label_smoothing / num_classes
    
    # Compute loss only on valid positions
    loss = -(one_hot * log_probs).sum(dim=-1)
    loss = loss[valid_mask].mean()
    
    return loss
```

### Backward Pass（反向传播）
```python
# 梯度累加（用于gradient accumulation）
loss = loss / accumulation_steps  # 梯度累加时需缩放loss
loss.backward()                   # 计算梯度，累加到.grad属性

# 梯度裁剪（防止梯度爆炸）
# 方法1：按范数裁剪
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,     # 通常0.5-1.0
    norm_type=2.0     # L2范数
)

# 方法2：按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Mixed Precision Training (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(
    init_scale=2**16,   # 初始scale，通常65536
    growth_factor=2.0,   # scale增长因子
    backoff_factor=0.5,  # 遇到inf/nan时scale回退因子
    growth_interval=2000 # 每多少步增长scale
)

with autocast():
    logits = model(input_ids)
    loss = compute_loss(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 反缩放梯度以便裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### Optimizer（优化器更新）
```python
# AdamW优化器（带解耦的weight decay）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,                    # 学习率
    betas=(0.9, 0.999),         # 一阶和二阶动量系数
    eps=1e-8,                   # 数值稳定性项
    weight_decay=0.01           # 权重衰减
)

# 关键：哪些参数不做weight decay
# 最佳实践：LayerNorm、bias、embedding不做weight decay
param_dict = {
    'decay': [],    # 应用weight decay
    'no_decay': []  # 不应用weight decay
}

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # LayerNorm权重、bias、embedding不做decay
    if 'ln' in name.lower() or 'norm' in name.lower() or 'bias' in name or 'embed' in name:
        param_dict['no_decay'].append(param)
    else:
        param_dict['decay'].append(param)

optimizer_grouped_parameters = [
    {'params': param_dict['decay'], 'weight_decay': 0.01},
    {'params': param_dict['no_decay'], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# AdamW更新公式（算子级）
# w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_{t-1})
# 
# 其中：
# m_t = beta1 * m_{t-1} + (1 - beta1) * grad      # 一阶动量
# v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2    # 二阶动量
# m_hat = m_t / (1 - beta1^t)                      # 偏差修正
# v_hat = v_t / (1 - beta2^t)
```

## 为什么（2-3 个因果链）

### 1. 为什么需要Shift Logits和Labels？
**现象**：因果语言模型需要基于当前位置预测下一个token  
**根因**：Transformer的因果mask确保位置i只能看到位置0~i的信息，但输出层每个位置都产生logits  
**结果**：必须错位对齐——位置i的logits预测位置i+1的token，所以shift_logits = logits[:-1]，shift_labels = input_ids[1:]

### 2. 为什么需要Label Masking？
**现象**：训练数据包含prompt（用户输入）和response（模型输出），但只想训练response部分  
**根因**：模型不应该被训练去预测prompt，浪费计算且可能学到错误模式  
**结果**：将prompt位置的label设为-100（PyTorch默认ignore_index），cross-entropy loss会在内部自动忽略这些位置

### 3. 为什么LayerNorm/Bias/Embedding不做Weight Decay？
**现象**：直接对所有参数做weight decay会导致模型性能下降  
**根因**：
- LayerNorm参数（scale和bias）控制归一化后的分布，decay会破坏这个分布
- Bias参数本身尺度小，decay会导致欠拟合
- Embedding参数直接承载语义信息，过度decay会损害表示能力  
**结果**：现代实践（如LLaMA、Qwen）对这些参数显式设置weight_decay=0.0

## 怎么做（可落地步骤）

### 标准训练Loop流程

#### 步骤1：数据预处理（txt → 训练样本）
```python
# 假设输入：math_problems.txt
# 格式：每行一个数学题 + 解答

import json
from transformers import AutoTokenizer

# 1.1 读取原始文本
with open('math_problems.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1.2 构造instruction格式
samples = []
for line in lines:
    # 假设格式："题目: xxx\n解答: yyy"
    parts = line.strip().split('\n解答: ')
    if len(parts) == 2:
        question = parts[0].replace('题目: ', '')
        answer = parts[1]
        samples.append({
            'instruction': '请解答以下数学问题。',
            'input': question,
            'output': answer
        })

# 1.3 保存为JSONL
with open('math_sft.jsonl', 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
```

#### 步骤2：Tokenization与Packing
```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B')
tokenizer.pad_token = tokenizer.eos_token

# 2.1 定义tokenize函数
def tokenize_function(example):
    # 构造对话格式
    prompt = f"User: {example['input']}\nAssistant: "
    full_text = prompt + example['output']
    
    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        return_tensors=None,
        add_special_tokens=True
    )
    
    # 关键：设置labels，prompt位置为-100
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
    labels = [-100] * len(prompt_tokens) + tokenized['input_ids'][len(prompt_tokens):]
    
    # 补齐labels到相同长度
    labels = labels + [-100] * (len(tokenized['input_ids']) - len(labels))
    tokenized['labels'] = labels
    
    return tokenized

# 2.2 加载数据集并tokenize
dataset = Dataset.from_json('math_sft.jsonl')
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    num_proc=4
)

# 2.3 Packing（可选，提升训练效率）
def pack_samples(examples, max_length=4096):
    """将多个短样本pack成一个长序列"""
    packed_input_ids = []
    packed_labels = []
    
    current_input_ids = []
    current_labels = []
    
    for input_ids, labels in zip(examples['input_ids'], examples['labels']):
        current_input_ids.extend(input_ids)
        current_labels.extend(labels)
        
        if len(current_input_ids) >= max_length:
            packed_input_ids.append(current_input_ids[:max_length])
            packed_labels.append(current_labels[:max_length])
            current_input_ids = current_input_ids[max_length:]
            current_labels = current_labels[max_length:]
    
    return {
        'input_ids': packed_input_ids,
        'labels': packed_labels,
        'attention_mask': [[1] * len(ids) for ids in packed_input_ids]
    }
```

#### 步骤3：完整训练Loop
```python
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

# 3.1 加载模型
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen-7B',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# 3.2 配置optimizer（关键：区分weight decay）
def get_optimizer_grouped_parameters(model, weight_decay=0.01):
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'ln.weight', 'embed']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    return optimizer_grouped_parameters

optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay=0.01)
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=2e-5,
    betas=(0.9, 0.95),
    eps=1e-8
)

# 3.3 学习率调度器
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.03 * num_training_steps)  # 3% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 3.4 训练循环
scaler = GradScaler()
accumulation_steps = 4  # 梯度累加步数

model.train()
global_step = 0

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        # Forward
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / accumulation_steps  # 梯度累加需缩放loss
        
        # Backward
        scaler.scale(loss).backward()
        
        # 梯度累加：每N步更新一次
        if (step + 1) % accumulation_steps == 0:
            # 梯度裁剪（在unscale之后）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            scaler.step(optimizer)
            scaler.update()
            
            # 清空梯度
            optimizer.zero_grad()
            scheduler.step()
            
            global_step += 1
            
            # 日志
            if global_step % 10 == 0:
                print(f"Step {global_step}, Loss: {loss.item() * accumulation_steps:.4f}")
```

### 关键配置/参数

| 参数名 | 推荐值 | 原因 |
|--------|--------|------|
| learning_rate | 1e-5 ~ 3e-5 (SFT) | 过大会导致灾难性遗忘，过小收敛慢 |
| weight_decay | 0.01 ~ 0.1 | LLM常用0.01，防止过拟合但不过度正则化 |
| max_grad_norm | 0.5 ~ 1.0 | 防止梯度爆炸，太大无效太小欠拟合 |
| accumulation_steps | 4 ~ 16 | 模拟大batch size，降低显存需求 |
| warmup_ratio | 0.03 ~ 0.1 | 训练初期稳定，避免梯度震荡 |
| label_smoothing | 0.0 ~ 0.1 | 防止过度自信，通常SFT不用或用很小的值 |

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **Label Smoothing (0.1)** | 防止过度自信，提升泛化 | 略降低训练精度，可能损害生成质量 | 分类任务推荐，生成任务慎用或用很小值 |
| **梯度累加** | 降低显存需求，模拟大batch | 增加训练时间，BN统计不准 | 显存受限时必用，与LayerNorm兼容好 |
| **Mixed Precision (FP16)** | 节省显存50%，加速训练 | 数值不稳定，需要loss scaling | 需配合GradScaler，现代常用BF16替代 |
| **Mixed Precision (BF16)** | 动态范围大，训练稳定 | 需要Ampere+架构GPU | 推荐用于LLM训练 |
| **Gradient Clipping** | 防止梯度爆炸，训练稳定 | 可能欠拟合，需调参 | 大模型训练标配 |
| **Weight Decay分离** | 保护关键参数，提升效果 | 代码复杂度增加 | 现代LLM标准实践 |
| **Sample Packing** | 提升训练吞吐30-50% | 实现复杂，可能影响batch内样本独立性 | 大规模SFT推荐 |

## 高频追问（至少 5 个）

### 1. Q: Transformer的forward计算包含哪些部件？
**A:** 
```
Input Embedding
  ↓
Positional Encoding (RoPE/ALiBi/绝对位置编码)
  ↓
×N Layers:
  ├─ Multi-Head Attention
  │   ├─ Q = Linear_Q(x)
  │   ├─ K = Linear_K(x)
  │   ├─ V = Linear_V(x)
  │   ├─ Attention = softmax(QK^T / √d_k + Mask) × V
  │   └─ Output = Linear_O(concat(heads))
  ├─ Add & LayerNorm (Residual)
  ├─ FFN (SwiGLU/GeLU)
  │   ├─ Gate = Linear_gate(x) × activation
  │   ├─ Up = Linear_up(x)
  │   └─ Down = Linear_down(Gate × Up)
  └─ Add & LayerNorm (Residual)
  ↓
Final LayerNorm
  ↓
Output Projection (lm_head)
  ↓
Logits [B, L, V]
```

### 2. Q: 为什么Cross-Entropy Loss要手动实现？和PyTorch内置有什么区别？
**A:** 
- **PyTorch内置**：`nn.CrossEntropyLoss`内部做了log_softmax + nll_loss，数值稳定（使用logsumexp trick）
- **手动实现场景**：
  1. 需要自定义label smoothing
  2. 需要访问中间概率值（如蒸馏）
  3. 需要特殊mask逻辑
  4. 调试和理解原理
- **关键差异**：手动实现需注意数值稳定性（减去max logits），否则容易溢出

### 3. Q: AdamW和Adam的weight decay有什么本质区别？
**A:** 
- **Adam**：weight decay加在梯度上，`grad = grad + wd * weight`，导致自适应学习率也作用在正则项上
- **AdamW**：weight decay直接作用在权重上，`weight = weight - lr * (m_hat/√v_hat + wd * weight)`，解耦正则化
- **效果**：AdamW的weight decay更纯粹，不会因自适应学习率而失效，在大模型训练中显著更好

### 4. Q: 混合精度训练中GradScaler的作用是什么？为什么会溢出？
**A:** 
- **原因**：FP16动态范围小（max≈65504），梯度容易underflow（变成0）
- **GradScaler机制**：
  1. Forward时将loss乘以scale factor（如65536）
  2. Backward时梯度也相应放大
  3. 参数更新前unscale梯度，保持梯度值正确
  4. 如果检测到inf/nan，跳过更新并降低scale
- **动态调整**：`scaler.update()`会自动调整scale，遇到inf就backoff，稳定就增长

### 5. Q: 训练时loss震荡怎么定位问题？
**A:** 
**排查顺序**：
1. **数据问题**：检查batch内样本长度分布、是否有异常长序列、是否有重复样本
2. **学习率**：太大导致震荡，尝试降低10倍或增加warmup
3. **梯度爆炸**：打印梯度范数，添加/调整gradient clipping
4. **混合精度**：检查是否频繁出现inf/nan，调整loss scaler初始值
5. **BatchNorm统计**：如果是BN，检查batch size是否太小
6. **数据泄漏**：验证集和训练集是否有重叠

### 6. Q: Label Mask除了-100还能用什么？为什么选-100？
**A:** 
- **PyTorch约定**：`CrossEntropyLoss`默认`ignore_index=-100`
- **原理**：PyTorch内部实现时，遇到该索引直接跳过，不计入loss和梯度
- **其他选择**：理论上任何不在`[0, vocab_size-1]`的值都可以，但-100是约定俗成
- **注意事项**：如果自定义loss函数，需要手动实现mask逻辑

### 7. Q: 多轮对话训练时loss mask怎么设计？
**A:** 
三种策略：
1. **只算最后一轮**：只有最后一轮response参与loss计算，前面所有轮次（包括历史对话）mask掉
2. **算所有assistant轮次**：所有assistant回复都参与训练，更充分但可能学到早期错误模式
3. **权重衰减**：早期轮次权重低，后期轮次权重高（如权重随轮次线性增长）

**推荐**：数据量充足用方案2，数据量少或质量不均用方案1

## 常见错误（至少 3 个）

### 1. **错误：忘记shift logits和labels**
**描述**：直接用`logits`和`input_ids`计算loss，导致位置对齐错误  
**正确做法**：
```python
# 错误
loss = F.cross_entropy(logits.view(-1, vocab_size), input_ids.view(-1))

# 正确
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = input_ids[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

### 2. **错误：所有参数都做weight decay**
**描述**：包括LayerNorm、bias、embedding在内的所有参数都应用weight_decay=0.01，导致性能下降  
**正确做法**：
```python
# 错误
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 正确
no_decay = ['bias', 'LayerNorm', 'ln', 'embed']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() 
               if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() 
               if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
```

### 3. **错误：梯度累加时忘记缩放loss**
**描述**：accumulation_steps=4，但不除以4，导致梯度放大4倍，训练不稳定  
**正确做法**：
```python
# 错误
loss.backward()
if step % 4 == 0:
    optimizer.step()

# 正确
loss = loss / accumulation_steps
loss.backward()
if step % accumulation_steps == 0:
    optimizer.step()
```

### 4. **错误：Label mask范围错误**
**描述**：mask了prompt的token，但忘记mask掉padding或EOS等特殊token  
**正确做法**：
```python
# 构造labels时，prompt部分mask，但也要检查padding
prompt_length = len(tokenizer(prompt)['input_ids'])
labels = input_ids.clone()
labels[:prompt_length] = -100  # mask prompt
labels[labels == tokenizer.pad_token_id] = -100  # mask padding
```

### 5. **错误：混合精度训练时梯度裁剪顺序错误**
**描述**：在`scaler.step()`之后才做梯度裁剪，或忘记`unscale_`  
**正确做法**：
```python
# 错误
scaler.scale(loss).backward()
scaler.step(optimizer)  # 错误：还未unscale就step
clip_grad_norm_(model.parameters(), 1.0)

# 正确
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 必须先unscale
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

## 反问面试官的问题

### 技术深度类
1. **训练稳定性**：你们训练大模型时最常遇到的数值稳定性问题是什么？是梯度爆炸、loss溢出还是其他？
2. **Loss监控**：除了看整体loss，你们会监控哪些中间指标（如各层梯度范数、attention分布等）来定位训练问题？
3. **Weight Decay策略**：你们团队有尝试过更细粒度的weight decay策略吗？比如对不同层使用不同系数？

### 业务场景类
1. **SFT数据配比**：对于垂类模型，你们是如何平衡通用数据和领域数据的配比的？
2. **训练效率**：在实际训练中，你们更看重训练速度还是显存效率？有用过哪些显存优化技巧？
3. **故障排查**：训练中出现过最难的bug是什么？最后怎么定位和解决的？

## 自测题

### 口述（能流畅讲清楚的知识点）
1. **完整训练流程**：从txt文件到模型参数更新，每一步做了什么？为什么这么做？
2. **Forward细节**：Transformer的forward包含哪些算子？每个算子的输入输出shape是什么？
3. **Loss计算**：Cross-Entropy Loss的计算过程？为什么要shift logits？label mask如何生效？
4. **Backward细节**：梯度如何传播？为什么要裁剪？AMP如何工作？
5. **Optimizer更新**：AdamW的更新公式？为什么某些参数不做weight decay？

### 手写（5 分钟能写出的代码/公式）
1. **手写Shift Loss**：给定logits [B, L, V]和input_ids [B, L]，写出正确的loss计算代码
2. **手写Optimizer分组**：写出让bias和LayerNorm不做weight decay的optimizer初始化代码
3. **手写Cross-Entropy**：用numpy手写cross-entropy loss（含label smoothing）
4. **手写梯度累加**：写出完整的梯度累加训练循环（含loss缩放、梯度清零）
5. **手写Adam更新**：写出Adam的参数更新公式（含偏差修正）

```python
# 练习：手写Shift Loss
def compute_loss(logits, input_ids, vocab_size, ignore_index=-100):
    """
    logits: [batch_size, seq_len, vocab_size]
    input_ids: [batch_size, seq_len]
    返回：scalar loss
    """
    # TODO: 实现shift和loss计算
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )
    return loss

# 练习：手写Adam更新公式
def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    """
    param: 当前参数值
    grad: 当前梯度
    m: 一阶动量
    v: 二阶动量
    t: 当前步数（从1开始）
    返回：新的param, m, v
    """
    # 更新动量
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    # 偏差修正
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # 参数更新（AdamW风格）
    param = param - lr * (m_hat / (torch.sqrt(v_hat) + eps) + wd * param)
    
    return param, m, v
```

## 标签

#训练 #SFT #pretrain #工程 #handwrite #derive #字节 #阿里

## 相关文档

- [[01-数据处理全链路]] - 数据清洗、去重、格式化等预处理流程
- [[02-tokenize与packing]] - Tokenization原理、packing策略与效率优化
- [[../07-分布式训练ZeRO/01-并行策略总览]] - 分布式训练中的数据并行与梯度同步
- [[../08-数值精度量化/01-训练精度选择]] - FP16/BF16混合精度训练原理
- [[../03-SFT与LoRA/01-SFT目标与边界]] - SFT的作用、边界与RLHF的关系