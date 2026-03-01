# SFT 目标与边界

## 一句话结论

SFT 通过高质量指令数据对齐模型行为（指令遵循、风格迁移），但无法从根本上解决偏好对齐、安全性约束和奖励黑客问题，因此 RLHF 仍有必要。

## 核心定义/公式

**SFT（Supervised Fine-Tuning，监督微调）**：在预训练模型基础上，使用指令-回复对（instruction-response pairs）进行监督学习，使模型学会遵循指令、对齐输出风格。

**训练目标**：
```python
# 标准 SFT Loss
loss = CrossEntropyLoss(
    logits=model(input_ids).logits,  # [batch, seq_len, vocab_size]
    target=labels,                    # [batch, seq_len]，通常 labels = input_ids shifted
    ignore_index=PAD_TOKEN_ID         # 忽略 padding 和可选的 user tokens
)

# 多轮对话 Loss Mask（只计算 assistant 回复部分）
loss_mask = create_loss_mask(roles)  # user tokens -> 0, assistant tokens -> 1
loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()
```

**典型数据格式**：
```json
{
  "instruction": "请解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}
```

## 为什么（2-3 个因果链）

### 1. SFT 能实现指令遵循（行为塑造）

- **现象**：预训练模型只学会"预测下一个 token"，但不知道"听从指令"
- **根因**：指令数据包含明确的任务描述与期望回复，模型通过最大似然学习"指令 → 正确回复"的映射
- **结果**：模型学会在输入特定指令时，生成符合预期的输出结构

**关键机制**：
- In-context learning（上下文学习）：few-shot 示例强化指令理解
- Pattern matching：模型记住指令模式并泛化到新指令
- 行为克隆（Behavior Cloning）：从专家回复中学习策略

### 2. SFT 能实现风格对齐（域适配）

- **现象**：不同应用场景需要不同风格（学术/口语/代码/安全）
- **根因**：SFT 数据集中回复的风格成为模型的"先验"，通过少量代表性样本即可泛化
- **结果**：模型输出与目标风格对齐，无需改变核心能力

**示例**：
- 学术风格："从理论角度分析，该方法的优势在于..."
- 代码风格：缩进规范、注释完整、类型提示
- 安全风格：拒绝危险请求、添加免责声明

### 3. 只用 SFT 无法完全解决偏好对齐（本质局限）

- **现象**：SFT 模型可能输出有害内容、编造事实、迎合用户但不真实
- **根因**：
  - SFT 只是"模仿"，学习的是数据中的模式，而非"理解什么是好"
  - 数据无法覆盖所有边界情况（尤其是对抗性输入）
  - 缺乏明确的"好坏信号"，无法区分高质量回复 vs 低质量回复
- **结果**：RLHF/DPO 通过偏好数据 + 奖励模型提供明确的优化方向

**为什么需要 RLHF**：
1. **处理模糊性**：同一个指令可能有多个合理回复，SFT 学一个，RLHF 学"最好的一个"
2. **安全对齐**：SFT 可以学会拒答，但可能被越狱；RLHF 强化安全边界
3. **减少幻觉**：RLHF 通过负反馈（rejected samples）惩罚编造行为
4. **避免奖励黑客**：SFT 可能学捷径（重复、模板化），RLHF 提供复杂奖励信号

## 怎么做（可落地步骤）

### 标准做法

#### 1. 数据准备阶段

```python
# 数据清洗流程
def clean_sft_data(raw_data):
    # 1. 去重（instruction + input 组合去重）
    data = deduplicate(data, keys=['instruction', 'input'])
    
    # 2. 质量过滤
    data = filter_by_length(data, min_len=10, max_len=2048)
    data = filter_by_quality_score(data, threshold=0.7)  # 使用模型打分
    
    # 3. 多样性采样
    data = stratified_sample(data, by='task_type', ratio=0.3)
    
    # 4. 格式统一
    data = format_to_template(data, template='chatml')
    
    return data
```

#### 2. 训练配置

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft_model",
    
    # 核心超参
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 有效 batch = 32
    learning_rate=2e-5,             # SFT 通常比 pretrain 小
    
    # 学习率调度
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # 数值稳定性
    fp16=True,                      # 或 bf16
    gradient_checkpointing=True,    # 节省显存
    
    # 正则化
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Logging
    logging_steps=10,
    save_strategy="epoch",
)
```

#### 3. 多轮对话处理

```python
def format_multiturn_conversation(history, max_turns=5):
    """
    多轮对话格式化
    - 只计算 assistant 的 loss
    - 处理超长上下文（滑动窗口）
    """
    formatted = []
    
    for i, turn in enumerate(history[-max_turns:]):  # 保留最近 N 轮
        formatted.append({
            "role": "user",
            "content": turn["user"]
        })
        formatted.append({
            "role": "assistant",
            "content": turn["assistant"]
        })
    
    # ChatML 格式
    text = ""
    for msg in formatted:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    return text
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `learning_rate` | 1e-5 ~ 5e-5 | SFT 数据量小，避免过拟合 |
| `epochs` | 2~5 | 太少学不好，太多过拟合 |
| `batch_size` | 32~128（有效 batch） | 稳定训练 |
| `max_length` | 1024~4096 | 根据任务调整，越长显存越大 |
| `warmup_ratio` | 0.03~0.1 | 预训练模型需要 warm-up |
| `weight_decay` | 0.01 | 防止过拟合 |

### 数据设计原则

#### 1. 质量优先于数量

```python
# 坏示例：低质量指令
{
  "instruction": "写个故事",
  "output": "从前有座山..."
}

# 好示例：高质量指令
{
  "instruction": "请写一个 200 字左右的童话故事，主题是友谊，主角是一只小兔子和一只小狐狸",
  "output": "森林里住着一只活泼的小兔子和一只聪明的小狐狸。起初他们互不理睬..."
}
```

**原则**：
- 指令清晰、具体、可执行
- 回复完整、准确、符合预期
- 避免"噪音指令"（模糊、歧义、不可达）

#### 2. 多样性覆盖

```python
# 任务类型配比（示例）
task_distribution = {
    "creative_writing": 0.15,
    "code_generation": 0.20,
    "qa": 0.25,
    "summarization": 0.15,
    "translation": 0.10,
    "math_reasoning": 0.10,
    "safety_rejection": 0.05
}

# 难度梯度
difficulty_distribution = {
    "easy": 0.30,      # 单步任务
    "medium": 0.50,    # 多步推理
    "hard": 0.20       # 复杂规划/创意
}
```

#### 3. 指令复杂度递增

```python
# Stage 1: 单指令
instruction = "翻译这句话为英文"

# Stage 2: 带约束的指令
instruction = "翻译这句话为英文，使用正式语气，不超过 50 个单词"

# Stage 3: 多任务指令
instruction = "翻译这句话为英文，然后总结翻译结果的关键信息"
```

### 代码示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# 数据集
class SFTDataset:
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.data = load_json(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造 prompt
        prompt = f"### Instruction:\n{item['instruction']}\n\n"
        if item.get('input'):
            prompt += f"### Input:\n{item['input']}\n\n"
        prompt += f"### Response:\n{item['output']}"
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels = input_ids（shifted）
        labels = encoded["input_ids"].clone()
        
        # 只计算 response 部分的 loss
        response_start = prompt.find("### Response:")
        response_tokens = self.tokenizer.encode("### Response:", add_special_tokens=False)
        # ... 找到 response 起始位置，前面的 token labels 设为 -100
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=SFTDataset("train.json", tokenizer)
)
trainer.train()
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **纯 SFT** | 实现简单、训练稳定、数据易获取 | 无法对齐偏好、安全边界弱、可能过拟合 | 快速原型、风格迁移、领域适配 |
| **SFT + RLHF** | 偏好对齐、安全性强、减少幻觉 | 实现复杂、训练不稳定、数据成本高 | 生产级对齐、安全敏感场景 |
| **SFT + DPO** | 比 PPO 简单、无需显式 RM、稳定性好 | 需要高质量偏好数据、数据效率可能较低 | 有偏好数据的对齐场景 |
| **仅 Continued Pretrain** | 知识注入强、无监督、数据量大 | 无法指令遵循、行为不改变 | 领域知识增强（医疗/法律/代码） |

**关键决策点**：
1. **是否需要 SFT**：预训练模型能否直接用？ → 大部分场景需要 SFT
2. **是否需要 RLHF**：是否有安全/偏好要求？ → 生产环境通常需要
3. **SFT 数据量**：质量 > 数量，几千到几十万高质量样本通常足够

## 高频追问（至少 5 个）

### 1. Q: SFT 和 RLHF 的作用分别是什么？

A:
- **SFT**：指令遵循（让模型听懂指令）、风格对齐（输出格式/语气）、行为塑造（学会任务模式）。本质是"教模型怎么做"
- **RLHF**：偏好对齐（区分好坏回复）、安全约束（拒绝有害请求）、减少幻觉（惩罚编造）。本质是"教模型什么是好"

**关系**：SFT 是"学会开车"，RLHF 是"学会遵守交通规则"

### 2. Q: 为什么 SFT 之后还要做 RLHF，只用 SFT 可以吗？

A:
**只用 SFT 的局限**：
1. **偏好对齐不足**：SFT 学习"一个回复"，RLHF 学习"最好的回复"
2. **安全边界弱**：SFT 可能被越狱，RLHF 强化安全边界
3. **幻觉问题**：SFT 无法有效惩罚编造，RLHF 通过负反馈减少
4. **奖励黑客**：SFT 可能学捷径（重复/模板化），RLHF 提供复杂奖励信号

**何时只用 SFT 可以**：
- 风格迁移（领域适配）
- 快速原型验证
- 无严格安全要求的场景

**何时需要 RLHF**：
- 生产环境部署
- 安全敏感场景（客服/教育/医疗）
- 需要区分细微偏好差异

### 3. Q: DPO 和 SFT 的区别是什么？可以先 DPO 后 SFT 吗？

A:
**DPO vs SFT**：
- **目标不同**：SFT 是行为克隆（学习"怎么写"），DPO 是偏好优化（学习"哪个更好"）
- **数据不同**：SFT 需要 instruction-output 对，DPO 需要 instruction-chosen-rejected 三元组
- **Loss 不同**：SFT 是交叉熵（最大似然），DPO 是偏好差最大化

**先 DPO 后 SFT 可以吗**：
- **理论上**：可以，但通常不推荐
- **实践上**：推荐顺序是 SFT → DPO
  - SFT 先学会基本指令遵循
  - DPO 在此基础上优化偏好
- **反过来可能的问题**：
  - DPO 需要模型已经有一定的指令跟随能力
  - 先 DPO 可能破坏预训练知识

### 4. Q: SFT 数据量和预训练数据量差多少？为什么 SFT 数据量这么小也能有效？

A:
**数量对比**：
- 预训练：万亿 tokens（1T~10T）
- SFT：几千到百万级样本（高质量）

**为什么小数据也有效**：
1. **知识已具备**：预训练模型已学会语言理解/生成，SFT 只是"激活"特定能力
2. **质量优先**：SFT 数据质量要求远高于预训练，几万高质量样本 > 百万噪声样本
3. **分布集中**：SFT 聚焦于下游任务分布，无需覆盖整个语言空间
4. **行为塑形**：SFT 更多是改变"行为模式"（何时做什么），而非注入知识

**关键**：SFT 的价值在于"对齐行为"，而非"注入知识"

### 5. Q: SFT 和 Continued Pretrain 有什么区别？

A:

| 维度 | SFT | Continued Pretrain |
|------|-----|-------------------|
| **目标** | 指令遵循、行为塑造 | 知识注入、领域适配 |
| **数据** | 指令-回复对（有监督） | 纯文本（无监督） |
| **学习信号** | 监督信号（最大似然） | 自监督信号（next token） |
| **改变什么** | 行为（怎么回答） | 知识（懂什么） |
| **典型场景** | 对话、问答、任务执行 | 医疗/法律/代码领域知识增强 |

**何时组合使用**：
1. 医疗场景：先 continued pretrain（注入医学知识）→ SFT（学会回答患者问题）
2. 代码场景：continued pretrain（学代码语法/模式）→ SFT（学会遵循编码规范）

### 6. Q: SFT 常见的失败模式有哪些？如何避免？

A:
**常见失败模式**：
1. **过拟合**：训练数据少、epoch 多，在验证集上 loss 不降反升
   - 解决：early stopping、数据增强、正则化
   
2. **灾难性遗忘**：SFT 后模型失去预训练知识
   - 解决：混合预训练数据（10~20%）、降低学习率、减少 epochs
   
3. **分布偏移**：SFT 数据与实际使用场景不匹配
   - 解决：数据多样性、domain adaptation、持续迭代
   
4. **奖励黑客**：模型学会"表面符合"但不真正解决问题
   - 解决：数据质量控制、多样本验证、加入 RLHF

### 7. Q: 如何评估 SFT 的效果？除了 loss 还看什么？

A:
**评估维度**：
1. **自动指标**：
   - Perplexity（困惑度）
   - Rouge/BLEU（与参考回复的重叠）
   - Accuracy（特定任务，如问答正确率）

2. **人工评估**：
   - Helpful：是否有帮助
   - Honest：是否真实
   - Harmless：是否安全
   - Style：风格是否符合预期

3. **业务指标**：
   - 任务完成率
   - 用户满意度
   - 拒答率/幻觉率

4. **对抗测试**：
   - 越狱测试
   - 长尾 case 覆盖
   - 边界情况

## 常见错误（至少 3 个）

### 1. 错误：SFT 数据越多越好

**错误做法**：
```python
# 爬取大量数据，不做清洗
data = scrape_all_available_data()  # 100万条
train_sft(data)  # 直接训练
```

**问题**：
- 低质量数据会引入噪声
- 模型学习错误模式
- 浪费计算资源

**正确做法**：
```python
# 质量优先
data = scrape_data()  # 100万条
data = quality_filter(data, threshold=0.7)  # 过滤到 30万条
data = deduplicate(data)  # 去重到 15万条
data = diversity_sample(data, task_distribution)  # 最终 10万条高质量数据
```

**经验法则**：1万高质量样本 > 10万噪声样本

### 2. 错误：SFT 不需要 RLHF，可以完全替代

**错误认知**：
- "只要 SFT 数据足够好，就不需要 RLHF"
- "RLHF 只是锦上添花"

**问题**：
- SFT 无法学习"相对偏好"（chosen > rejected）
- 安全边界靠 SFT 数据难以完全覆盖
- 复杂的偏好信号（有帮助 + 真实 + 安全）需要显式优化

**正确认知**：
- SFT + RLHF 是标准对齐流程
- 生产环境通常需要 RLHF/DPO
- 只有非安全敏感场景可以只用 SFT

### 3. 错误：SFT 学习率和预训练一样大

**错误做法**：
```python
training_args = TrainingArguments(
    learning_rate=1e-4,  # 和预训练一样大
    num_train_epochs=10,  # 训练很多轮
)
```

**问题**：
- SFT 数据量小，大学习率容易过拟合
- 破坏预训练知识（灾难性遗忘）
- 训练不稳定

**正确做法**：
```python
training_args = TrainingArguments(
    learning_rate=2e-5,    # SFT 通常 1e-5 ~ 5e-5
    num_train_epochs=3,    # 2~5 轮足够
    warmup_ratio=0.03,     # 需要 warm-up
    weight_decay=0.01,     # 正则化
)
```

### 4. 错误：忽略多轮对话的 loss mask

**错误做法**：
```python
# 对整个序列计算 loss（包括 user 输入）
loss = CrossEntropyLoss(logits, labels)  # labels = input_ids
```

**问题**：
- 模型学习生成 user 的输入（浪费容量）
- 无法专注于 assistant 回复
- 训练效率低

**正确做法**：
```python
# 只对 assistant 回复计算 loss
def create_loss_mask(roles):
    mask = torch.zeros_like(labels)
    # 根据角色标记，只有 assistant 部分为 1
    for i, role in enumerate(roles):
        if role == "assistant":
            mask[i] = 1
    return mask

loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()
```

### 5. 错误：SFT 数据格式不一致

**错误做法**：
```python
# 混用多种格式
data1 = {"instruction": "...", "output": "..."}
data2 = {"prompt": "...", "response": "..."}  # 不同字段名
data3 = "<human>: ... <bot>: ..."              # 纯文本格式
```

**问题**：
- 模型需要学习多种格式，浪费容量
- 推理时需要猜测格式
- 容易混淆

**正确做法**：
```python
# 统一使用标准格式（如 ChatML）
def format_to_chatml(instruction, input_text, output):
    if input_text:
        return f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    else:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
```

## 反问面试官的问题

### 1. 技术深度类

**Q**: 在团队的实际项目中，SFT 数据的质量评估主要依赖人工还是自动化指标？如果依赖人工，如何保证评估的一致性和效率？

**目的**：了解团队的工程成熟度、数据质量控制流程

**Q**: 团队是否尝试过只用 SFT 的对齐方案？如果尝试过，在哪些场景下效果可以接受，哪些场景下必须上 RLHF？

**目的**：了解团队对 SFT 边界的实践经验、技术决策依据

**Q**: 在 SFT 阶段，如何平衡数据多样性与特定任务的深度？比如如果业务聚焦于代码生成，是否应该让代码数据占比更高？

**目的**：了解数据配比策略、业务导向思维

### 2. 业务场景类

**Q**: 如果业务场景对安全性要求极高（如医疗、金融），SFT 阶段如何设计数据来覆盖安全边界？团队是否有积累的安全测试数据集？

**目的**：了解安全对齐实践、数据资产积累

**Q**: 在实际部署中，SFT 模型的迭代频率是怎样的？是否遇到过"新数据破坏旧能力"的情况，如何处理？

**目的**：了解持续迭代机制、灾难性遗忘处理

**Q**: 团队如何衡量 SFT 的 ROI（投入产出比）？在什么情况下认为"数据/算力投入已经足够"，不需要继续增加 SFT 数据？

**目的**：了解业务决策框架、资源分配逻辑

## 自测题

### 口述（能流畅讲清楚的知识点）

1. **SFT 的三大作用是什么？分别从因果链角度解释"为什么有效"**

   参考答案：
   - 指令遵循：指令数据提供明确任务描述 → 模型学习指令-回复映射 → 泛化到新指令
   - 风格对齐：代表性样本建立风格先验 → 少量样本即可泛化 → 输出风格一致
   - 能力迁移：SFT 数据包含任务模式 → 模型学会"何时做什么" → 唤醒预训练知识

2. **为什么只用 SFT 无法完全替代 RLHF？从偏好对齐、安全性、幻觉三个角度解释**

   参考答案：
   - 偏好对齐：SFT 学习"一个回复"，RLHF 学习"最好的回复"（相对偏好）
   - 安全性：SFT 可能被越狱，RLHF 强化安全边界（负反馈惩罚危险输出）
   - 幻觉：SFT 无法惩罚编造，RLHF 通过 rejected samples 提供负反馈

3. **SFT 数据设计的关键原则是什么？如何平衡质量、数量、多样性？**

   参考答案：
   - 质量优先：1万高质量 > 10万噪声
   - 多样性覆盖：任务类型、难度梯度、指令复杂度
   - 平衡策略：质量过滤 → 去重 → 多样性采样

4. **SFT 和 Continued Pretrain 的区别是什么？什么时候应该组合使用？**

   参考答案：
   - 区别：SFT 改变行为，Continued Pretrain 注入知识
   - 组合场景：领域应用（医疗/法律/代码）
   - 顺序：Continued Pretrain → SFT

### 手写（5 分钟能写出的代码/公式）

1. **写出 SFT 的 Loss 函数（单轮对话和多轮对话的 loss mask 版本）**

   参考答案：
   ```python
   # 单轮对话
   loss = CrossEntropyLoss(logits[:, :-1, :], labels[:, 1:])
   
   # 多轮对话（只计算 assistant 部分）
   def masked_loss(logits, labels, roles):
       loss_per_token = cross_entropy_per_token(logits, labels)
       mask = (roles == "assistant").float()  # 只有 assistant 为 1
       loss = (loss_per_token * mask).sum() / mask.sum()
       return loss
   ```

2. **写出 SFT 训练的关键超参数配置及理由**

   参考答案：
   ```python
   TrainingArguments(
       learning_rate=2e-5,      # SFT 数据量小，避免过拟合
       num_train_epochs=3,     # 2~5 轮足够
       warmup_ratio=0.03,      # 预训练模型需要 warm-up
       weight_decay=0.01,      # 防止过拟合
       gradient_checkpointing=True,  # 节省显存
   )
   ```

3. **画出 SFT → RLHF 的完整流程图（数据、模型、训练步骤）**

   参考答案：
   ```
   预训练模型
       ↓
   SFT 阶段：
     - 数据：指令-回复对
     - 训练：监督学习（最大似然）
     - 输出：SFT 模型（能听懂指令）
       ↓
   RLHF 阶段：
     - 数据：偏好数据（prompt, chosen, rejected）
     - 步骤1：训练奖励模型（RM）
     - 步骤2：用 PPO/DPO 优化策略模型
     - 输出：对齐模型（偏好对齐 + 安全）
   ```

## 标签

#训练 #SFT #RLHF #对齐 #指令遵循 #风格对齐 #偏好学习 #字节 #腾讯 #阿里

## 相关文档

- [[02-LoRA原理]] - SFT 的参数高效微调实现
- [[../04-RLHF对齐/01-RLHF总览]] - RLHF 完整流程与 SFT 的关系
- [[../04-RLHF对齐/03-DPO算法]] - DPO 与 SFT 的区别
- [[../02-训练数据/01-数据处理全链路]] - SFT 数据清洗与处理
- [[../02-训练数据/02-Tokenize与Packing]] - SFT 的 tokenize 策略
