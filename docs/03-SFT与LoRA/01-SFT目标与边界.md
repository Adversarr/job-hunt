# SFT 目标与边界

## 一句话结论

SFT 能显著改善指令遵循与风格，但对长尾偏好/安全边界/一致性等目标，通常需要偏好优化（RLHF/DPO/RLAIF）与检索/校验等系统手段配合；是否必须 RLHF 取决于应用风险与质量门槛。

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
- **条件语言建模**：学习"指令格式 → 回复格式"的条件生成分布
- **行为克隆（Behavior Cloning）**：在给定上下文下模仿专家回复分布
- **数据分布塑形（Format/Role Conditioning）**：通过训练数据格式统一指令解析、角色切换能力
- Pattern matching：模型记住指令模式并泛化到新指令

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

**为什么需要 RLHF/DPO**：
1. **处理相对偏好**：SFT 主要拟合参考答案分布；偏好优化（RLHF/DPO）显式学习相对排序/效用函数
   - 注：SFT 也可以通过多参考答案、拒绝样本过滤、pairwise ranking loss 部分注入偏好
2. **安全对齐**：SFT 可以学会拒答，但可能被越狱；RLHF 强化安全边界
3. **降低幻觉表现**：RLHF 可能减少部分幻觉（尤其是低置信度时的编造倾向），但系统性降低幻觉需配合检索（RAG）、引用约束、校验器/拒答策略等
4. **解决 SFT 的 Shortcut 问题**：SFT 可能学捷径（模板化、迎合式回答），需要偏好信号优化

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

### 工业级最佳实践（OLMo SFT 流程）

#### 1. Bin-Packing 数据加载（关键优化）

**问题**：传统 SFT 数据加载效率低，大量 padding tokens 浪费计算资源。

**OLMo 方案**：使用 Optimized Best-Fit Decreasing (OBFD) 算法最小化 padding。

```python
from olmo_core.data.utils import InstancePacker

class InstancePacker:
    """
    Bin-packing 算法实现（来自 OLMo-core）
    论文：Fewer Truncations Improve Language Modeling (arXiv:2404.10830)
    
    核心思想：
    1. 将文档按长度降序排序
    2. 使用线段树快速找到"最合适的 bin"（best-fit）
    3. 将多个短文档打包到一个固定长度 instance 中
    4. 最小化 padding tokens
    """
    def __init__(self, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.seg_tree = SegmentTree(max_sequence_length)  # O(log n) 查询
        self.instance_bins = []  # 每个 bin 包含的文档 ID
        self.space_to_bins = defaultdict(deque)  # 剩余空间 -> bin IDs
    
    def pack_documents(self, document_indices: np.ndarray):
        # 按文档长度降序排序
        document_lengths = document_indices[:, 1] - document_indices[:, 0]
        sorted_index = np.argsort(-1 * document_lengths)
        document_indices = np.take(document_indices, sorted_index, axis=0)
        
        # 对每个文档执行 best-fit 打包
        for document_id, (start_idx, end_idx) in enumerate(document_indices):
            document_len = int(end_idx - start_idx)
            self._pack_document(document_id, document_len)
        
        return self.instance_bins, document_indices, self.total_tokens
    
    def _pack_document(self, document_id: int, document_length: int):
        # 查询最合适的 bin（剩余空间 >= document_length 且最小）
        best_fit_leaf_id = self.seg_tree.query(document_length).leaf_id
        best_fit_capacity = best_fit_leaf_id + 1
        
        if best_fit_capacity == self.max_sequence_length:
            # 需要新 bin
            self.instance_bins.append([])
            bin_id = len(self.instance_bins) - 1
        else:
            # 使用已有 bin
            bin_id = self.space_to_bins[best_fit_capacity].popleft()
        
        # 将文档加入 bin
        self.instance_bins[bin_id].append(document_id)
        
        # 更新 bin 的剩余空间
        bin_space = best_fit_capacity - document_length
        if bin_space > 0:
            self.space_to_bins[bin_space].append(bin_id)
```

**效果**：相比 naive 数据加载，计算效率提升 **8x**（OLMo 实测）。

**关键点**：
- 使用 EOS token 标记文档边界（而非特殊分隔符）
- Chat template 必须在对话末尾使用单个 EOS token（OLMo template）
- 支持 document-level attention mask（避免跨文档 attention）

#### 2. 数据格式与 Loss Mask

**OLMo SFT 数据格式**：
```python
# 数据文件结构（numpy mmap 格式）
# token_ids_part_*.npy    - token IDs [num_tokens]
# labels_mask_*.npy       - loss mask [num_tokens], 0=user, 1=assistant

# 数据转换示例（从 HF datasets 转换为 OLMo 格式）
def convert_sft_to_olmo_format(dataset, tokenizer, output_dir, max_seq_length=32768):
    """
    将 SFT 数据转换为 OLMo-core 兼容格式
    
    关键步骤：
    1. 使用正确的 chat template（OLMo template 使用单个 EOS）
    2. Tokenize 并生成分段的 labels_mask
    3. 保存为 numpy mmap 格式
    """
    all_token_ids = []
    all_label_masks = []
    
    for example in dataset:
        # 应用 chat template
        formatted = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = tokenizer(formatted, add_special_tokens=False)
        token_ids = tokenized['input_ids']
        
        # 生成 labels_mask（只计算 assistant 回复）
        # OLMo 通过 EOS token 定位文档边界
        label_mask = create_label_mask(example['messages'], tokenizer)
        
        # 截断到 max_seq_length
        if len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]
            label_mask = label_mask[:max_seq_length]
        
        all_token_ids.extend(token_ids)
        all_label_masks.extend(label_mask)
    
    # 保存为 numpy 文件
    np.save(f"{output_dir}/token_ids_part_0.npy", np.array(all_token_ids, dtype=np.uint32))
    np.save(f"{output_dir}/labels_mask_0.npy", np.array(all_label_masks, dtype=np.bool_))
```

**多轮对话 Loss Mask 关键点**：
```python
def create_label_mask(messages, tokenizer):
    """
    生成 labels_mask，只计算 assistant 回复的 loss
    
    关键：
    - user tokens -> 0（不计算 loss）
    - assistant tokens -> 1（计算 loss）
    - 使用 EOS token 标记对话结束
    """
    label_mask = []
    
    for msg in messages:
        text = msg['content']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if msg['role'] == 'user':
            label_mask.extend([0] * len(tokens))
        else:  # assistant
            label_mask.extend([1] * len(tokens))
    
    # 添加 EOS token（OLMo template）
    label_mask.append(1)  # EOS 也计算 loss
    
    return label_mask
```

#### 3. 训练配置（OLMo 实践）

**关键配置差异**（相比预训练）：

| 参数 | 预训练 | SFT | 原因 |
|------|--------|-----|------|
| `learning_rate` | 6e-4 | **8e-5** | SFT 数据量小，避免过拟合 |
| `weight_decay` | 0.1 | **0.0** | SFT 不需要强正则化 |
| `scheduler` | Cosine | **Linear** (α_f=0.0) | 简单有效，最后 lr=0 |
| `epochs` | 1 | **3** | SFT 数据量少，需要多轮 |
| `warmup_fraction` | 0.01 | **0.03** | SFT 需要更多 warm-up |
| `max_grad_norm` | 1.0 | **1.0** | 相同 |

**OLMo 7B SFT 配置示例**：
```python
from olmo_core.train import TrainerConfig, Duration
from olmo_core.optim import SkipStepAdamWConfig, LinearWithWarmup

train_module_config = TransformerTrainModuleConfig(
    rank_microbatch_size=16384,  # tokens per microbatch
    max_sequence_length=32768,   # 推荐 32k（与 tokenize 一致）
    
    # 优化器（SFT 特有配置）
    optim=SkipStepAdamWConfig(
        lr=8e-05,              # SFT 学习率
        weight_decay=0.0,      # SFT 不需要 weight decay
        betas=(0.9, 0.95),
    ),
    
    # 学习率调度
    scheduler=LinearWithWarmup(
        warmup_fraction=0.03,
        alpha_f=0.0,           # 最终 lr=0
    ),
    
    # 梯度裁剪
    max_grad_norm=1.0,
    
    # 并行策略
    dp_config=TransformerDataParallelConfig(
        name=DataParallelType.hsdp,  # Hybrid Sharded Data Parallel
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=8,  # 每节点 8 GPUs
    ),
    
    # Activation Checkpointing（节省显存）
    ac_config=TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.selected_modules,
        modules=["blocks.*.feed_forward"],  # 只 checkpoint FFN
    ),
)

trainer_config = TrainerConfig(
    max_duration=Duration.epochs(3),  # 3 epochs
    save_folder="/path/to/checkpoints",
    # ... callbacks
)
```

#### 4. 长文档处理策略

```python
from olmo_core.data.types import LongDocStrategy

# 策略 1: Truncate（默认）
# - 超长文档截断到 max_sequence_length
# - 丢弃超出部分
# - 适合对话场景（超长对话很少）

# 策略 2: Fragment
# - 将超长文档拆分为多个片段
# - 不丢弃任何 token
# - 适合长文档理解任务
dataset_config = NumpyPackedFSLDatasetConfig(
    # ...
    long_doc_strategy=LongDocStrategy.truncate,  # 或 LongDocStrategy.fragment
    generate_doc_lengths=True,  # 生成文档长度信息，用于 attention mask
)
```

**文档级 Attention Mask**：
```python
# OLMo 自动生成文档边界信息
# 避免不同文档间的 cross-attention

# 示例：instance 包含 3 个文档
# Doc1: [tokens...] EOS
# Doc2: [tokens...] EOS
# Doc3: [tokens...] EOS

# Attention mask 确保每个 token 只 attend to 同一文档内的 tokens
```

#### 5. 完整流程（数据准备 → 训练 → 评估）

```bash
# Step 1: 数据准备（使用 open-instruct）
python scripts/data/convert_sft_data_for_olmocore.py \
    --dataset_mixer_list \
        allenai/tulu-3-sft-mixture 1.0 \
    --tokenizer_name_or_path allenai/dolma2-tokenizer \
    --output_dir /path/to/sft_data \
    --chat_template_name olmo \
    --max_seq_length 32768

# Step 2: 训练
python src/scripts/train/sft/Olmo-3-7B-SFT.py launch \
    MODEL_NAME /path/to/pretrain/checkpoint \
    ai2/jupiter \
    --trainer.max_duration.value=3 \
    --train_module.optim.lr=8e-5 \
    --seq_len=32768 \
    --dataset_path /path/to/sft_data

# Step 3: 转换为 HF 格式（评估用）
python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /path/to/sft/checkpoint \
    -o /path/to/hf_model \
    --max-sequence-length 32768

# Step 4: 验证 chat template（关键！）
# 确保 tokenizer_config.json 包含正确的 chat_template
# OLMo 3 特殊情况：Think models 需要特殊 chat template
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
        
        # 分段 tokenize（避免字符索引 ≠ token 索引的问题）
        instruction_text = f"### Instruction:\n{item['instruction']}\n\n"
        input_text = f"### Input:\n{item['input']}\n\n" if item.get('input') else ""
        response_text = f"### Response:\n{item['output']}"
        
        # 分别 tokenize
        instruction_tokens = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=False) if item.get('input') else []
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        
        # 拼接
        input_ids = instruction_tokens + input_tokens + response_tokens
        
        # 截断到 max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Labels = input_ids（shifted）
        labels = [-100] * len(instruction_tokens + input_tokens) + response_tokens
        
        # Padding
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
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
1. **偏好对齐不足**：SFT 主要拟合参考答案分布；RLHF/DPO 显式学习相对偏好（chosen > rejected）
2. **安全边界弱**：SFT 可能被越狱，RLHF 强化安全边界
3. **幻觉问题**：SFT 无法有效惩罚编造，RLHF 可能减少部分幻觉表现，但系统性降低需配合 RAG/校验等策略
4. **Shortcut 问题**：SFT 可能学捷径（重复/模板化/迎合式回答）

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
   
4. **Shortcut learning**：模型学会"表面符合"（模板化、迎合式回答）但不真正解决问题
   - 解决：数据质量控制、多样本验证、加入偏好优化（RLHF/DPO）
   
5. **奖励黑客（Reward Hacking）**：当使用奖励模型优化时，模型学会利用 RM 的盲点获得高分
   - 注：这是 RLHF 的风险，不是 SFT 的局限
   - 解决：RM 校准、对抗数据、KL 约束、在线监控

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

### 8. Q: OLMo 的 Bin-Packing 算法如何提升 SFT 效率？

A:
**问题**：传统 SFT 数据加载效率低，大量 padding tokens 浪费计算。

**OLMo 方案**：Optimized Best-Fit Decreasing (OBFD) 算法
- **论文**：Fewer Truncations Improve Language Modeling (arXiv:2404.10830)
- **实现**：使用线段树加速查询，O(log n) 找到 best-fit bin
- **效果**：计算效率提升 **8x**（OLMo 实测）

**核心步骤**：
1. 将文档按长度降序排序
2. 对每个文档找到"最合适的 bin"（剩余空间 >= doc_len 且最小）
3. 将多个短文档打包到一个固定长度 instance
4. 最小化 padding tokens

**关键要求**：
- 使用 EOS token 标记文档边界
- Chat template 必须在对话末尾使用单个 EOS（OLMo template）
- 支持 document-level attention mask

**适用场景**：
- 大规模 SFT（百万级样本）
- 变长文档训练
- 资源受限环境

### 9. Q: OLMo SFT 与传统 SFT 的配置差异是什么？

A:
**关键差异**（相比预训练和传统 SFT）：

| 参数 | 预训练 | 传统 SFT | OLMo SFT |
|------|--------|----------|----------|
| `learning_rate` | 6e-4 | 1e-5 ~ 5e-5 | **8e-5** |
| `weight_decay` | 0.1 | 0.01 | **0.0** |
| `scheduler` | Cosine | Cosine | **Linear** (α_f=0.0) |
| `epochs` | 1 | 2~5 | **3** |
| `warmup_fraction` | 0.01 | 0.03~0.1 | **0.03** |
| `max_seq_length` | 2048 | 1024~4096 | **32768** |

**为什么 weight_decay=0.0**：
- SFT 数据量小，不需要强正则化
- 避免破坏预训练知识
- 实践证明效果更好

**为什么 Linear scheduler**：
- 简单有效
- 最后 lr=0.0，确保收敛
- 相比 Cosine，SFT 训练轮数少，Linear 足够

**为什么 32k 上下文**：
- 支持长对话/长文档
- Tokenize 和训练使用相同长度（避免长度不匹配）
- 使用 Context Parallelism 处理长序列

### 10. Q: 如何处理 SFT 中的长文档？

A:
**两种策略**：

**1. Truncate（默认）**：
```python
long_doc_strategy = LongDocStrategy.truncate
```
- 超长文档截断到 max_sequence_length
- 丢弃超出部分
- 适合对话场景（超长对话很少）

**2. Fragment**：
```python
long_doc_strategy = LongDocStrategy.fragment
```
- 将超长文档拆分为多个片段
- 不丢弃任何 token
- 适合长文档理解任务

**OLMo 实践**：
- 使用 `generate_doc_lengths=True` 生成文档边界信息
- 自动创建 document-level attention mask
- 避免不同文档间的 cross-attention
- 推荐：对话场景用 truncate，长文档理解用 fragment

**注意**：Fragment 会增加数据量，需要调整训练轮数。

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

### 6. 错误：忽略 Chat Template 对 Bin-Packing 的影响

**错误做法**：
```python
# 使用不兼容 bin-packing 的 chat template
# 例如：每个对话末尾使用多个特殊 token
chat_template = """
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n<|extra_eos|>\n'}}
{% endfor %}
"""
```

**问题**：
- Bin-packing 算法依赖 EOS token 定位文档边界
- 多个 EOS token 会导致文档边界识别错误
- 无法正确打包多个文档到一个 instance

**正确做法**：
```python
# OLMo template：对话末尾使用单个 EOS token
chat_template = """
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}
"""
# 关键：每个对话只添加一个 EOS token
```

**OLMo 实践**：
- 使用官方 OLMo chat template
- Tokenize 时确保每个对话末尾只有一个 EOS
- 验证方式：检查 tokenizer 生成的 token 序列是否正确标记文档边界

### 7. 错误：Tokenize 和训练上下文长度不匹配

**错误做法**：
```python
# 数据准备阶段
tokenized_data = tokenize_data(dataset, max_length=4096)

# 训练阶段
train_config = TrainerConfig(
    max_sequence_length=8192,  # 与 tokenize 不一致！
)
```

**问题**：
- 训练时会进行二次截断或 padding
- 效率低下，可能丢失数据
- 模型在训练时学到的长度与实际不匹配

**正确做法**：
```python
# 保持一致！
MAX_SEQ_LENGTH = 32768  # 推荐 32k

# 数据准备阶段
tokenized_data = tokenize_data(
    dataset, 
    max_length=MAX_SEQ_LENGTH
)

# 训练阶段
train_config = TrainerConfig(
    max_sequence_length=MAX_SEQ_LENGTH,
)
```

**OLMo 推荐**：
- 统一使用 32k 上下文（支持长对话）
- Tokenize、训练、推理使用相同长度
- 如果显存不足，使用 Context Parallelism

### 8. 错误：SFT 后未验证 Chat Template 和 Tokenizer

**错误做法**：
```python
# 训练完成后直接用于评估
model = AutoModelForCausalLM.from_pretrained("/path/to/sft_model")
# 未检查 tokenizer_config.json 中的 chat_template
```

**问题**：
- 训练时保存的 tokenizer 可能 chat_template 不正确
- 评估时模型行为与预期不符
- 例如：Think models 需要特殊 chat template（包含 ` Hawthen` token）

**正确做法**：
```python
# 训练后验证 chat template
import json

tokenizer_path = "/path/to/sft_model/tokenizer_config.json"
with open(tokenizer_path) as f:
    config = json.load(f)

# 检查关键字段
assert "chat_template" in config, "Missing chat_template"
assert config["eos_token"] == "<|im_end|>", "Wrong EOS token"
assert config["pad_token"] == "<|pad|>", "Wrong PAD token"

# OLMo 3 特殊情况：Think models
# 需要确保 add_generation_prompt 包含 ` Hawthen` token
if is_think_model:
    assert " Hawthen" in config["chat_template"], "Missing thinking token"
```

**OLMo 实践**：
- 转换为 HF 格式后，手动验证 tokenizer_config.json
- 参考官方 OLMo 3 tokenizer 配置
- 测试生成样例确保行为正确

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
   - 偏好对齐：SFT 主要拟合参考答案分布；RLHF/DPO 显式学习相对偏好（chosen > rejected）
   - 安全性：SFT 可能被越狱，RLHF 强化安全边界（负反馈惩罚危险输出）
   - 幻觉：SFT 无法有效惩罚编造；RLHF 可能减少部分幻觉表现，但系统性降低需配合 RAG/校验等策略

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
   
   # OLMo SFT 特有配置
   TrainingArguments(
       learning_rate=8e-5,     # OLMo 实践：稍大的学习率
       weight_decay=0.0,       # OLMo 实践：不需要 weight decay
       lr_scheduler_type="linear",  # OLMo 实践：Linear scheduler
       warmup_ratio=0.03,
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

4. **解释 OLMo Bin-Packing 算法的核心思想**

   参考答案：
   ```
   核心思想：
   1. 将文档按长度降序排序
   2. 对每个文档使用 Best-Fit 策略找到最合适的 bin
      - 剩余空间 >= doc_len 且最小
   3. 使用线段树加速查询（O(log n)）
   4. 最小化 padding tokens
   
   关键要求：
   - 使用 EOS token 标记文档边界
   - Chat template 在对话末尾使用单个 EOS
   - 生成 document-level attention mask
   
   效果：计算效率提升 8x
   ```

## 标签

#pretrain #SFT #DPO #RLHF #字节 #腾讯 #阿里 #OLMo #bin-packing

## 相关文档

- [[02-LoRA原理]] - SFT 的参数高效微调实现
- [[../04-RLHF对齐/01-RLHF总览]] - RLHF 完整流程与 SFT 的关系
- [[../04-RLHF对齐/03-DPO算法]] - DPO 与 SFT 的区别
- [[../02-训练数据/01-数据处理全链路]] - SFT 数据清洗与处理
- [[../02-训练数据/02-Tokenize与Packing]] - SFT 的 tokenize 策略与 Bin-Packing

## 工业级 SFT 参考资料

### OLMo SFT 最佳实践

**代码仓库**：
- [OLMo-core](https://github.com/allenai/OLMo-core) - AI2 的训练库
- [open-instruct](https://github.com/allenai/open-instruct) - 数据准备与评估

**关键论文**：
- **Bin-Packing 算法**：[Fewer Truncations Improve Language Modeling](https://arxiv.org/pdf/2404.10830) (arXiv:2404.10830)
  - OBFD（Optimized Best-Fit Decreasing）算法
  - 最小化 padding tokens，提升训练效率

**核心组件**：
1. **数据格式**：Numpy mmap + labels mask
   - `token_ids_part_*.npy` - token IDs
   - `labels_mask_*.npy` - loss mask (0=user, 1=assistant)

2. **Bin-Packing 实现**：`olmo_core.data.utils.InstancePacker`
   - 使用线段树加速查询（O(log n)）
   - 支持文档级 attention mask

3. **并行策略**：
   - HSDP（Hybrid Sharded Data Parallel）
   - Context Parallelism（长序列支持）
   - Activation Checkpointing（显存优化）

4. **关键配置**：
   - `weight_decay=0.0`（SFT 特有）
   - `scheduler=Linear`（α_f=0.0）
   - `max_seq_length=32768`（推荐）

### 数据准备工具

**open-instruct 数据转换**：
```bash
# 从 HF datasets 转换为 OLMo 格式
python scripts/data/convert_sft_data_for_olmocore.py \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --tokenizer_name_or_path allenai/dolma2-tokenizer \
    --output_dir /path/to/output \
    --chat_template_name olmo \
    --max_seq_length 32768
```

**关键参数**：
- `--chat_template_name olmo` - 必须使用 OLMo template（单个 EOS）
- `--max_seq_length` - 与训练时保持一致
- `--visualize True` - 验证 tokenize 结果

### 训练脚本示例

**OLMo 7B SFT**：
```bash
# Launch 训练（Beaker 集群）
python src/scripts/train/sft/Olmo-3-7B-SFT.py launch \
    MODEL_NAME /path/to/pretrain/checkpoint \
    ai2/jupiter \
    --trainer.max_duration.value=3 \
    --train_module.optim.lr=8e-5 \
    --seq_len=32768 \
    --num_nodes=1 \
    --dataset_path /path/to/sft_data
```

**关键配置**：
- `--seq_len` - 上下文长度（推荐 32k）
- `--train_module.optim.lr` - 学习率（8e-5）
- `--trainer.max_duration.value` - 训练轮数（3 epochs）

### 评估与验证

**转换为 HF 格式**：
```bash
python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /path/to/sft/checkpoint \
    -o /path/to/hf_model \
    --max-sequence-length 32768
```

**验证 Chat Template**：
- 检查 `tokenizer_config.json` 中的 `chat_template`
- 确认 `eos_token`、`pad_token` 正确
- OLMo 3 Think models：确保包含 ` Hawthen` token

### 性能优化技巧

1. **Bin-Packing 效率**：
   - 使用 EOS token 标记文档边界
   - 避免在对话中间插入额外 EOS
   - 效果：**8x 计算效率提升**

2. **显存优化**：
   - Activation Checkpointing（只 checkpoint FFN）
   - Gradient Checkpointing
   - Context Parallelism（长序列）

3. **数据加载优化**：
   - Numpy mmap（零拷贝加载）
   - 并行数据预处理
   - 文档级打包（减少 padding）

### 常见问题排查

**问题 1：Bin-Packing 失败**
- 检查 chat template 是否使用单个 EOS
- 验证 tokenizer 的 EOS token ID 正确
- 查看数据预处理日志

**问题 2：训练效率低**
- 确认使用 bin-packing 数据格式
- 检查 padding token 占比（应该 < 5%）
- 验证数据加载是否并行

**问题 3：模型行为异常**
- 验证 tokenizer_config.json 配置
- 检查 chat template 是否正确
- 测试生成样例，确认格式对齐

**问题 4：显存不足**
- 降低 microbatch size
- 启用 activation checkpointing
- 使用 context parallelism（长序列）
- 减少 max_sequence_length
