# Tokenize 与 Packing

## 一句话结论
Padding 通过填充对齐序列长度，Packing 通过拼接多个短样本提升 GPU 利用率；Packing 需配合 loss mask 正确处理多轮对话和样本边界，避免在 padding token 上计算损失；DeepSeek 使用字节级 BPE + 增量词表，相比传统 BPE 在中文和代码场景下 tokenization 效率提升 40%+。

## 核心定义/公式

### Padding vs Packing

**Padding（填充策略）**
```python
# 静态 padding：预填充到固定长度
padded_ids = [1, 234, 567, 0, 0, 0, 0]  # pad_token_id = 0
attention_mask = [1, 1, 1, 0, 0, 0, 0]

# 动态 padding：batch 内对齐到最长序列
def dynamic_padding(batch, pad_token_id=0):
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in batch]
    masks = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded), torch.tensor(masks)
```

**Packing（样本拼接）**
```python
# 多样本拼接为单序列
sample1 = [1, 234, 567]  # length=3
sample2 = [89, 10, 11, 12]  # length=4
sample3 = [345, 678]  # length=2

packed_sequence = [1, 234, 567, 89, 10, 11, 12, 345, 678]  # total=9
packed_attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 全部为有效 token

# 关键：position_ids 需要重置
position_ids = [0, 1, 2, 0, 1, 2, 3, 0, 1]  # 每个样本重新计数
```

### Loss Mask 实现

**Loss Mask 核心公式**
```python
# 标准 cross-entropy loss
loss = CrossEntropyLoss(logits, labels)  # 不考虑 mask

# Masked cross-entropy loss
loss_per_token = CrossEntropyLoss(reduction='none')(logits, labels)  # [batch, seq_len]
masked_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()  # 只在 mask=1 的位置计算

# loss_mask 定义
loss_mask = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]  # 0=不计算损失, 1=计算损失
```

**多轮对话 Loss Mask 策略**
```python
# 示例：多轮对话
conversation = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你？"},
    {"role": "user", "content": "介绍一下大模型"},
    {"role": "assistant", "content": "大模型是指..."}
]

# 策略1：只计算 assistant token（推荐）
loss_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
            # sys   user  assistant        user   assistant

# 策略2：只计算最后一轮 assistant token
loss_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 策略3：全轮加权（user 也计算损失）
loss_mask = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 权重：user=0.3, assistant=1.0
```

### Tokenizer 实现方式对比

**BPE (Byte Pair Encoding)**
```python
# 核心思想：迭代合并高频字符对
# 步骤：
# 1. 初始化词表为字符级
# 2. 统计所有相邻字符对频次
# 3. 合并最高频字符对为新 token
# 4. 重复 2-3 直到词表达到目标大小

# 示例
"lower" → ['l', 'o', 'w', 'e', 'r']  # 初始
→ ['lo', 'w', 'e', 'r']  # 合并 'l' + 'o'
→ ['low', 'e', 'r']  # 合并 'lo' + 'w'
→ ['low', 'er']  # 合并 'e' + 'r'
```

**Unigram Language Model**
```python
# 核心思想：基于概率模型初始化，迭代删除低概率 token
# 步骤：
# 1. 初始化大词表（候选 token）
# 2. 用 EM 算法估计每个 token 概率
# 3. 计算删除每个 token 的损失
# 4. 删除损失最小的 token
# 5. 重复 3-4 直到词表达到目标大小

# 概率计算
P(token) = count(token) / total_tokens
P(sentence) = ∏ P(token_i)  # 假设 token 独立
```

**WordPiece**
```python
# 核心思想：类似 BPE，但合并依据是语言模型概率
# 特点：使用 ## 前缀标记非词首子词

# 示例
"unwanted" → ['un', '##want', '##ed']
"playing" → ['play', '##ing']

# 合并评分
score(pair) = P(pair) / (P(first) * P(second))
# 选择能最大化语言模型概率的 pair
```

**SentencePiece**
```python
# 核心思想：端到端训练，直接在原始字节流上操作
# 支持 BPE 和 Unigram 两种算法

# 特点：
# 1. 无需预分词（对中文/日文友好）
# 2. 将空格视为特殊符号 ▁ (U+2581)
# 3. 支持子词正则化（subword regularization）

# 示例
"Hello world" → ['▁Hello', '▁world']
"深度学习" → ['▁深度', '学习']  # 中文示例
```

### DeepSeek Tokenizer 创新

**DeepSeek-V2/V3 Tokenizer 特点**
```python
# 1. 字节级 BPE（Byte-level BPE）
# 直接在 UTF-8 字节上训练，覆盖所有 Unicode

# 2. 增量词表（Incremental Vocabulary）
# 预训练基础词表 + 领域增量词表
base_vocab_size = 100000  # 基础词表
domain_vocab = {
    "code": 5000,  # 代码增量词
    "math": 3000,  # 数学增量词
    "chinese": 8000  # 中文增量词
}

# 3. 中文优化
# 高频词直接作为 token，减少序列长度
"人工智能" → ['人工智能']  # 1 token
# 传统 BPE 可能拆分为：
"人工智能" → ['人', '工', '智', '能']  # 4 tokens

# 4. 代码优化
# 常见代码结构作为整体 token
"def __init__(self):" → ['def', ' __init__', '(self):']  # 3 tokens
# 传统 BPE 可能拆分为 10+ tokens
```

## 为什么（2-3 个因果链）

### 1. 为什么需要 Padding/Packing？

**现象**：训练数据样本长度不一（短样本 10 tokens，长样本 4000 tokens），GPU 要求 batch 内张量形状一致  
**根因**：硬件限制 → GPU 并行计算要求张量形状一致 → 无法直接 batch 不等长序列  
**结果**：
- **Padding**：填充到统一长度，简单但浪费计算（padding token 也参与 attention）
- **Packing**：拼接多个样本，高效但需正确处理样本边界（position_ids 重置、attention mask）

### 2. 为什么 Packing 能提升吞吐？

**现象**：使用 packing 后，训练吞吐提升 2-3 倍  
**根因**：
1. **减少无效计算**：padding token 不参与实际计算（attention_mask=0）
2. **提升 GPU 利用率**：相同 batch_size 下，有效 token 数增加 50-80%
3. **降低内存碎片**：避免大量 padding token 占用显存

**结果**：
- **吞吐提升**：从 1000 tokens/s → 2500 tokens/s
- **显存节省**：相同显存下可增大 batch_size
- **训练加速**：整体训练时间缩短 40-60%

### 3. 为什么多轮对话需要 Loss Mask？

**现象**：不加 loss mask，模型会学习预测 user 输入，而非生成 assistant 回复  
**根因**：
1. **训练目标错位**：SFT 目标是让模型学会生成 assistant 回复，而非预测 user 输入
2. **优化方向偏移**：在 user token 上计算损失会降低 assistant token 的学习信号
3. **过拟合 user 模式**：模型可能学会"模仿 user"而非"响应 user"

**结果**：
- **正确做法**：loss_mask 只在 assistant token 上为 1，其余为 0
- **错误做法**：所有 token 都计算损失，导致模型学会预测 user 输入
- **权衡**：只算最后一轮 vs 全轮训练（数据效率 vs 泛化能力）

## 怎么做（可落地步骤）

### 标准做法

#### 1. 动态 Padding 实现

**Step 1: 数据加载与动态 padding**
```python
from transformers import DataCollatorForSeq2Seq

# 使用 Hugging Face 的 DataCollator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # 动态 padding
    pad_to_multiple_of=8,  # 对齐到 8 的倍数（GPU 优化）
    return_tensors="pt"
)

# 或手动实现
def dynamic_padding_collator(batch):
    # batch: List[Dict] with keys 'input_ids', 'attention_mask', 'labels'
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_len - seq_len
        
        # 左填充（生成任务）
        input_ids.append([tokenizer.pad_token_id] * padding_len + item['input_ids'])
        attention_masks.append([0] * padding_len + [1] * seq_len)
        labels.append([-100] * padding_len + item['labels'])  # -100 表示忽略
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_masks),
        'labels': torch.tensor(labels)
    }
```

**Step 2: 配置 DataLoader**
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,
    pin_memory=True
)
```

#### 2. Sample Packing 实现

**Step 1: 定义 Packing 策略**
```python
def pack_samples(samples, max_length=4096):
    """
    将多个短样本拼接为长序列
    
    Args:
        samples: List[Dict] with 'input_ids', 'attention_mask', 'labels'
        max_length: 最大序列长度
    
    Returns:
        packed_batch: Dict with packed sequences
    """
    packed_sequences = []
    current_pack = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'position_ids': [],
        'sample_indices': []  # 记录每个 token 属于哪个样本
    }
    current_length = 0
    sample_idx = 0
    
    for sample in samples:
        sample_len = len(sample['input_ids'])
        
        # 如果当前 pack 放不下，保存并开始新 pack
        if current_length + sample_len > max_length:
            if current_length > 0:
                packed_sequences.append(current_pack.copy())
                current_pack = {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': [],
                    'position_ids': [],
                    'sample_indices': []
                }
                current_length = 0
                sample_idx = 0
        
        # 添加样本到当前 pack
        current_pack['input_ids'].extend(sample['input_ids'])
        current_pack['attention_mask'].extend(sample['attention_mask'])
        current_pack['labels'].extend(sample['labels'])
        current_pack['position_ids'].extend(range(sample_len))
        current_pack['sample_indices'].extend([sample_idx] * sample_len)
        
        current_length += sample_len
        sample_idx += 1
    
    # 添加最后一个 pack
    if current_length > 0:
        packed_sequences.append(current_pack)
    
    return packed_sequences
```

**Step 2: 修改 Attention Mask（关键）**
```python
def create_packed_attention_mask(sample_indices):
    """
    创建 block-diagonal attention mask
    防止不同样本之间的 attention
    
    Args:
        sample_indices: [seq_len], 每个 token 的样本索引
    
    Returns:
        attention_mask: [seq_len, seq_len]
    """
    seq_len = len(sample_indices)
    attention_mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        for j in range(seq_len):
            # 同一样本内的 token 可以互相 attention
            if sample_indices[i] == sample_indices[j]:
                attention_mask[i, j] = 1
    
    return attention_mask

# 优化版本：使用矩阵运算
def create_packed_attention_mask_fast(sample_indices):
    """向量化实现，更快"""
    sample_indices = torch.tensor(sample_indices)
    attention_mask = (sample_indices.unsqueeze(0) == sample_indices.unsqueeze(1)).float()
    return attention_mask
```

**Step 3: 训练循环集成**
```python
# 使用 FlashAttention 的文档级 mask（推荐）
def pack_with_flash_attention(samples, max_length=4096):
    """使用 FlashAttention 的 cu_seqlens"""
    packed = pack_samples(samples, max_length)
    
    # 计算 cumulative sequence lengths
    cu_seqlens = [0]
    for sample in samples:
        cu_seqlens.append(cu_seqlens[-1] + len(sample['input_ids']))
    
    packed['cu_seqlens'] = torch.tensor(cu_seqlens, dtype=torch.int32)
    
    return packed

# 训练时
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

def forward_with_packing(model, packed_batch):
    # FlashAttention 支持 variable-length sequences
    qkv = model.get_qkv(packed_batch['input_ids'])
    output = flash_attn_varlen_qkvpacked_func(
        qkv,
        packed_batch['cu_seqlens'],
        max_seqlen=packed_batch['max_seqlen']
    )
    return output
```

#### 3. 多轮对话 Loss Mask 实现

**策略 1：只计算 assistant token（推荐）**
```python
def create_loss_mask_assistant_only(tokenized_conversation, tokenizer):
    """
    只在 assistant token 上计算损失
    
    Args:
        tokenized_conversation: 包含 'input_ids', 'role_ids' 的字典
        tokenizer: 包含特殊 token 的 tokenizer
    
    Returns:
        loss_mask: [seq_len], 1 表示计算损失，0 表示忽略
    """
    loss_mask = []
    current_role = None
    
    for i, role_id in enumerate(tokenized_conversation['role_ids']):
        if role_id == tokenizer.assistant_role_id:
            loss_mask.append(1)  # assistant token 计算损失
        else:
            loss_mask.append(0)  # 其他 role 不计算损失
    
    return loss_mask

# 实际使用
def tokenize_with_loss_mask(conversation, tokenizer):
    """完整的 tokenization + loss mask 生成"""
    input_ids = []
    labels = []
    loss_mask = []
    
    for turn in conversation:
        role = turn['role']
        content = turn['content']
        
        # Tokenize
        turn_ids = tokenizer.encode(content, add_special_tokens=False)
        
        # 添加 role token
        if role == 'system':
            role_token_id = tokenizer.system_token_id
        elif role == 'user':
            role_token_id = tokenizer.user_token_id
        elif role == 'assistant':
            role_token_id = tokenizer.assistant_token_id
        
        input_ids.append(role_token_id)
        input_ids.extend(turn_ids)
        
        # 创建 labels（shifted by 1）
        labels.append(-100)  # role token 不计算损失
        if role == 'assistant':
            labels.extend(turn_ids)  # assistant 内容作为 labels
            loss_mask.extend([1] * len(turn_ids))
        else:
            labels.extend([-100] * len(turn_ids))  # 其他 role 不作为 labels
            loss_mask.extend([0] * len(turn_ids))
    
    return {
        'input_ids': input_ids[:-1],  # 去掉最后一个 token
        'labels': labels[1:],  # shift
        'loss_mask': loss_mask[1:]  # shift
    }
```

**策略 2：只计算最后一轮 assistant token**
```python
def create_loss_mask_last_turn(conversation, tokenizer):
    """只计算最后一轮 assistant token"""
    loss_mask = [0] * len(conversation['input_ids'])
    
    # 找到最后一轮 assistant 的起始位置
    last_assistant_start = None
    for i in range(len(conversation['role_ids']) - 1, -1, -1):
        if conversation['role_ids'][i] == tokenizer.assistant_role_id:
            last_assistant_start = i
            break
    
    # 标记最后一轮 assistant token
    if last_assistant_start is not None:
        for i in range(last_assistant_start, len(conversation['role_ids'])):
            if conversation['role_ids'][i] == tokenizer.assistant_role_id:
                loss_mask[i] = 1
    
    return loss_mask
```

**策略 3：全轮加权**
```python
def create_loss_mask_weighted(conversation, tokenizer, user_weight=0.3, assistant_weight=1.0):
    """
    全轮训练，user 和 assistant 使用不同权重
    
    Args:
        user_weight: user token 的损失权重
        assistant_weight: assistant token 的损失权重
    """
    loss_mask = []
    
    for role_id in conversation['role_ids']:
        if role_id == tokenizer.assistant_role_id:
            loss_mask.append(assistant_weight)
        elif role_id == tokenizer.user_role_id:
            loss_mask.append(user_weight)
        else:
            loss_mask.append(0)
    
    return loss_mask
```

#### 4. DeepSeek 风格 Tokenizer 训练

**Step 1: 准备多领域语料**
```python
# 语料配比
corpus_distribution = {
    "general": 0.40,  # 通用文本
    "code": 0.20,  # 代码
    "math": 0.15,  # 数学
    "chinese": 0.25  # 中文
}

# 合并语料
import glob

def merge_corpus(corpus_dirs, output_file, distribution):
    """按比例混合语料"""
    with open(output_file, 'w', encoding='utf-8') as outf:
        for corpus_type, ratio in distribution.items():
            files = glob.glob(f"{corpus_dirs[corpus_type]}/*.txt")
            target_lines = int(total_lines * ratio)
            
            # 随机采样
            sampled_files = random.sample(files, min(len(files), target_lines // 1000))
            for file in sampled_files:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if random.random() < ratio:
                            outf.write(line)
```

**Step 2: 训练 Tokenizer**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 初始化字节级 BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 配置训练器
trainer = BpeTrainer(
    vocab_size=128000,  # DeepSeek-V2 词表大小
    special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "<|im_start|>", "<|im_end|>",  # 对话特殊 token
        "<|code|>", "<|math|>",  # 领域特殊 token
    ],
    min_frequency=2,
    limit_alphabet=1000  # 支持更多 Unicode 字符
)

# 训练
tokenizer.train(files=["merged_corpus.txt"], trainer=trainer)

# 保存
tokenizer.save("deepseek_tokenizer.json")
```

**Step 3: 增量训练领域词表**
```python
def incremental_vocab_training(base_tokenizer, domain_corpus, domain_name):
    """
    增量训练领域词表
    
    Args:
        base_tokenizer: 基础 tokenizer
        domain_corpus: 领域语料路径
        domain_name: 领域名称（code/math/chinese）
    """
    # 加载基础 tokenizer
    tokenizer = Tokenizer.from_file(base_tokenizer)
    
    # 统计领域高频 token
    freq = {}
    with open(domain_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenizer.encode(line).tokens
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
    
    # 筛选高频 token（频次 > 100）
    new_tokens = [token for token, count in freq.items() if count > 100]
    
    # 添加到词表
    tokenizer.add_tokens(new_tokens)
    
    # 保存
    tokenizer.save(f"tokenizer_{domain_name}.json")
    
    return len(new_tokens)
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `max_length` | 4096（SFT）<br>8192-32k（长文本） | 根据 GPU 显存和任务需求调整 |
| `padding_side` | "left"（生成）<br>"right"（理解） | 左填充避免 attention 偏移 |
| `pack_max_length` | max_length * 1.2 | 允许轻微超长，提升 packing 效率 |
| `loss_mask_strategy` | assistant_only（推荐）<br>last_turn（数据效率优先）<br>weighted（泛化能力优先） | 根据任务需求选择 |
| `vocab_size` | 100k-150k（多语言）<br>32k-50k（单语言） | 平衡覆盖率与参数量 |
| `min_frequency` | 2-5 | 过低导致噪声，过高丢失低频词 |

### 代码示例：完整 Pipeline

```python
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch

class SFTDataPipeline:
    def __init__(self, tokenizer_name, max_length=4096, use_packing=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_packing = use_packing
    
    def tokenize_conversation(self, conversation):
        """Tokenize 多轮对话"""
        input_ids = []
        labels = []
        loss_mask = []
        
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            
            # 添加 role token
            if role == 'user':
                input_ids.append(self.tokenizer.user_token_id)
            elif role == 'assistant':
                input_ids.append(self.tokenizer.assistant_token_id)
            
            # Tokenize content
            turn_ids = self.tokenizer.encode(content, add_special_tokens=False)
            input_ids.extend(turn_ids)
            
            # 创建 labels 和 loss_mask
            if role == 'assistant':
                labels.extend(turn_ids)
                loss_mask.extend([1] * len(turn_ids))
            else:
                labels.extend([-100] * len(turn_ids))
                loss_mask.extend([0] * len(turn_ids))
        
        # Shift labels
        return {
            'input_ids': input_ids[:-1],
            'labels': labels[1:],
            'loss_mask': loss_mask[1:]
        }
    
    def pack_batch(self, batch):
        """Packing 多个样本"""
        if not self.use_packing:
            return batch
        
        packed = {
            'input_ids': [],
            'labels': [],
            'attention_mask': [],
            'position_ids': []
        }
        
        current_length = 0
        
        for sample in batch:
            sample_len = len(sample['input_ids'])
            
            if current_length + sample_len > self.max_length:
                break
            
            packed['input_ids'].extend(sample['input_ids'])
            packed['labels'].extend(sample['labels'])
            packed['attention_mask'].extend([1] * sample_len)
            packed['position_ids'].extend(range(sample_len))
            
            current_length += sample_len
        
        # Padding to max_length
        padding_len = self.max_length - current_length
        packed['input_ids'].extend([self.tokenizer.pad_token_id] * padding_len)
        packed['labels'].extend([-100] * padding_len)
        packed['attention_mask'].extend([0] * padding_len)
        packed['position_ids'].extend([0] * padding_len)
        
        # Convert to tensors
        return {k: torch.tensor(v) for k, v in packed.items()}
    
    def compute_loss(self, model, batch):
        """计算 masked loss"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
        )
        
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        labels = batch['labels']  # [batch, seq_len]
        loss_mask = torch.tensor(batch['loss_mask'])  # [batch, seq_len]
        
        # 计算每个 token 的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        
        # 应用 mask
        masked_loss = (token_losses * loss_mask).sum() / loss_mask.sum()
        
        return masked_loss

# 使用示例
pipeline = SFTDataPipeline("Qwen/Qwen2-7B", use_packing=True)

# 假设有对话数据
conversations = [
    [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}],
    [{"role": "user", "content": "介绍一下大模型"}, {"role": "assistant", "content": "大模型是指..."}]
]

# Tokenize
tokenized_data = [pipeline.tokenize_conversation(conv) for conv in conversations]

# Pack and create batch
batch = pipeline.pack_batch(tokenized_data)

# Training loop
# loss = pipeline.compute_loss(model, batch)
```

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **静态 Padding** | 实现简单、无需动态计算 | 显存浪费、吞吐低、长序列效率极差 | 小模型调试、固定长度任务 |
| **动态 Padding** | 减少显存占用、提升吞吐 | batch 内长度差异大时仍有效率问题 | 通用 SFT 场景、样本长度分布集中 |
| **Sample Packing** | 吞吐提升 2-3 倍、显存利用率高 | 实现复杂、需正确处理 position_ids 和 attention mask | 大规模训练、短样本为主、显存受限 |
| **只算 Assistant Token** | 目标明确、学习信号强 | 数据效率低（user token 不参与） | 标准 SFT、强调指令遵循 |
| **只算最后一轮** | 数据效率最高、训练快 | 泛化能力弱、可能过拟合最后一轮 | 数据稀缺、快速迭代 |
| **全轮加权** | 数据效率高、泛化能力强 | 超参数调优、可能引入噪声 | 多轮对话理解、需要强泛化能力 |
| **BPE** | 训练快、实现简单、多语言友好 | 词表需预设大小、低频词易 OOV | 通用场景、GPT/LLaMA/Qwen |
| **Unigram** | 概率建模、词表质量高 | 训练慢、需大规模语料 | 需要精细控制词表的场景、T5 |
| **字节级 BPE（DeepSeek）** | 中文/代码效率高、无 OOV | 词表大（100k+）、embedding 参数多 | 多语言、代码、长文本场景 |

## 高频追问（至少 5 个）

### 1. Q: Padding 和 Packing 对训练效果有什么影响？
**A:**
- **Padding**：padding token 参与 attention 计算（除非 mask），会引入噪声；但实现简单，不影响模型结构
- **Packing**：不同样本拼接，需正确处理 attention mask 和 position_ids，否则样本间会互相干扰
- **关键**：Packing 必须使用 block-diagonal attention mask 或 FlashAttention 的 variable-length 支持，否则模型会学习到跨样本的错误依赖
- **实践**：Packing 需要仔细调试，建议先用 padding 验证效果，再切换到 packing 提升吞吐

### 2. Q: 多轮对话训练，只算最后一轮和全轮训练有什么区别？
**A:**
- **只算最后一轮**：
  - 优点：数据效率高（一个样本只训练一轮），训练快
  - 缺点：泛化能力弱，模型可能只学会最后一轮的回复模式
  - 适用：数据稀缺、快速迭代、对多轮理解要求不高
- **全轮训练**：
  - 优点：数据利用率高（一个样本训练多轮），泛化能力强
  - 缺点：训练慢、可能过拟合早期轮次
  - 适用：多轮对话理解任务、需要强泛化能力
- **推荐**：标准 SFT 用"只算 assistant token"，多轮理解任务用"全轮加权"（user=0.3, assistant=1.0）

### 3. Q: DeepSeek 的 tokenizer 相比 LLaMA 的 BPE 有什么创新？
**A:**
- **字节级 BPE**：直接在 UTF-8 字节上训练，覆盖所有 Unicode，无 OOV
- **增量词表**：预训练基础词表 + 领域增量词表（code/math/chinese），支持灵活扩展
- **中文优化**：高频中文词直接作为 token，序列长度减少 40%（相比字符级）
- **代码优化**：常见代码结构（`def __init__`）作为整体 token，代码 tokenization 效率提升 50%
- **词表大小**：DeepSeek-V2 使用 100k 词表，LLaMA-2 使用 32k 词表，DeepSeek 在多语言场景下优势明显

### 4. Q: Packing 时如何处理不同样本的 attention？
**A:**
- **错误做法**：直接拼接，不处理 attention mask → 不同样本互相 attention → 模型学习错误依赖
- **正确做法 1**：Block-diagonal attention mask
  ```python
  # 创建块对角 attention mask
  attention_mask = torch.zeros(seq_len, seq_len)
  for i, sample_idx_i in enumerate(sample_indices):
      for j, sample_idx_j in enumerate(sample_indices):
          if sample_idx_i == sample_idx_j:
              attention_mask[i, j] = 1
  ```
- **正确做法 2**：FlashAttention variable-length（推荐）
  ```python
  from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
  
  # 使用 cu_seqlens 标记样本边界
  output = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen)
  ```
- **性能对比**：FlashAttention 实现 2-3 倍加速，显存占用降低 50%

### 5. Q: 动态 padding 时，batch_size 如何选择？
**A:**
- **传统做法**：固定 batch_size（如 8），batch 内动态 padding
- **问题**：batch 内长度差异大时，短样本浪费大量显存（padding token）
- **优化 1**：按长度分桶（bucket），每个 bucket 内 padding
  ```python
  # 将样本按长度分桶
  buckets = {0-512: [], 512-1024: [], 1024-2048: [], 2048-4096: []}
  for sample in samples:
      bucket = get_bucket(len(sample))
      buckets[bucket].append(sample)
  ```
- **优化 2**：动态 batch_size（根据序列长度调整）
  ```python
  # 短序列用大 batch，长序列用小 batch
  if avg_seq_len < 512:
      batch_size = 32
  elif avg_seq_len < 1024:
      batch_size = 16
  else:
      batch_size = 8
  ```
- **推荐**：使用 Hugging Face 的 `DataCollatorForSeq2Seq`，自动处理动态 padding

### 6. Q: Tokenizer 训练时，词表大小如何选择？
**A:**
- **考虑因素**：
  1. **语言覆盖**：多语言需要大词表（100k+），单语言可以小词表（30k-50k）
  2. **领域支持**：代码/数学/特殊符号需要额外 token
  3. **显存占用**：Embedding 参数 = vocab_size × hidden_dim
     - 7B 模型：32k 词表 → 128M 参数（~500MB FP32）
     - 7B 模型：100k 词表 → 400M 参数（~1.6GB FP32）
  4. **序列长度**：大词表 → token 数少 → 序列短 → 计算快
- **推荐值**：
  - 英文为主：32k-50k（LLaMA-2: 32k）
  - 多语言：100k-150k（Qwen2: 151k, DeepSeek-V2: 100k）
  - 代码+多语言：120k-150k
- **调优**：检查 tokenization 效率（字符/token 比率），中文 > 1.5，英文 > 3.0

### 7. Q: Loss mask 中，为什么要用 -100 而不是 0？
**A:**
- **PyTorch CrossEntropyLoss 默认行为**：`ignore_index=-100`
  ```python
  torch.nn.CrossEntropyLoss(ignore_index=-100)
  ```
- **原因**：
  1. `-100` 不在词表范围内（0 到 vocab_size-1），不会与真实 label 混淆
  2. PyTorch 会自动跳过 `ignore_index` 的位置，不计算梯度
  3. 标准约定，所有主流框架（Transformers, Megatron）都使用 `-100`
- **错误做法**：用 `0` 作为 ignore_index → 如果词表中有 ID=0 的 token（通常是 padding token），会误伤有效 token

## 常见错误（至少 3 个）

### 1. **错误：Packing 时未正确处理 position_ids**
**描述**：多个样本拼接后，position_ids 连续计数（0, 1, 2, ..., 99），导致第二个样本的 position_id 从 100 开始，而非 0  
**后果**：模型认为第二个样本的 token 在位置 100-199，position embedding 错误，学习到错误的位置依赖  
**正确做法**：
```python
# 错误：连续 position_ids
position_ids_wrong = list(range(total_length))  # [0, 1, 2, ..., 99]

# 正确：每个样本重置 position_ids
position_ids_correct = []
for sample in samples:
    position_ids_correct.extend(range(len(sample)))
# 结果：[0, 1, 2, 0, 1, 2, 3, 0, 1]
```

### 2. **错误：Packing 时不同样本互相 attention**
**描述**：拼接多个样本后，未使用 block-diagonal attention mask，导致样本 A 的 token 可以 attention 到样本 B 的 token  
**后果**：模型学习到跨样本的错误依赖，训练效果下降，可能生成不连贯的内容  
**正确做法**：
```python
# 错误：全局 attention
attention_mask_wrong = torch.ones(seq_len, seq_len)

# 正确：block-diagonal attention
def create_block_diagonal_mask(sample_indices):
    sample_indices = torch.tensor(sample_indices)
    return (sample_indices.unsqueeze(0) == sample_indices.unsqueeze(1)).float()

attention_mask_correct = create_block_diagonal_mask([0, 0, 0, 1, 1, 1, 1, 2, 2])
# 结果：前 3 个 token 互相 attention，中间 4 个互相 attention，最后 2 个互相 attention
```

### 3. **错误：多轮对话训练时，在 user token 上也计算损失**
**描述**：训练多轮对话模型时，将所有 token（包括 user token）都作为 labels，导致模型学习预测 user 输入  
**后果**：模型可能学会"模仿 user"而非"响应 user"，生成质量下降  
**正确做法**：
```python
# 错误：所有 token 都作为 labels
labels_wrong = input_ids.copy()

# 正确：只在 assistant token 上计算损失
labels_correct = []
for i, token_id in enumerate(input_ids):
    if role_ids[i] == assistant_role_id:
        labels_correct.append(token_id)
    else:
        labels_correct.append(-100)  # ignore_index
```

### 4. **错误：动态 padding 时使用右填充，但任务是生成**
**描述**：生成任务使用右填充（padding token 在序列末尾），导致模型 attention 到 padding token  
**后果**：生成质量下降，尤其是批量推理时，可能生成重复或不连贯的内容  
**正确做法**：
```python
# 错误：生成任务使用右填充
input_ids_wrong = [1, 234, 567, 0, 0, 0]  # padding 在右侧
attention_mask_wrong = [1, 1, 1, 0, 0, 0]

# 正确：生成任务使用左填充
input_ids_correct = [0, 0, 0, 1, 234, 567]  # padding 在左侧
attention_mask_correct = [0, 0, 0, 1, 1, 1]

# 配置
tokenizer.padding_side = "left"  # 生成任务
```

### 5. **错误：Tokenizer 训练语料与模型训练语料不一致**
**描述**：Tokenizer 在通用语料训练，但模型在代码/数学语料微调，导致代码符号 tokenization 效率低  
**后果**：代码 token 被拆分为多个字符 token（如 `def` → `d`, `e`, `f`），序列长度增加，模型理解能力下降  
**正确做法**：
- Tokenizer 训练语料应覆盖所有领域（通用+代码+数学+特殊符号）
- 或使用已有的多领域 tokenizer（Qwen/DeepSeek 已包含代码支持）
- 检查 tokenization 效率：
  ```python
  code_text = "def __init__(self):"
  tokens = tokenizer.encode(code_text)
  print(f"Token 数: {len(tokens)}")  # 理想值: 3-5, 差值: 10+
  ```

## 反问面试官的问题

### 技术深度类
1. **"贵团队在训练大规模模型时，packing 策略有没有遇到什么坑？比如显存碎片、attention mask 实现等，是如何解决的？"**  
   （了解实际工程经验，判断团队对训练优化的重视程度）

2. **"对于多轮对话训练，你们采用的是哪种 loss mask 策略？有没有对比过不同策略的效果差异？"**  
   （展示对细节的关注，了解团队的技术选型思路）

3. **"DeepSeek 的 tokenizer 在代码和中文场景下效果很好，你们在 tokenizer 训练时有没有类似的优化实践？比如增量词表、领域适配等。"**  
   （结合前沿技术，展示对细节的理解）

### 业务场景类
1. **"如果要在现有模型基础上支持一个新的垂直领域（如医疗/法律），tokenizer 和 packing 策略需要做哪些调整？从成本和效果角度如何权衡？"**  
   （展示对业务落地的理解，结合技术选型）

2. **"线上训练时，数据预处理（tokenization + packing）的瓶颈通常在哪里？有没有做过优化（如并行化、缓存、GPU 加速）？"**  
   （关注工程实践，展示对训练全链路的理解）

## 自测题

### 口述
- **能流畅讲清楚**：
  1. Padding vs Packing 的原理、优劣和适用场景
  2. Packing 时如何正确处理 position_ids 和 attention mask
  3. 多轮对话的三种 loss mask 策略及各自的适用场景
  4. BPE、Unigram、WordPiece、SentencePiece 四者的核心差异
  5. DeepSeek tokenizer 相比传统 BPE 的创新点
  6. Tokenizer 词表大小的选择依据和权衡
  7. 动态 padding 的实现细节和优化方法

### 手写
- **5 分钟能写出**：
  1. **动态 padding 的 collate function**
  ```python
  def dynamic_padding_collator(batch, pad_token_id=0):
      max_len = max(len(item['input_ids']) for item in batch)
      input_ids = [item['input_ids'] + [pad_token_id] * (max_len - len(item['input_ids'])) 
                   for item in batch]
      attention_mask = [[1] * len(item['input_ids']) + [0] * (max_len - len(item['input_ids'])) 
                        for item in batch]
      return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}
  ```
  
  2. **Packing 的 position_ids 重置逻辑**
  ```python
  def reset_position_ids(sample_lengths):
      position_ids = []
      for length in sample_lengths:
          position_ids.extend(range(length))
      return position_ids
  ```
  
  3. **Block-diagonal attention mask**
  ```python
  def create_block_diagonal_mask(sample_indices):
      sample_indices = torch.tensor(sample_indices)
      return (sample_indices.unsqueeze(0) == sample_indices.unsqueeze(1)).float()
  ```
  
  4. **多轮对话 loss mask（只算 assistant）**
  ```python
  def create_loss_mask(role_ids, assistant_role_id):
      return [1 if role_id == assistant_role_id else 0 for role_id in role_ids]
  ```
  
  5. **Masked cross-entropy loss**
  ```python
  def masked_cross_entropy_loss(logits, labels, loss_mask):
      loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
      token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
      token_losses = token_losses.view(labels.shape)
      return (token_losses * loss_mask).sum() / loss_mask.sum()
  ```

## 标签
#SFT #字节 #阿里

## 相关文档
- [[01-数据处理全链路]] - 数据清洗、格式化与 tokenization 在预处理中的应用
- [[../01-Transformer基础/04-Tokenizer与Embedding]] - Tokenizer 算法详解与 Embedding 层实现
- [[../03-SFT与LoRA/01-SFT目标与边界]] - SFT 训练目标与 tokenization 的关系
- [[../04-RLHF对齐/05-偏好数据设计]] - DPO 数据格式与多轮对话处理
- [[../10-FlashAttention/02-FlashAttention工程]] - FlashAttention variable-length 支持与 packing 实现
