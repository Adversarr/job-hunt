# Tokenizer 与 Embedding

## 一句话结论
Tokenizer 通过 BPE/Unigram/WordPiece 等算法将文本切分为 token ID 序列，Embedding 层通过查表将 ID 映射为稠密向量；两者是 LLM 输入处理的核心组件，直接影响模型对多语言/特殊符号的支持能力。

## 核心定义/公式

### Tokenizer 类型详解

#### BPE (Byte Pair Encoding)

**是什么**：基于频次的贪心合并算法。从字符（或字节）级别开始，把语料里出现最频繁的相邻 token 对不断合并，直到达到目标词表大小。

**算法示例**：
```
初始: ('t', 'h'), ('h', 'e') -> 频次统计
第1轮: ('t','h') 频次最高 -> 合并为 'th'
第2轮: ('th','e') 频次最高 -> 合并为 'the'
最终词表包含: t, h, e, th, the, ...
```

**Byte-level BPE**：现代 LLM 常见的做法，先把文本变成字节（0–255），再做 BPE。对多语言、emoji、未知字符更鲁棒。

**为什么用它**：
- 简单、快、工程成熟
- 开放词表（open-vocabulary）：遇到新词也能拆成子词/字节，不会 OOV
- 对英文及混合文本很强，尤其 byte-level 让跨语言更鲁棒

**关键超参**：
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `vocab_size` | 32k-100k | 越大越"整词化"，序列更短但词表更大 |
| `byte_level` | True | 多语言/噪声场景强烈建议启用 |
| `min_frequency` | 2-5 | 合并阈值，过低导致噪声 token |

---

#### Unigram Language Model（SentencePiece 默认）

**是什么**：把"分词"视为概率模型。先有一个候选子词集合（通常很大），每个子词有概率 p(token)。对一个句子，存在多种切分方式，模型用概率来评估切分。训练时会迭代删除贡献小/概率低的 token，把词表缩到目标大小。

**核心思想**：不是"合并出词表"，而是"从大词表里裁剪出最好的一套"。

**训练流程**：
1. **生成超大候选词表**：从语料提取大量子串，初始词表可能是目标的 5–20 倍
2. **EM 迭代**：
   - E 步：对每个句子，计算在当前 p(token) 下的最可能切分（Viterbi）
   - M 步：根据切分统计更新每个 token 的概率
3. **裁剪（pruning）**：删除一批贡献度最低的 token
4. 重复直到词表达到目标大小

**为什么用它**：
- 全局更稳：不是贪心合并，而是用概率评估 token 对整体语料的解释能力
- 更容易精确控制词表大小：通过裁剪到 vocab_size 很自然
- 更适合非空格语言/多粒度（日文、中文）

**关键超参**：
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `seed_sentencepiece_size` | 目标×5-20 | 候选生成规模影响质量与时间 |
| `character_coverage` | 0.9995 | 多语言尤其重要（覆盖稀有字符比例） |
| `shrinking_factor` | 0.75 | 每轮裁剪比例，影响收敛速度 |

---

#### WordPiece（BERT 系）

**是什么**：和 BPE 一样是逐步构造子词词表，但合并标准不是纯频次，而是让语言模型似然提升最大。常见实现评分公式：

```
score(x,y) = count(xy) / (count(x) × count(y))
```

**## 标记**：`playing -> play + ##ing`，`##` 表示该子词只能出现在词中间/结尾。

**为什么用它**：
- 更贴近语言建模目标：合并不是"出现多就合"，而是"合了以后更有用"
- 对英文词形变化友好：`##ing`、`##ed` 等后缀体系自然
- BERT 生态成熟

**关键差异**：
| 特性 | BPE | WordPiece |
|------|-----|-----------|
| 合并依据 | 频次 | 语言模型似然增益 |
| 子词标记 | 无 | `##` 前缀 |
| 编码策略 | 贪心合并 | 最长匹配优先 |
| 代表模型 | GPT/LLaMA | BERT/RoBERTa |

---

### Embedding 层实现

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # 初始化：标准正态分布 * 0.02
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        return self.embedding(input_ids)

# Tie Embedding: 输入 embedding 和输出 projection 共享权重
class TiedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
        # 共享权重
        self.output_projection.weight = self.embedding.weight
```

## 为什么（2-3 个因果链）

### 1. 为什么需要 Tokenizer 而非直接字符级输入？
**现象**：字符级词表虽小（~100），但序列极长，计算开销大  
**根因**：Transformer 复杂度 $O(n^2)$，序列长度直接决定计算成本  
**结果**：Tokenizer 平衡词表大小与序列长度，将常用词/子词编码为单个 token，减少序列长度 3-5 倍，大幅提升训练推理效率

### 2. 为什么 Embedding 层通常参与微调？
**现象**：微调时 embedding 权重会更新  
**根因**：预训练词表可能缺少领域术语/特殊符号，embedding 需要学习领域语义；不微调会限制模型对新词的理解能力  
**结果**：全参微调通常更新 embedding；但参数量大时（如 70B+），可冻结 embedding 节省显存，代价是领域适应能力下降

### 3. 为什么 BPE/Unigram 更适合多语言？
**现象**：英文 tokenizer 对中文效果差，单个汉字一个 token  
**根因**：BPE 按字节/字符频次合并，天然支持任意 Unicode；Unigram 基于统计初始化，对低频语言更鲁棒  
**结果**：Qwen/DeepSeek 使用定制 BPE，中文 tokenization 效率提升 40%+（相同语义下 token 数更少）

## 怎么做（可落地步骤）

### 完整训练流程

#### Phase 1: 准备阶段

**Step 1: 定义目标与约束**
```
□ 语言覆盖：中文/英文/代码/emoji 占比
□ 目标词表大小：32k / 50k / 100k（代码多可偏大）
□ 特殊符号：<pad> <unk> <bos> <eos> <mask>
□ 算法选择：BPE（GPT类）/ Unigram（精细控制）/ WordPiece（BERT类）
```

**Step 2: 语料准备与清洗**
```python
# 多语言语料混合（示例比例）
corpus_config = {
    "zh": 0.40,      # 中文
    "en": 0.40,      # 英文
    "code": 0.15,    # 代码
    "math/special": 0.05  # 数学/特殊符号
}

# 清洗要点
# - 去重：MinHash/LSH 去重
# - 过滤：极端乱码、低质量文本
# - 均衡：避免单一域污染词表
```

#### Phase 2: Tokenizer 训练

**BPE 训练（GPT/LLaMA 类）**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 初始化
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# Byte-level 预分词（强烈推荐多语言场景）
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 训练器配置
trainer = BpeTrainer(
    vocab_size=32000,           # 目标词表大小
    min_frequency=2,            # 最小出现频次
    special_tokens=[
        "<pad>", "<unk>", "<bos>", "<eos>",
        "<|im_start|>", "<|im_end|>"  # 对话模板
    ],
    # Byte-level BPE 关键：在字节流上操作
    initial_alphabet=ByteLevel.alphabet()  # 256 个字节
)

# 训练
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("tokenizer-bpe.json")
```

**Unigram 训练（SentencePiece）**
```python
import sentencepiece as spm

# 训练命令
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tokenizer-unigram",
    model_type="unigram",           # 算法类型
    vocab_size=32000,
    character_coverage=0.9995,      # 字符覆盖率（多语言关键）
    input_sentence_size=10000000,   # 采样句子数
    seed_sentencepiece_size=100000, # 初始候选词表大小
    shrinking_factor=0.75,          # 每轮裁剪比例
    num_threads=48,
    # 特殊符号
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=["<|im_start|>", "<|im_end|>"]
)

# 加载使用
sp = spm.SentencePieceProcessor()
sp.load("tokenizer-unigram.model")
```

**WordPiece 训练（BERT 类）**
```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

# 初始化（通常需要预分词）
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 英文预分词：按空格和标点
tokenizer.pre_tokenizer = Whitespace()  # 或 BertPreTokenizer

trainer = WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    continuing_subword_prefix="##"  # 子词前缀标记
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

#### Phase 3: 离线评估

```python
def evaluate_tokenizer(tokenizer, test_corpus):
    """Tokenizer 质量评估"""
    results = {
        "avg_tokens_per_char": [],  # 压缩率
        "unk_rate": [],             # OOV 率
        "max_token_length": [],     # 最长 token
        "special_coverage": {}      # 特殊符号覆盖
    }
    
    for text in test_corpus:
        # 编码
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens if hasattr(encoded, 'tokens') else encoded
        
        # 指标计算
        results["avg_tokens_per_char"].append(
            len(tokens) / len(text)
        )
        
        # OOV 统计
        if hasattr(tokenizer, 'unk_token_id'):
            unk_count = tokens.count(tokenizer.unk_token_id)
            results["unk_rate"].append(unk_count / len(tokens))
    
    # 理想指标（中文）
    # - 字符/token 比率 > 1.5（越低越好）
    # - OOV 率 < 0.1%（byte-level 应为 0）
    
    return {
        "compression_ratio": 1 / np.mean(results["avg_tokens_per_char"]),
        "avg_unk_rate": np.mean(results["unk_rate"])
    }
```

#### Phase 4: 冻结与版本化

```python
# 保存完整配置（必须包含 normalization 规则）
tokenizer.save("tokenizer-final.json")

# 版本化清单
# □ vocab.json / merges.txt（BPE）
# □ tokenizer.model（SentencePiece）
# □ special_tokens_map.json
# □ tokenizer_config.json
# □ 训练语料 hash（用于复现）
```

### Embedding 层配置

**关键参数：**
- `vocab_size`：词表大小（LLaMA-2: 32000, Qwen2: 151643, GPT-4: ~100000）
- `hidden_dim`：隐藏层维度（7B: 4096, 70B: 8192）
- `tie_embedding`：是否共享输入输出权重（节省参数，但可能影响表达）

**显存估算：**
```python
# Embedding 参数量
vocab_size = 32000
hidden_dim = 4096
params = vocab_size * hidden_dim  # 131M 参数

# FP32 显存
memory_fp32 = params * 4 / 1024**2  # ~500 MB
# FP16 显存
memory_fp16 = params * 2 / 1024**2  # ~250 MB
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `vocab_size` | 32000-100000 | 平衡覆盖率与参数量；中文需更大（Qwen: 150k+） |
| `min_frequency` | 2-5 | 过低导致噪声 token，过高丢失低频词 |
| `tie_embedding` | True（小模型）<br>False（大模型） | 小模型省参数，大模型需独立表达能力 |
| `padding_side` | "left"（生成）<br>"right"（理解） | 左填充避免 attention 偏移 |

### 代码示例：完整 Tokenizer + Embedding Pipeline

```python
from transformers import AutoTokenizer
import torch
import torch.nn as nn

# 加载预训练 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
print(f"词表大小: {len(tokenizer)}")  # 151643

# 文本 → token IDs
text = "大模型训练需要大量数据和算力"
input_ids = tokenizer.encode(text, return_tensors="pt")
print(f"Token 数: {input_ids.shape[1]}")  # 如: 12 tokens

# Embedding 层
embedding_layer = nn.Embedding(len(tokenizer), 4096)
embeddings = embedding_layer(input_ids)
print(f"Embedding 形状: {embeddings.shape}")  # [1, 12, 4096]

# 检查特殊 token 处理
special_tokens = tokenizer.special_tokens_map
print(f"特殊 token: {special_tokens}")
# {'pad_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', ...}
```

## 权衡分析

### 算法选择决策树

```
你在训练什么模型？
│
├─ GPT/通用 LLM（多语言+代码+长文本）
│  ├─ 优先：byte-level BPE 或 SentencePiece Unigram
│  ├─ 实现简单、延续 GPT 生态 → byte-level BPE
│  └─ 更平衡的子词粒度、中日文更好 → Unigram
│
├─ BERT 风格 MLM（英文为主、词形规则重要）
│  └─ 优先：WordPiece
│
└─ 非常在意词表大小固定且质量稳定
   └─ Unigram（候选大 → 剪到目标）
```

### 场景对照表

| 场景 | 推荐算法 | 推荐词表大小 | 关键配置 |
|------|----------|--------------|----------|
| **英文为主 GPT** | Byte-level BPE | 32k-50k | `byte_level=True` |
| **中英混合 GPT** | Byte-level BPE | 100k-150k | 中文语料 > 40% |
| **代码-heavy** | Byte-level BPE | 100k+ | 代码语料 > 30% |
| **日文/中文 NLP** | Unigram | 32k-64k | `character_coverage=0.9995` |
| **BERT/RoBERTa** | WordPiece | 30k-50k | `##` 前缀，预分词 |
| **资源受限** | BPE（非 byte-level）| 16k-32k | 牺牲多语言支持 |

### 算法详细对比

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **BPE (Byte-level)** | 实现简单、训练快、对多语言鲁棒、无 OOV | 词表需人工设定大小、低频词粒度可能过细 | **通用场景首选**（GPT/LLaMA/Qwen/DeepSeek） |
| **Unigram** | 概率建模全局最优、易控制词表质量、删除低频 token | 训练慢、EM 迭代复杂、需大规模语料 | 需要精细词表控制、非空格语言（T5） |
| **WordPiece** | 子词粒度合理、LM 目标驱动、英文词形友好 | 对中文支持差、需要 ## 前缀、需预分词 | 英文为主的理解任务（BERT） |
| **字符级** | 无 OOV、词表极小（~100） | 序列极长、计算开销大、语义稀疏 | 极小词表、特殊领域符号 |

### 关键超参影响

| 超参 | 过大影响 | 过小影响 | 推荐调优方法 |
|------|----------|----------|--------------|
| `vocab_size` | embedding 参数量大、训练慢 | 序列长、OOV 多 | 32k（英文）/ 100k+（多语言） |
| `min_frequency` | 丢失低频但重要的词 | 噪声 token 多、词表稀疏 | 2-5，根据语料规模调整 |
| `character_coverage` | 冗余 token 多 | 稀有字符 OOV | 0.9995（多语言）/ 1.0（单语言） |
| `byte_level` | 词表基数 256，初期合并慢 | 遇到未知 Unicode 会 OOV | **多语言必开** |

### Embedding 层方案对比

## 高频追问（至少 5 个）

### 1. Q: BPE、Unigram、WordPiece 三者的核心差异是什么？
**A:** 

| 维度 | BPE | Unigram | WordPiece |
|------|-----|---------|-----------|
| **核心思想** | 频次驱动，贪心合并 | 概率模型，迭代裁剪 | LM 似然增益驱动 |
| **词表构建** | 从小到大合并 | 从大到小裁剪 | 从小到大添加 |
| **合并标准** | `freq(pair)` 最高 | `P(token)` 贡献度 | `score(x,y)=count(xy)/(count(x)*count(y))` |
| **编码策略** | 贪心应用 merge 规则 | Viterbi 最优路径 | 最长匹配优先 + ## 前缀 |
| **子词标记** | 无 | 无 | `##` 前缀标记非首子词 |
| **代表模型** | GPT/LLaMA/Qwen/DeepSeek | T5/ALBERT | BERT/RoBERTa |

**一句话总结**：BPE 简单快速，Unigram 全局最优，WordPiece LM 驱动；现代 LLM 多用 byte-level BPE。

### 2. Q: 为什么 byte-level BPE 对多语言更鲁棒？
**A:**
- **字节基础**：任何 Unicode 字符都可表示为 1-4 个字节（UTF-8），256 个基础单元覆盖所有语言
- **无 OOV**：即使遇到训练时未见过的字符/emoji，也能拆成字节序列，不会 `<unk>`
- **跨语言共享**：不同语言的相似字节模式可以共享子词，如中文"的"和日文"の"可能有共同字节级特征
- **实际案例**：LLaMA-2 用 byte-level BPE，Qwen/DeepSeek 在 byte-level 基础上定制中文词表

### 3. Q: Embedding 层是否会参与微调？为什么？
**A:** 
- **全参微调**：通常参与，原因：(1) 领域术语需要学习新语义 (2) 特殊符号需要适配 (3) 预训练词表可能不完整
- **LoRA 微调**：通常冻结，原因：(1) LoRA 只加在 attention/FFN (2) Embedding 参数量大（~300MB），冻结可省显存 (3) 预训练 embedding 已有足够语义
- **场景差异**：领域数据差异大（医疗/法律）建议微调；通用对话场景可冻结

### 4. Q: 中文 tokenizer 常见坑有哪些？
**A:**
- **长度膨胀**：英文 tokenizer 对中文按字符切分，"深度学习" → 4 tokens（应为 1-2），序列长度翻倍
- **标点符号**：中文标点 `，。！？` 和英文不同，未覆盖时会拆分为多个 token
- **空格处理**：中文无空格分词，需依赖预分词（jieba）或字节级 BPE
- **特殊符号**：Emoji、数学符号、生僻字易 OOV
- **解决方案**：使用定制 tokenizer（Qwen/DeepSeek），或 SentencePiece + 大词表（100k+）

### 5. Q: 如何评估 tokenizer 质量？关键指标有哪些？
**A:**
```python
# 核心评估指标
def evaluate_tokenizer(tokenizer, corpus):
    metrics = {
        # 1. 压缩率（字符数/token 数）
        "compression_ratio": sum(len(t) for t in corpus) / sum(
            len(tokenizer.encode(t)) for t in corpus
        ),
        
        # 2. OOV 率（byte-level 应为 0）
        "unk_rate": count_unk(corpus) / total_tokens,
        
        # 3. 词表利用率
        "vocab_utilization": len(used_tokens) / vocab_size,
        
        # 4. 领域覆盖率（代码/数学/特殊符号）
        "domain_coverage": check_special_tokens(corpus)
    }
    return metrics
```

**理想阈值**：
- 中文压缩率 > 1.5（越高越好）
- OOV 率 < 0.1%（byte-level 目标 0%）
- 词表利用率 > 80%

### 6. Q: Unigram 的 EM 算法具体是怎么工作的？
**A:**
**E 步（期望）**：对每个句子，基于当前 token 概率计算最优切分
```
P(切分) = ∏ P(token_i)  # 假设 token 独立
用 Viterbi 算法找最大概率切分
```

**M 步（最大化）**：根据切分统计更新 token 概率
```
count(token) = 所有切分中该 token 出现次数的期望
P(token) = count(token) / total_count
```

**裁剪**：删除对整体似然贡献最小的 token，重复 EM 直到词表大小达标。

### 7. Q: 为什么有的模型用 150k 词表（Qwen），有的只用 32k（LLaMA）？
**A:**
- **Qwen 150k**：多语言支持，中文/代码/数学符号需大量 token；收益：中文 tokenization 效率高、序列短；代价：embedding 参数大（600MB vs 250MB）
- **LLaMA 32k**：英文为主，词表小；收益：参数少、训练快；代价：中文支持差、易 OOV
- **选择依据**：多语言场景用大词表（100k+），单语言/资源受限用小词表（30k-50k）

### 8. Q: Tokenizer 训练时 min_frequency 设多少合适？
**A:**
- **推荐值**：2-5（取决于语料规模）
- **过低（=1）**：噪声 token 多、词表稀疏、过拟合罕见词
- **过高（>10）**：丢失低频但重要的词（专业术语）、增加 OOV 风险
- **调优方法**：(1) 检查词表覆盖率（test set OOV rate） (2) 观察 token 数/序列长度分布 (3) 小模型用较小 min_frequency 补偿词表

### 9. Q: Tie embedding 会影响模型效果吗？什么场景不建议 tie？
**A:**
- **理论影响**：输入输出共享权重，限制了输出层的表达能力（输入要理解语义，输出要生成词表分布，两者目标不完全一致）
- **实际表现**：小模型（<7B）tie 影响小，大模型（>70B）tie 可能降低生成质量
- **不建议 tie**：(1) 大模型（参数充足） (2) 领域适配（输出需学习领域术语） (3) 多任务（不同任务输入输出语义差异大）

## 常见错误（至少 3 个）

### 1. **错误：用英文 tokenizer 处理中文，序列长度爆炸**
**描述**：使用 LLaMA tokenizer 处理中文文本，"深度学习模型训练" 被拆分为 8 个 token（每个汉字一个），而 Qwen tokenizer 只需 2-3 个 token。  
**后果**：序列长度翻倍，计算开销 O(n²) 增大 4 倍，推理延迟显著上升。  
**正确做法**：
- 使用中文优化的 tokenizer（Qwen/DeepSeek/ChatGLM）
- 或用 SentencePiece 在中文语料上重新训练
- 评估 tokenizer 效率：`len(text) / len(tokenizer.encode(text))`

### 2. **错误：微调时冻结 embedding，但添加了新 token**
**描述**：在 LoRA 微调中添加了领域特殊 token（如 `<|medical|>`），但 embedding 层冻结，导致新 token embedding 随机初始化且不更新。  
**后果**：模型无法理解新 token，生成质量下降。  
**正确做法**：
```python
# 添加新 token 后，必须更新 embedding
tokenizer.add_tokens(["<|medical|>", "<|legal|>"])
model.resize_token_embeddings(len(tokenizer))

# 微调时解冻 embedding 或单独训练新 token
model.get_input_embeddings().weight.requires_grad = True
```

### 3. **错误：忽略 padding_side 对 attention 的影响**
**描述**：生成任务使用 right padding，导致 attention mask 计算错误（padding token 被 attention 到）。  
**后果**：生成质量下降，尤其是批量推理时。  
**正确做法**：
```python
# 生成任务用 left padding
tokenizer.padding_side = "left"  # 批量生成
tokenizer.pad_token = tokenizer.eos_token

# 理解任务（分类）用 right padding
tokenizer.padding_side = "right"  # 分类任务
```

### 4. **错误：Tokenizer 训练语料与模型训练语料不一致**
**描述**：Tokenizer 在通用语料训练，但模型在代码/数学语料微调，导致代码符号 tokenization 效率低。  
**后果**：代码 token 被拆分为多个字符 token，序列长度增加，模型理解能力下降。  
**正确做法**：
- Tokenizer 训练语料应覆盖所有领域（通用+代码+数学+特殊符号）
- 或使用已有的多领域 tokenizer（Qwen/DeepSeek 已包含代码支持）

### 5. **错误：未处理特殊符号导致 OOV**
**描述**：训练数据包含 Emoji、数学符号、生僻字，但 tokenizer 未覆盖，导致大量 `[UNK]` token。  
**后果**：模型无法理解特殊符号，生成质量下降。  
**正确做法**：
- 检查 tokenizer 覆盖率：`coverage = len(known_tokens) / len(all_tokens)`
- 使用字节级 BPE（SentencePiece 默认）覆盖所有 Unicode
- 或添加特殊 token 到词表

## 反问面试官的问题

### 技术深度类
1. **"贵团队在 tokenizer 选择上遇到过哪些坑？比如多语言支持、领域适配等，是如何解决的？"**  
   （了解实际工程经验，判断团队对 tokenizer 的重视程度）

2. **"对于超长上下文（100k+ tokens），tokenizer 的效率瓶颈在哪里？是否有考虑过动态词表或稀疏 embedding？"**  
   （展示对长上下文+tokenizer 结合的思考，深入技术细节）

### 业务场景类
1. **"如果要在现有模型基础上支持一个新的垂直领域（如医疗/法律），tokenizer 和 embedding 需要做哪些适配？从成本和效果角度如何权衡？"**  
   （展示对业务落地的理解，结合技术选型）

2. **"线上推理时，tokenizer 的速度是否会成为瓶颈？是否有做过优化（如缓存、预分词、批量处理）？"**  
   （关注工程实践，展示对推理全链路的理解）

## 自测题

### 口述
- **能流畅讲清楚**：
  1. BPE、Unigram、WordPiece 三者的算法原理和差异（合并/裁剪/评分标准）
  2. 为什么 byte-level BPE 对多语言更鲁棒（字节级覆盖所有 Unicode）
  3. Embedding 层在微调时是否更新、为什么、什么场景需要更新
  4. 如何根据场景选择 tokenizer 算法和词表大小
  5. Unigram 的 EM 算法具体流程（E步/M步/裁剪）

### 手写
- **5 分钟能写出**：

  1. **BPE 合并算法核心逻辑**（伪代码）
  ```python
  def bpe_train(corpus, vocab_size):
      """BPE 训练核心：频次驱动贪心合并"""
      # 初始化：字符级词表（或 byte-level：256个字节）
      vocab = set(''.join(corpus))
      
      # 预分词：按空格切词，词内拆成字符
      splits = [[c for c in word] for word in corpus.split()]
      
      while len(vocab) < vocab_size:
          # 统计相邻字符对频次
          pairs = defaultdict(int)
          for word in splits:
              for i in range(len(word) - 1):
                  pairs[(word[i], word[i+1])] += 1
          
          # 合并最高频字符对
          if not pairs:
              break
          best_pair = max(pairs, key=pairs.get)
          new_token = ''.join(best_pair)
          vocab.add(new_token)
          
          # 更新所有词的分词结果
          splits = [merge_pair(word, best_pair) for word in splits]
      
      return vocab
  
  def merge_pair(word, pair):
      """在词中合并指定字符对"""
      merged = []
      i = 0
      while i < len(word):
          if i < len(word) - 1 and (word[i], word[i+1]) == pair:
              merged.append(''.join(pair))
              i += 2
          else:
              merged.append(word[i])
              i += 1
      return merged
  ```

  2. **Unigram 概率计算与裁剪**（核心逻辑）
  ```python
  def unigram_train(corpus, vocab_size, initial_vocab):
      """Unigram 训练核心：概率驱动迭代裁剪"""
      # 初始化：大候选词表 + 均匀概率
      vocab = initial_vocab  # 如所有 n-gram
      probs = {token: 1/len(vocab) for token in vocab}
      
      while len(vocab) > vocab_size:
          # E步：用 Viterbi 找最优切分
          segmentations = []
          for sentence in corpus:
              seg = viterbi_segment(sentence, vocab, probs)
              segmentations.append(seg)
          
          # M步：更新概率
          counts = defaultdict(float)
          for seg in segmentations:
              for token in seg:
                  counts[token] += 1
          total = sum(counts.values())
          probs = {t: c/total for t, c in counts.items()}
          
          # 裁剪：删除对似然贡献最小的 token
          losses = compute_removal_loss(vocab, corpus, probs)
          to_remove = sorted(losses.items(), key=lambda x: x[1])[
              :int(len(vocab) * 0.2)  # 每轮删 20%
          ]
          for token, _ in to_remove:
              vocab.remove(token)
              del probs[token]
      
      return vocab, probs
  
  def viterbi_segment(sentence, vocab, probs):
      """Viterbi 算法找最优切分"""
      n = len(sentence)
      dp = [0] + [-float('inf')] * n  # dp[i] = 前i个字符的最大log概率
      backtrack = [None] * (n + 1)
      
      for i in range(1, n + 1):
          for j in range(i):
              token = sentence[j:i]
              if token in vocab:
                  score = dp[j] + math.log(probs[token])
                  if score > dp[i]:
                      dp[i] = score
                      backtrack[i] = j
      
      # 回溯得到切分
      tokens = []
      i = n
      while i > 0:
          j = backtrack[i]
          tokens.append(sentence[j:i])
          i = j
      return tokens[::-1]
  ```

  3. **WordPiece 评分计算**（核心公式）
  ```python
  def wordpiece_score(pairs_count, token_count):
      """WordPiece 合并评分：LM 似然增益"""
      # score(x,y) = count(xy) / (count(x) * count(y))
      # 等价于：合并后相比独立出现的概率提升
      best_score = 0
      best_pair = None
      
      for (x, y), xy_count in pairs_count.items():
          x_count = token_count[x]
          y_count = token_count[y]
          score = xy_count / (x_count * y_count)
          
          if score > best_score:
              best_score = score
              best_pair = (x, y)
      
      return best_pair, best_score
  
  # 合并后 token 标记
  # "playing" -> ["play", "##ing"]
  # 其中 "##" 表示该子词非词首
  ```

  4. **Embedding 层前向传播**（PyTorch）
  ```python
  def forward(self, input_ids):
      # input_ids: [batch, seq_len]
      # embedding: [vocab_size, hidden_dim]
      return self.embedding(input_ids)  # [batch, seq_len, hidden_dim]
  ```

  5. **检查 tokenizer 覆盖率**
  ```python
  def check_coverage(tokenizer, corpus):
      """检查 tokenizer 对语料的覆盖情况"""
      total_tokens = 0
      unk_tokens = 0
      
      for text in corpus:
          tokens = tokenizer.encode(text)
          total_tokens += len(tokens)
          unk_tokens += tokens.count(tokenizer.unk_token_id)
      
      coverage = 1 - unk_tokens / total_tokens
      return {
          "coverage": coverage,      # 应 > 99.9%
          "unk_rate": unk_tokens / total_tokens,  # 应 < 0.1%
          "total_tokens": total_tokens
      }
  ```

  6. **Tokenizer 压缩率计算**
  ```python
  def compute_compression_ratio(tokenizer, corpus):
      """计算 tokenizer 的压缩率（字符数 / token数）"""
      total_chars = sum(len(text) for text in corpus)
      total_tokens = sum(
          len(tokenizer.encode(text)) for text in corpus
      )
      
      ratio = total_chars / total_tokens
      # 中文理想值 > 1.5
      # 英文理想值 > 3.0
      return ratio
  ```

## 标签
#Transformer #架构 #Tokenizer #Embedding #BPE #SentencePiece #多语言 #字节 #阿里

## 相关文档
- [[01-Transformer总览]] - Transformer 整体架构与 forward 流程
- [[02-Attention机制]] - Attention 层如何接收 embedding 输入
- [[03-FFN与归一化]] - FFN 层的结构与参数配置
- [[05-手撕MHA]] - 手写多头注意力实现
- [[../02-训练数据流水线/01-数据处理全链路]] - Tokenization 在数据预处理中的应用
- [[../06-模型架构对比/01-Qwen-LLaMA-DeepSeek对比]] - 不同模型的 tokenizer 差异
