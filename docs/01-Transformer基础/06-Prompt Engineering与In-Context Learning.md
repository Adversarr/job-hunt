# Prompt Engineering 与 In-Context Learning

## 一句话结论
Prompt Engineering 通过结构化设计（角色、任务、示例、格式）激活模型能力，In-Context Learning 利用少量示例引导模型推理，二者核心是降低任务难度、对齐训练分布，而非改变模型参数。

## 核心定义/公式

### Prompt Engineering 核心框架
```
Effective Prompt = Role + Task + Context + Examples + Format + Constraints

Role: 你是一个专业的[角色]，擅长[能力]
Task: 请完成[具体任务]，要求[明确标准]
Context: 背景：[必要信息]，输入：[具体输入]
Examples: 示例1：输入→输出，示例2：输入→输出
Format: 输出格式：[格式要求]
Constraints: 约束：[限制条件]
```

### In-Context Learning 定义
```python
# In-Context Learning 公式化表达
# 输入: prompt = instruction + k examples + query
# 输出: y ~ P(y | prompt; θ)

# Few-Shot Prompt 示例
prompt = """
任务：情感分类

示例1：这个产品很好用 → 正面
示例2：服务态度太差了 → 负面
示例3：物流速度快，质量不错 → 正面

输入：这个商品性价比很高
输出：
"""
# 模型输出: 正面（无需参数更新）
```

### Zero-Shot / Few-Shot / Fine-Tuning 对比

| 方法 | 参数更新 | 示例数量 | 推理成本 | 适用场景 |
|------|---------|---------|---------|----------|
| Zero-Shot | 无 | 0 | 低（仅 query） | 通用任务、快速验证 |
| Few-Shot (ICL) | 无 | 1-100 | 中（query + examples） | 任务对齐、格式约束 |
| Fine-Tuning | 有 | 1000+ | 高（训练成本） | 领域迁移、性能优化 |

### Prompt 长度对性能的影响公式
```
有效信息密度 = 任务相关 tokens / 总 tokens

建议比例：
- System Prompt + Task: 10-20%
- Examples: 20-40%
- Query + Context: 40-70%

避免信息稀释：
总长度 < 上下文窗口的 70%（预留输出空间）
```

## 为什么（3 个因果链）

### 1. 为什么 Few-Shot 示例能有效引导模型？

**现象**：提供 3-5 个示例后，模型输出质量显著提升，格式更规范，任务理解更准确。

**根因**：
- **分布对齐**：Few-Shot 示例来自目标任务分布，帮助模型从预训练分布迁移到目标任务分布（贝叶斯视角：P(y|x, examples) 比 P(y|x) 更准确）。
- **模式识别**：模型在预训练时学习过"模式复制"能力（如代码补全、格式推理），Few-Shot 示例激活这种能力。
- **隐式推理**：示例提供了推理路径（输入→输出的逻辑），模型通过注意力机制学习这种映射，而非简单记忆。

**结果**：Few-Shot 在格式对齐、任务理解上比 Zero-Shot 强，但示例选择和顺序会影响效果。

### 2. 为什么 Prompt 长度增加但效果反而下降？

**现象**：Prompt 从 200 tokens 增加到 2000 tokens，模型输出质量下降，出现遗忘、偏题、格式混乱。

**根因**：
- **注意力稀释**：模型注意力机制在长序列中分散，关键信息（任务、格式）被噪声淹没（Attention(Q, K) 中 K 包含大量无关 token）。
- **信息噪声比下降**：有效信息（任务描述、关键约束）占比降低，模型难以区分信号和噪声。
- **中间位置遗忘**：Transformer 对序列中间位置信息敏感度较低（"Lost in the Middle" 现象），长 Prompt 中关键信息易被忽略。

**结果**：Prompt 长度应遵循"足够但不冗余"原则，关键信息放在开头或结尾（注意力高峰区）。

### 3. 为什么 Few-Shot 示例顺序会影响结果？

**现象**：同样的示例，顺序 A 得到 85% 准确率，顺序 B 只有 70%。

**根因**：
- **近因效应**：模型对接近 query 的示例注意力更强（最后一两个示例影响最大）。
- **训练分布偏差**：模型在预训练时见过某些顺序模式（如"定义→示例→任务"），类似顺序更符合模型预期。
- **推理路径激活**：不同顺序激活不同的推理路径，某些顺序更容易引导正确推理（如"简单示例→复杂示例"逐步引导）。

**结果**：示例顺序是重要的超参数，建议将高质量示例放在靠后位置，简单到复杂排序。

## 怎么做（可落地步骤）

### 标准做法：设计有效 Prompt 的 6 步法

#### 步骤 1：明确任务目标
```python
# 任务定义模板
task_definition = """
角色：你是一个专业的{role}，擅长{capability}

任务：请{action}，要求{standard}

输出格式：
{format_template}
"""

# 示例：文本摘要任务
prompt = """
角色：你是一个专业的文本摘要专家，擅长提取关键信息、保持简洁准确

任务：请对以下新闻进行摘要，要求：
1. 不超过100字
2. 包含5W要素（谁、什么、何时、何地、为何）
3. 保持客观中立

输出格式：
- 主要事件：[一句话描述]
- 关键信息：[3-5个要点]
- 影响分析：[简要分析]
"""
```

#### 步骤 2：提供高质量示例
```python
# Few-Shot 示例选择原则
example_selection_criteria = {
    "多样性": "覆盖不同类型、难度、边界情况",
    "代表性": "示例分布与真实数据分布一致",
    "质量": "示例正确无误、格式规范、语言简洁",
    "数量": "3-5个（太多会稀释注意力，太少无法建立模式）"
}

# 示例排序策略
example_ordering = [
    "简单→复杂：逐步增加难度，引导模型理解",
    "通用→特殊：先通用规则，后特殊情况",
    "高质量示例靠后：近因效应，最后示例影响最大"
]
```

#### 步骤 3：设计清晰的格式约束
```python
# 格式约束示例
format_constraints = """
输出要求：
1. 格式：JSON 格式，包含以下字段：
   - "summary": 摘要内容（字符串）
   - "keywords": 关键词列表（数组，3-5个）
   - "sentiment": 情感倾向（"positive"/"negative"/"neutral"）

2. 约束：
   - 摘要长度：50-100字
   - 关键词：必须出现在原文中
   - 只输出 JSON，不要额外解释

示例：
输入：[原文]
输出：
{
  "summary": "...",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "sentiment": "positive"
}
"""
```

#### 步骤 4：处理长 Prompt 的策略
```python
# 长 Prompt 优化策略
long_prompt_strategies = {
    "关键信息前置": """
        将任务、格式、约束放在 Prompt 开头
        （模型对开头注意力最强）
    """,
    
    "分层结构": """
        System Prompt（角色、任务）
        ↓
        Examples（示例）
        ↓
        Context（背景信息）
        ↓
        Query（具体输入）
        ↓
        Format Reminder（格式提醒）
    """,
    
    "精简冗余": """
        1. 删除重复信息
        2. 合并相似指令
        3. 用列表替代冗长描述
        4. 移除模型已知常识
    """,
    
    "动态示例选择": """
        根据输入内容，动态选择最相关的 3-5 个示例
        （基于 embedding 相似度或关键词匹配）
    """
}
```

#### 步骤 5：迭代优化与评估
```python
# Prompt 评估流程
def evaluate_prompt(prompt, test_cases):
    """
    评估 Prompt 质量的指标
    """
    metrics = {
        "准确率": "输出是否符合预期（分类/生成任务）",
        "格式正确率": "输出格式是否规范（JSON/列表等）",
        "一致性": "多次调用结果是否稳定",
        "效率": "平均响应时间和 token 数"
    }
    
    # 迭代优化流程
    optimization_cycle = [
        "1. 在小样本（10-20个）上测试",
        "2. 分析错误模式（格式错误/内容错误/遗漏信息）",
        "3. 针对性修改 Prompt（增加约束/补充示例/调整顺序）",
        "4. A/B 测试对比效果",
        "5. 在更大数据集（100+）上验证"
    ]
    
    return metrics

# 常见错误模式与修正
error_patterns = {
    "格式错误": {
        "原因": "格式描述不够清晰/示例不一致",
        "修正": "增加格式示例，强化格式约束"
    },
    "内容偏题": {
        "原因": "任务描述模糊/示例误导",
        "修正": "明确任务边界，增加负面示例"
    },
    "信息遗漏": {
        "原因": "约束太多/示例不全面",
        "修正": "精简约束，增加覆盖缺失场景的示例"
    },
    "输出冗长": {
        "原因": "未限制长度/示例过长",
        "修正": "明确长度约束，提供简洁示例"
    }
}
```

#### 步骤 6：Few-Shot 示例的工程实践
```python
import random
from typing import List, Dict

class FewShotPromptBuilder:
    """Few-Shot Prompt 构建器"""
    
    def __init__(self, instruction: str, examples: List[Dict]):
        self.instruction = instruction
        self.examples = examples
        self.max_examples = 5  # 建议 3-5 个
    
    def select_examples(self, query: str, strategy: str = "similarity"):
        """
        示例选择策略
        strategy: "random" | "similarity" | "curriculum"
        """
        if strategy == "random":
            return random.sample(self.examples, min(self.max_examples, len(self.examples)))
        
        elif strategy == "similarity":
            # 基于输入相似度选择（需实现 embedding）
            query_embedding = self.get_embedding(query)
            scored_examples = [
                (ex, self.cosine_similarity(query_embedding, self.get_embedding(ex["input"])))
                for ex in self.examples
            ]
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            return [ex for ex, _ in scored_examples[:self.max_examples]]
        
        elif strategy == "curriculum":
            # 从简单到复杂排序
            scored_examples = [(ex, self.estimate_difficulty(ex)) for ex in self.examples]
            scored_examples.sort(key=lambda x: x[1])  # 简单→复杂
            return [ex for ex, _ in scored_examples[:self.max_examples]]
    
    def build_prompt(self, query: str, examples: List[Dict], format_example: str = ""):
        """
        构建 Few-Shot Prompt
        结构：Instruction + Examples + Query + Format Reminder
        """
        prompt_parts = [self.instruction]
        
        # 添加示例（高质量示例靠后）
        for i, ex in enumerate(examples):
            prompt_parts.append(f"\n示例{i+1}：\n输入：{ex['input']}\n输出：{ex['output']}")
        
        # 添加当前查询
        prompt_parts.append(f"\n输入：{query}\n输出：")
        
        # 可选：格式提醒
        if format_example:
            prompt_parts.append(format_example)
        
        return "\n".join(prompt_parts)
    
    def estimate_difficulty(self, example: Dict) -> float:
        """
        估计示例难度（简单启发式）
        可基于：输出长度、任务复杂度、领域专业性等
        """
        output_len = len(example["output"])
        return output_len  # 简单：输出越短难度越低
    
    @staticmethod
    def get_embedding(text: str):
        """获取文本 embedding（需接入模型）"""
        # 实际实现：调用 OpenAI API 或本地模型
        pass
    
    @staticmethod
    def cosine_similarity(emb1, emb2):
        """计算余弦相似度"""
        import numpy as np
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# 使用示例
builder = FewShotPromptBuilder(
    instruction="任务：情感分类，判断文本情感倾向（正面/负面/中性）",
    examples=[
        {"input": "这个产品很好用", "output": "正面"},
        {"input": "服务态度太差了", "output": "负面"},
        {"input": "物流速度快，质量不错", "output": "正面"},
        {"input": "还行，没什么特别的", "output": "中性"},
        {"input": "价格太贵了，不划算", "output": "负面"}
    ]
)

# 动态选择示例
query = "这个商品性价比很高"
selected_examples = builder.select_examples(query, strategy="curriculum")
prompt = builder.build_prompt(query, selected_examples)
```

### 关键配置/参数

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| Few-Shot 示例数量 | 3-5 个 | 太少模式不明显，太多注意力稀释 |
| 示例排序 | 简单→复杂 | 逐步引导，近因效应利用高质量示例 |
| Prompt 总长度 | < 上下文窗口的 70% | 预留输出空间，避免截断 |
| 关键信息位置 | 开头或结尾 | 注意力高峰区，中间易遗忘 |
| 示例多样性 | 覆盖边界情况 | 避免模型过拟合单一模式 |
| 格式约束 | 明确 + 示例 | 文字描述 + 实际示例，双重保障 |

### 代码示例：Prompt 效果排查流程
```python
def diagnose_prompt(prompt: str, model_output: str, expected_output: str):
    """
    Prompt 效果排查清单
    """
    issues = []
    
    # 1. 检查 Prompt 长度
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > 2000:
        issues.append({
            "type": "长度过长",
            "detail": f"Prompt {prompt_tokens} tokens，建议精简",
            "action": "删除冗余信息，合并相似指令"
        })
    
    # 2. 检查关键信息位置
    key_info_keywords = ["任务", "要求", "格式", "输出"]
    key_info_positions = [i for i, kw in enumerate(key_info_keywords) if kw in prompt]
    if key_info_positions and max(key_info_positions) < len(prompt.split('\n')) // 3:
        issues.append({
            "type": "关键信息位置不佳",
            "detail": "任务/格式描述在 Prompt 前部，可能被遗忘",
            "action": "将关键信息移至开头或结尾"
        })
    
    # 3. 检查示例一致性
    examples = extract_examples(prompt)
    if len(examples) > 0:
        formats = [analyze_format(ex["output"]) for ex in examples]
        if len(set(formats)) > 1:
            issues.append({
                "type": "示例格式不一致",
                "detail": f"发现 {len(set(formats))} 种不同格式",
                "action": "统一示例输出格式"
            })
    
    # 4. 检查输出与预期差异
    output_format = analyze_format(model_output)
    expected_format = analyze_format(expected_output)
    if output_format != expected_format:
        issues.append({
            "type": "输出格式错误",
            "detail": f"输出格式 {output_format} 与预期 {expected_format} 不符",
            "action": "增加格式示例，强化格式约束"
        })
    
    # 5. 检查任务描述清晰度
    if not any(kw in prompt for kw in ["请", "任务", "要求"]):
        issues.append({
            "type": "任务描述模糊",
            "detail": "未找到明确的任务指令",
            "action": "添加明确的任务描述，如'请完成XX任务'"
        })
    
    return issues

def count_tokens(text: str) -> int:
    """估算 token 数量（简化版，实际需用 tokenizer）"""
    # 中文约 1.5 字/token，英文约 0.75 词/token
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len(text.split()) - chinese_chars // 2
    return int(chinese_chars / 1.5 + english_words / 0.75)

def extract_examples(prompt: str) -> List[Dict]:
    """从 Prompt 中提取示例"""
    # 实际实现：解析 Prompt 结构
    return []

def analyze_format(text: str) -> str:
    """分析文本格式"""
    if text.startswith('{') and text.endswith('}'):
        return "JSON"
    elif text.startswith('[') and text.endswith(']'):
        return "列表"
    elif '\n' in text:
        return "多行文本"
    else:
        return "单行文本"

# 使用示例
prompt = """
你是一个文本分类专家。

任务：判断文本情感。

示例1：很好用 → 正面
示例2：太差了 → 负面

输入：这个商品还行
输出：
"""

model_output = "这个商品还可以，属于中性评价"
expected_output = "中性"

issues = diagnose_prompt(prompt, model_output, expected_output)
for issue in issues:
    print(f"[{issue['type']}] {issue['detail']}\n建议：{issue['action']}\n")
```

## 权衡分析

### Zero-Shot vs Few-Shot vs Fine-Tuning

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| Zero-Shot | 无需标注数据，推理快，灵活性高 | 依赖模型通用能力，格式控制弱 | 通用任务、快速验证、API 调用 |
| Few-Shot (ICL) | 任务对齐好，格式控制强，无需训练 | 推理成本高（示例占用 tokens），示例选择影响大 | 任务对齐、格式约束、少样本场景 |
| Fine-Tuning | 领域迁移强，推理成本不变 | 需要大量标注数据，训练成本高，灵活性差 | 领域迁移、性能优化、固定任务 |

### Few-Shot 示例数量权衡

| 示例数量 | 收益 | 代价 | 建议场景 |
|---------|------|------|----------|
| 1-2 个 | 推理快，token 消耗少 | 模式建立不充分，易过拟合 | 任务简单、格式清晰 |
| 3-5 个 | 平衡效果和成本，模式清晰 | 可能包含噪声示例 | 大多数任务（推荐） |
| 6-10 个 | 覆盖更多情况 | 注意力稀释，推理慢，边际收益递减 | 任务复杂、边界情况多 |
| 10+ 个 | 理论上效果最好 | 显著增加推理成本，实际效果不一定更好 | 不推荐（考虑 Fine-Tuning） |

### Prompt 长度权衡

| 长度范围 | 收益 | 代价 | 建议 |
|---------|------|------|------|
| < 500 tokens | 推理快，注意力集中 | 可能信息不足 | 简单任务、清晰任务 |
| 500-2000 tokens | 信息充分，平衡成本 | 中间信息易遗忘 | 大多数任务（推荐） |
| 2000-4000 tokens | 信息丰富，约束详细 | 推理慢，注意力稀释，成本高 | 复杂任务、多约束任务 |
| > 4000 tokens | 理论覆盖全面 | 效果下降，成本激增，易超上下文 | 不推荐（考虑分阶段或 Fine-Tuning） |

## 高频追问（7 个）

### 1. Q: In-Context Learning 和 Fine-Tuning 有什么本质区别？

**A**: 本质区别在于**是否更新模型参数**：
- **In-Context Learning**：模型参数固定，通过 Prompt 中的示例激活模型已有的推理能力，是"检索+推理"过程（类似从预训练知识库中检索相关模式）。
- **Fine-Tuning**：更新模型参数，模型学习新知识或新任务，是"学习+存储"过程（类似将新知识写入模型权重）。

从贝叶斯视角：
- ICL: P(y|x, examples, θ) ≈ P(y|x, θ) with context shift
- Fine-Tuning: θ' = θ + Δθ where Δθ from training

### 2. Q: Few-Shot 示例选择有哪些策略？如何选择最优示例？

**A**: 示例选择策略：
1. **相似度选择**：基于输入相似度（embedding cosine similarity）选择与 query 最相关的示例。
2. **多样性选择**：覆盖不同类型、难度、边界情况，避免模型过拟合单一模式。
3. **课程学习**：从简单到复杂排序，逐步引导模型理解任务。
4. **聚类采样**：对示例聚类，从每个簇中采样，保证代表性。

最优示例的特征：
- **代表性**：示例分布与真实数据分布一致
- **正确性**：示例输出准确无误
- **清晰性**：格式规范，无歧义
- **多样性**：覆盖不同场景和边界情况

### 3. Q: 为什么模型会"遗忘"长 Prompt 中间位置的信息？

**A**: "Lost in the Middle" 现象的原因：
1. **注意力机制**：Transformer 的注意力在长序列中分布不均，开头和结尾的 token 获得 higher attention score（开头是第一个 token，所有后续 token 都会关注；结尾靠近 query，注意力强）。
2. **位置编码**：相对位置编码可能对中间位置敏感度较低（RoPE 等编码在长距离下性能下降）。
3. **信息压缩**：模型在中间层对长序列信息压缩，中间位置信息易被"覆盖"或"稀释"。

缓解策略：
- 将关键信息放在开头或结尾
- 使用分隔符（如 "---"、"###"）增强结构
- 重复关键约束（开头+结尾各一次）
- 长信息分阶段处理（多轮对话）

### 4. Q: 如果 Prompt 效果不好，排查流程是什么？

**A**: 排查流程（5 步法）：
1. **检查任务描述**：是否清晰、明确？是否包含必要的约束？
2. **检查示例质量**：示例是否正确？格式是否一致？是否覆盖关键场景？
3. **检查示例顺序**：尝试不同排序（简单→复杂，高质量靠后）
4. **检查 Prompt 长度**：是否过长导致注意力稀释？关键信息是否在中间？
5. **检查格式约束**：是否明确输出格式？是否提供格式示例？

快速诊断清单：
- [ ] 任务描述包含"请完成XX任务"
- [ ] 有 3-5 个高质量示例
- [ ] 示例格式一致
- [ ] 关键信息在开头或结尾
- [ ] Prompt 长度 < 上下文窗口的 70%
- [ ] 有明确的格式约束和示例

### 5. Q: In-Context Learning 的示例数量越多效果越好吗？为什么？

**A**: 不是，存在边际效应递减和负面影响：
1. **边际收益递减**：3-5 个示例通常效果最好，超过 5 个后收益不明显（论文：Liu et al., 2022 "What Makes Good In-Context Examples"）。
2. **注意力稀释**：更多示例占用更多 tokens，分散模型对 query 的注意力。
3. **噪声增加**：示例越多，包含噪声或错误示例的概率越大，可能误导模型。
4. **推理成本**：示例增加导致推理成本线性增长（输入 tokens 增加）。
5. **上下文限制**：示例过多可能挤占输出空间或超出上下文窗口。

最佳实践：3-5 个高质量示例 > 10+ 个示例

### 6. Q: 如何评估一个 Prompt 的好坏？有哪些量化指标？

**A**: Prompt 评估指标：
1. **准确率**：输出是否符合预期（分类任务）
   ```python
   accuracy = correct_outputs / total_outputs
   ```
2. **格式正确率**：输出格式是否符合规范（生成任务）
   ```python
   format_accuracy = correctly_formatted_outputs / total_outputs
   ```
3. **一致性**：多次调用结果是否稳定（温度=0 时）
   ```python
   consistency = mode_frequency / total_outputs  # 最常见输出的频率
   ```
4. **效率**：平均 token 数和响应时间
   ```python
   efficiency = {
       "avg_input_tokens": prompt_length,
       "avg_output_tokens": output_length,
       "avg_latency": response_time
   }
   ```
5. **鲁棒性**：输入轻微变化时输出是否稳定
   ```python
   robustness = similar_outputs_on_perturbed_inputs / total_tests
   ```

### 7. Q: Prompt Engineering 在不同模型上的效果一致吗？为什么？

**A**: 效果不一致，原因：
1. **训练数据差异**：不同模型在预训练时见过的 Prompt 格式不同（GPT 系列 vs LLaMA 系列对 Few-Shot 格式偏好不同）。
2. **指令微调差异**：经过指令微调的模型（如 GPT-4、Claude）对结构化 Prompt 理解更好，未微调模型需要更明确的格式。
3. **上下文长度**：不同模型上下文窗口不同（GPT-4-turbo: 128k vs LLaMA-2: 4k），长 Prompt 效果差异大。
4. **对齐程度**：对齐好的模型（RLHF）更容易理解模糊指令，未对齐模型需要更明确的示例。

最佳实践：
- 针对目标模型优化 Prompt 格式
- 在目标模型上 A/B 测试
- 遵循模型官方 Prompt 指南（如 GPT-4 的 system message 格式）

## 常见错误（5 个）

### 1. 错误：示例格式不一致

**错误 Prompt**：
```
任务：情感分类

示例1：这个很好 → 正面
示例2：
输入：服务太差
输出：负面

输入：还行
输出：
```

**问题**：示例格式不一致（有的单行，有的多行），模型困惑。

**正确做法**：
```
任务：情感分类

示例1：
输入：这个很好
输出：正面

示例2：
输入：服务太差
输出：负面

示例3：
输入：还行
输出：中性

输入：这个产品不错
输出：
```

### 2. 错误：Prompt 过于冗长，关键信息淹没

**错误 Prompt**：
```
你是一个专业的文本分析师，在过去的10年里，你一直从事文本分类工作，
积累了丰富的经验。你的主要工作是对各种类型的文本进行情感分析，
包括产品评论、社交媒体帖子、新闻报道等。你擅长识别文本中的情感倾向，
包括正面、负面和中性。在分析时，你需要考虑文本的语境、语气、用词等因素。
你的分析结果需要准确、客观、可靠。你需要注意不要被反讽或隐喻误导。
下面是一些示例：

示例1：（略）
示例2：（略）
示例3：（略）

现在请对以下文本进行情感分析。输入：这个商品还可以。
```

**问题**：关键信息"情感分析"被大量无关信息淹没，模型注意力分散。

**正确做法**：
```
任务：情感分析（正面/负面/中性）

要求：
1. 判断文本情感倾向
2. 考虑语境、语气、用词
3. 注意反讽和隐喻

示例1：输入：这个很好 → 输出：正面
示例2：输入：服务太差 → 输出：负面

输入：这个商品还可以
输出：
```

### 3. 错误：示例数量过多且质量参差不齐

**错误 Prompt**：
```
任务：文本分类

示例1：很好 → 正面
示例2：不错 → 正面
示例3：还可以 → 正面
示例4：一般 → 中性
示例5：还行 → 中性
示例6：挺好的 → 正面
示例7：太差了 → 负面
示例8：很糟糕 → 负面
示例9：不好不坏 → 中性
示例10：非常满意 → 正面

输入：这个产品性价比高
输出：
```

**问题**：
1. 示例太多（10个），注意力稀释
2. 示例质量参差不齐（"挺好的"、"非常满意"冗余）
3. 缺少负面示例的多样性

**正确做法**：
```
任务：文本分类（正面/负面/中性）

示例1：输入：这个很好用 → 输出：正面
示例2：输入：服务态度太差了 → 输出：负面
示例3：输入：还行，没什么特别的 → 输出：中性
示例4：输入：价格太贵，不划算 → 输出：负面

输入：这个产品性价比高
输出：
```

### 4. 错误：任务描述模糊，缺少明确约束

**错误 Prompt**：
```
请分析这段文本的情感。

示例1：很好 → 正面

输入：这个商品还行
```

**问题**：
1. 任务描述模糊（"分析情感"未明确输出格式）
2. 示例太少，模式不清
3. 缺少格式约束

**正确做法**：
```
任务：情感分析

要求：判断文本情感倾向，输出"正面"、"负面"或"中性"

示例1：输入：这个很好用 → 输出：正面
示例2：输入：服务太差了 → 输出：负面
示例3：输入：还行，没什么特别的 → 输出：中性

输入：这个商品还行
输出：
```

### 5. 错误：忽略模型能力边界，过度依赖 Prompt

**错误做法**：
```
期望通过 Prompt Engineering 让模型：
1. 输出训练数据中未见过的领域知识
2. 完成复杂多步推理（如数学证明、代码生成）
3. 记住大量事实信息（如"请列出2024年所有新闻"）

只优化 Prompt，不考虑模型本身能力
```

**问题**：Prompt Engineering 只能激活模型已有能力，无法让模型获得新能力。

**正确做法**：
1. **评估模型能力边界**：测试模型在目标任务上的基线表现
2. **选择合适策略**：
   - 任务对齐：用 Few-Shot
   - 领域迁移：用 Fine-Tuning
   - 知识注入：用 RAG
   - 复杂推理：用 Chain-of-Thought 或 Tool Use
3. **组合策略**：Prompt Engineering + Fine-Tuning + RAG

## 反问面试官的问题

### 1. 技术深度类
- "贵团队在 Prompt Engineering 上有没有自动化工具或流程？如何评估和迭代 Prompt？"
- "Few-Shot 示例选择是否有策略？是基于相似度、多样性还是课程学习？"
- "长 Prompt 场景下，如何平衡信息完整性和注意力效率？有没有实践过的优化策略？"

### 2. 业务场景类
- "在生产环境中，Prompt 是如何管理的？版本控制、A/B 测试怎么做？"
- "不同模型（GPT-4、Claude、开源模型）的 Prompt 是否需要分别优化？差异大吗？"
- "用户自定义 Prompt 的场景下，如何防止 Prompt 注入攻击？"

## 自测题

### 口述（能流畅讲清楚）
1. **Zero-Shot、Few-Shot、Fine-Tuning** 的区别和适用场景
2. **Few-Shot 示例选择**的原则和策略（多样性、代表性、排序）
3. **Prompt Engineering 的 6 个核心要素**（角色、任务、示例、格式、约束、背景）
4. **长 Prompt 效果下降**的原因和缓解策略
5. **In-Context Learning 的工作原理**（为什么示例能引导模型）

### 手写（5 分钟能写出）
1. **Few-Shot Prompt 构建器**：
```python
def build_few_shot_prompt(instruction, examples, query):
    """
    要求：
    1. 示例格式统一
    2. 关键信息前置
    3. 高质量示例靠后
    """
    pass
```

2. **Prompt 效果诊断函数**：
```python
def diagnose_prompt(prompt, model_output, expected_output):
    """
    要求：
    1. 检查 Prompt 长度
    2. 检查示例格式一致性
    3. 检查关键信息位置
    """
    pass
```

3. **示例选择策略**：
```python
def select_examples(query, example_pool, strategy="similarity", k=5):
    """
    要求：
    1. 实现相似度选择
    2. 实现多样性选择
    3. 返回 top-k 示例
    """
    pass
```

## 标签
#Transformer #PromptEngineering #InContextLearning #Few-Shot #Zero-Shot #Fine-Tuning #Attention #序列建模 #阿里

## 相关文档
- [[02-Attention机制详解]]：理解注意力机制在 In-Context Learning 中的作用
- [[01-Tokenizer与Embedding]]：Prompt 的 tokenization 和信息表示
- [[05-长上下文]]：长 Prompt 的处理和外推
- [[09-KV Cache核心]]：Few-Shot 示例的 KV cache 复用
- [[11-评估体系]]：Prompt 效果的评估指标和方法