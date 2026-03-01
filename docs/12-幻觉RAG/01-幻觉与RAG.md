# 幻觉与 RAG

## 一句话结论
幻觉根因分为训练分布（数据噪声/缺失）、推理机制（next-token 自回归、曝光偏差）、对齐副作用（过度迎合）三层；RAG 通过检索外部知识库提供事实依据，能显著降低幻觉但无法"彻底解决"，因检索不准、知识库不全、模型仍可能忽略或误用检索内容。

## 核心定义/公式

### 幻觉（Hallucination）定义
**大语言模型生成与事实不符、无法验证或逻辑矛盾的内容**，典型表现为：
- **事实性幻觉**：生成错误的事实（如虚构历史事件、错误数据）
- **忠实性幻觉**：生成与输入/上下文矛盾的内容（如忽略 prompt 约束）
- **推理幻觉**：逻辑错误、错误推理链

### 幻觉的分类与根因
```
幻觉根因分层模型
├── 第一层：训练分布（数据层面）
│   ├── 数据噪声：训练语料包含错误信息
│   ├── 数据缺失：知识覆盖不全、时效性不足
│   └── 分布偏移：训练分布与真实使用分布不一致
│
├── 第二层：推理机制（模型层面）
│   ├── Next-token 自回归：局部最优导致全局幻觉
│   ├── 曝光偏差（Exposure Bias）：训练用 teacher forcing，推理用自回归
│   └── 概率采样：低概率 token 组合导致不可预测输出
│
└── 第三层：对齐副作用（优化层面）
    ├── 过度迎合：奖励模型偏好导致编造
    ├── SFT 模式记忆：模仿训练数据风格而非事实
    └── RLHF 奖励黑客：学会"讨好" RM 而非生成真实内容
```

### RAG（Retrieval-Augmented Generation）架构
```python
# RAG 核心流程
def rag_pipeline(query: str, knowledge_base: VectorStore):
    # 1. 检索（Retrieval）
    retrieved_docs = knowledge_base.retrieve(query, top_k=5)

    # 2. 重排（Reranking，可选）
    reranked_docs = rerank(query, retrieved_docs, top_n=3)

    # 3. 上下文构建
    context = "\n".join([doc.content for doc in reranked_docs])

    # 4. 生成（Generation）
    prompt = f"""基于以下参考资料回答问题：
参考资料：
{context}

问题：{query}
回答："""

    response = llm.generate(prompt)

    # 5. 引用与校验（可选）
    response_with_citations = add_citations(response, reranked_docs)
    fact_check_result = verify_facts(response, reranked_docs)

    return response_with_citations, fact_check_result
```

### 检索质量指标
```python
# 检索评估指标
def evaluate_retrieval(retrieved_docs, relevant_docs):
    """评估检索质量"""
    # Recall@K: 召回率
    recall_at_k = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)

    # Precision@K: 精确率
    precision_at_k = len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs)

    # MRR (Mean Reciprocal Rank): 第一个相关文档的排名倒数
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            mrr = 1.0 / rank
            break

    # NDCG: 考虑排序位置的指标
    ndcg = compute_ndcg(retrieved_docs, relevant_docs)

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "ndcg": ndcg
    }
```

## 为什么（2-3 个因果链）

### 1. 为什么大模型会产生幻觉？—— 三层根因分析（阿里）

**因果链 1：训练分布问题 → 知识缺陷**
```
训练语料包含噪声和缺失
  → 模型学习到错误或不完整的知识表示
  → 推理时调用错误知识或无法找到正确知识
  → 生成事实性幻觉
```

**具体表现**：
- **数据噪声**：维基百科错误信息、虚假新闻、标注错误
  - 案例：训练数据中包含"特朗普出生于 1947 年"（实际 1946 年）
  - 模型生成时复现错误信息
- **数据缺失**：最新事件、长尾知识、专业领域知识
  - 案例：模型训练截止日期为 2023 年 4 月，无法回答 2024 年事件
- **分布偏移**：训练数据与真实场景分布不一致
  - 案例：训练数据多为正式文本，用户询问口语化问题时模型"胡编"

**因果链 2：推理机制问题 → 生成缺陷**
```
Next-token 自回归生成
  → 每步只考虑当前最优，不考虑全局一致性
  → 早期错误 token 影响后续生成
  → 生成连贯但错误的推理链或事实
```

**具体表现**：
- **局部最优陷阱**：
  ```
  Prompt: "请介绍爱因斯坦的相对论"
  模型思维：
    Token 1: "相对论" (概率 0.8)
    Token 2: "是" (概率 0.9)
    Token 3: "20世纪" (概率 0.7) ← 错误：应为 1905 年提出
    Token 4: "最伟大的" (概率 0.85)
    ...
  模型生成：相对论是 20 世纪最伟大的物理学发现之一...（事实正确）
  但如果早期 token 错误：
    Token 1: "相对论"
    Token 2: "包括" (概率 0.6)
    Token 3: "量子力学" (概率 0.5) ← 错误分支
    后续 token 为保持连贯性，继续编造错误内容
  ```
- **曝光偏差（Exposure Bias）**：
  - 训练时：使用 teacher forcing，输入是真实上下文
  - 推理时：使用自回归生成，输入是模型自己生成的上下文
  - 分歧：模型从未见过自己生成的错误上下文，不知道如何纠正

**因果链 3：对齐副作用 → 编造倾向**
```
RLHF/SFT 对齐优化
  → 模型学会"回答用户问题"而不是"承认不知道"
  → 对模糊问题生成看似合理但无事实依据的回答
  → 生成忠实性幻觉
```

**具体表现**：
- **过度迎合**：用户问"为什么地球是平的"，模型生成支持地平说的论据
- **奖励黑客**：模型学会生成"权威性"回答风格，即使内容错误
- **SFT 模式记忆**：模型模仿训练数据中的回答模式，而非真正理解事实

### 2. 为什么 RAG 不能"彻底解决"幻觉问题？（阿里追问）

**因果链：RAG 的局限 → 幻觉残留**
```
RAG 检索 → 召回不准 → 相关性低的知识干扰生成 → 幻觉
       → 知识库不全 → 缺失关键事实 → 模型编造
       → 模型忽略检索内容 → 仍依赖内部知识 → 幻觉
       → 检索内容冲突 → 模型选择困难 → 错误融合
```

**具体局限**：

**局限 1：检索质量问题**
```python
# 案例：检索不准导致的幻觉
Query: "2024 年诺贝尔物理学奖得主是谁"

# 检索结果（召回不准）
Retrieved:
- "2023 年诺贝尔物理学奖授予..." (不相关)
- "诺贝尔物理学奖历年得主名单" (不包含 2024)
- "2024 年物理学前沿研究" (无关)

# 模型行为
# 方案 A：基于检索内容生成
# → "2024 年诺贝尔物理学奖尚未公布..." (错误，可能已公布)

# 方案 B：忽略检索内容，依赖内部知识
# → 模型编造或使用过时知识
```

**局限 2：知识库覆盖不全**
- **时效性**：知识库未更新到最新信息
- **长尾知识**：专业领域、罕见事件知识缺失
- **数据质量**：知识库本身包含错误信息

**局限 3：模型无法有效利用检索内容**
```python
# 案例：模型忽略检索内容
Query: "爱因斯坦在哪一年提出相对论？"

Retrieved Context:
"爱因斯坦于 1905 年发表了狭义相对论..."

# 模型生成（忽略检索内容）
Response: "爱因斯坦在 1915 年提出了相对论"
# 原因：模型内部知识权重更高，或 prompt 设计不当
```

**局限 4：多文档冲突与整合困难**
```python
# 案例：检索到冲突信息
Query: "Python 3.12 发布时间"

Retrieved:
- Doc 1: "Python 3.12 于 2023 年 10 月发布"
- Doc 2: "Python 3.12 预计 2023 年底发布"

# 模型行为
# → 选择错误信息，或编造中间答案
```

### 3. 为什么不同缓解策略效果差异巨大？

**因果链：策略本质 → 适用边界**
```
策略选择不当
  → RAG 用于无需事实的任务（如创意写作）
  → 事实监督用于主观性问题
  → 工具调用用于模型已有能力范围
  → 效果提升有限甚至负面
```

**策略对比**：
- **RAG**：适合事实性问答、知识密集型任务
- **事实监督**：适合训练阶段注入知识，推理无额外成本
- **工具调用**：适合动态数据、实时查询（如天气、股票）
- **约束解码**：适合格式要求严格的任务（如 JSON、代码）
- **自一致性**：适合推理任务、多解问题

## 怎么做（可落地步骤）

### 标准做法

#### 1. RAG 系统搭建（完整流程）

**步骤 1：知识库构建**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1.1 文档加载与预处理
def load_and_preprocess_documents(file_paths):
    """加载文档并预处理"""
    documents = []
    for file_path in file_paths:
        # 加载文档
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 预处理：清洗、去重、去噪声
        content = clean_text(content)
        content = remove_duplicates(content)

        documents.append({
            'content': content,
            'metadata': {
                'source': file_path,
                'timestamp': get_file_timestamp(file_path)
            }
        })

    return documents

# 1.2 文档切分
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """文档切分为适合检索的块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc['content'])
        for i, split in enumerate(splits):
            chunks.append({
                'content': split,
                'metadata': {
                    **doc['metadata'],
                    'chunk_id': i
                }
            })

    return chunks

# 1.3 向量化与索引构建
def build_vector_store(chunks, embedding_model_name="BAAI/bge-large-zh"):
    """构建向量索引"""
    # 初始化 embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cuda'}
    )

    # 提取文本和元数据
    texts = [chunk['content'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    # 构建向量存储
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vector_store

# 完整流程
documents = load_and_preprocess_documents(["wiki/physics.txt", "wiki/history.txt"])
chunks = split_documents(documents)
vector_store = build_vector_store(chunks)
vector_store.save_local("vector_store/physics_history")
```

**步骤 2：检索与重排**
```python
from sentence_transformers import CrossEncoder

# 2.1 初始检索（向量检索）
def retrieve_documents(query, vector_store, top_k=10):
    """向量检索"""
    retrieved = vector_store.similarity_search_with_score(
        query,
        k=top_k
    )
    return retrieved

# 2.2 重排（Reranking）
def rerank_documents(query, documents, rerank_model="BAAI/bge-reranker-large", top_n=5):
    """使用 Cross-Encoder 重排"""
    # 初始化 reranker
    reranker = CrossEncoder(rerank_model, max_length=512)

    # 构建查询-文档对
    pairs = [(query, doc.page_content) for doc, score in documents]

    # 计算相关性分数
    scores = reranker.predict(pairs)

    # 按分数排序
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 返回 top-n
    return scored_docs[:top_n]

# 2.3 多路召回（可选）
def multi_way_retrieval(query, vector_stores, top_k_per_store=5):
    """多向量库召回 + 去重"""
    all_retrieved = []

    for store_name, store in vector_stores.items():
        retrieved = store.similarity_search_with_score(query, k=top_k_per_store)
        for doc, score in retrieved:
            doc.metadata['source_store'] = store_name
            all_retrieved.append((doc, score))

    # 去重（基于内容相似度）
    unique_retrieved = deduplicate_by_content(all_retrieved)

    return unique_retrieved
```

**步骤 3：上下文构建与生成**
```python
# 3.1 Prompt 设计
def build_rag_prompt(query, retrieved_docs, max_context_length=3000):
    """构建 RAG prompt"""
    # 构建上下文
    context_parts = []
    current_length = 0

    for doc, score in retrieved_docs:
        doc_text = f"【文档{len(context_parts)+1}】{doc.page_content}"

        if current_length + len(doc_text) > max_context_length:
            break

        context_parts.append(doc_text)
        current_length += len(doc_text)

    context = "\n\n".join(context_parts)

    # 构建 prompt
    prompt = f"""你是一个专业的知识助手。请基于以下参考资料回答问题。
如果参考资料中没有相关信息，请明确说明"参考资料中未找到相关信息"，不要编造答案。

参考资料：
{context}

问题：{query}

回答要求：
1. 回答必须基于参考资料，不能编造信息
2. 如果使用参考资料中的信息，请在回答中标注引用编号，如[文档1]
3. 如果参考资料中没有相关信息，请如实说明
4. 回答要简洁、准确、逻辑清晰

回答："""

    return prompt

# 3.2 生成与引用
def generate_with_citations(query, retrieved_docs, llm):
    """生成回答并添加引用"""
    prompt = build_rag_prompt(query, retrieved_docs)
    response = llm.generate(prompt, temperature=0.3)

    # 添加引用详情
    citations = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        citations.append({
            'id': i,
            'content': doc.page_content[:200] + "...",
            'source': doc.metadata.get('source', 'unknown'),
            'score': float(score)
        })

    return {
        'response': response,
        'citations': citations
    }
```

**步骤 4：事实校验与过滤**
```python
# 4.1 基于检索内容的校验
def verify_response_with_retrieval(response, retrieved_docs):
    """验证回答是否基于检索内容"""
    # 提取回答中的关键陈述
    claims = extract_claims(response)

    # 对每个陈述检查是否有检索支持
    verification_results = []
    for claim in claims:
        supporting_docs = []
        for doc, _ in retrieved_docs:
            if is_claim_supported_by_doc(claim, doc.page_content):
                supporting_docs.append(doc)

        verification_results.append({
            'claim': claim,
            'supported': len(supporting_docs) > 0,
            'supporting_docs': supporting_docs
        })

    return verification_results

# 4.2 NLI 模型校验（可选）
def verify_with_nli(response, retrieved_docs, nli_model):
    """使用 NLI 模型验证事实一致性"""
    results = []
    for doc, _ in retrieved_docs:
        # premise: 检索文档，hypothesis: 模型回答
        entailment_score = nli_model.predict(
            premise=doc.page_content,
            hypothesis=response
        )

        results.append({
            'doc': doc.page_content[:100],
            'entailment': entailment_score['entailment'],
            'contradiction': entailment_score['contradiction']
        })

    # 如果存在高矛盾分数，标记为可能幻觉
    has_contradiction = any(r['contradiction'] > 0.7 for r in results)

    return {
        'verification_results': results,
        'has_contradiction': has_contradiction
    }

# 4.3 完整的 RAG 流程
def complete_rag_pipeline(query, vector_store, llm):
    """完整的 RAG 流程：检索 → 重排 → 生成 → 校验"""

    # 1. 检索
    retrieved = retrieve_documents(query, vector_store, top_k=10)

    # 2. 重排
    reranked = rerank_documents(query, retrieved, top_n=5)

    # 3. 生成
    result = generate_with_citations(query, reranked, llm)

    # 4. 事实校验
    verification = verify_response_with_retrieval(
        result['response'],
        reranked
    )

    # 5. 添加置信度
    confidence = compute_confidence_score(result['response'], verification)

    result['verification'] = verification
    result['confidence'] = confidence

    return result
```

#### 2. 事实监督训练

**核心思想**：在 SFT 阶段引入事实监督信号，让模型学会区分事实与编造。

```python
# 事实监督数据构造
def construct_fact_supervision_data(questions, knowledge_base):
    """构造事实监督训练数据"""
    data = []

    for question in questions:
        # 检索相关知识
        retrieved = knowledge_base.retrieve(question, top_k=3)

        if retrieved:
            # 有检索支持：生成基于事实的回答
            context = "\n".join([doc.content for doc in retrieved])
            prompt = f"基于以下信息回答：\n{context}\n\n问题：{question}"
            response = generate_ground_truth_response(prompt)

            # 标注：事实型回答
            data.append({
                'prompt': question,
                'response': response,
                'type': 'fact',
                'has_context': True
            })
        else:
            # 无检索支持：生成"不知道"型回答
            response = "抱歉，我没有找到相关信息来回答这个问题。"

            # 标注：拒绝型回答
            data.append({
                'prompt': question,
                'response': response,
                'type': 'refusal',
                'has_context': False
            })

    return data

# 事实监督 Loss
def fact_supervision_loss(model, batch):
    """事实监督损失函数"""
    prompts = batch['prompt']
    responses = batch['response']
    types = batch['type']

    # 标准 cross-entropy loss
    logits = model(prompts, responses)
    ce_loss = F.cross_entropy(logits, labels)

    # 额外的监督信号
    # 1. 对于 fact 类型，奖励引用检索内容
    # 2. 对于 refusal 类型，奖励简洁的拒绝回答
    # 3. 惩罚无依据的编造

    if 'fact' in types:
        # 检查回答是否包含检索内容关键词
        retrieval_keywords = extract_keywords(batch['context'])
        response_keywords = extract_keywords(responses)
        overlap = len(set(retrieval_keywords) & set(response_keywords))

        # 奖励重叠
        fact_reward = overlap / len(retrieval_keywords)
        loss = ce_loss - 0.1 * fact_reward

    elif 'refusal' in types:
        # 检查回答是否简洁且包含拒绝词
        refusal_words = ['抱歉', '不知道', '没有相关信息']
        has_refusal = any(word in responses for word in refusal_words)

        if has_refusal and len(responses) < 50:
            # 奖励简洁的拒绝
            loss = ce_loss - 0.05
        else:
            # 惩罚冗长或无拒绝的回答
            loss = ce_loss + 0.1

    return loss
```

#### 3. 工具调用缓解幻觉

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent

# 定义工具
def search_tool(query: str) -> str:
    """搜索工具"""
    results = web_search(query)
    return format_search_results(results)

def calculator_tool(expression: str) -> str:
    """计算器工具"""
    return str(eval(expression))

def database_query_tool(query: str) -> str:
    """数据库查询工具"""
    results = database.execute(query)
    return format_query_results(results)

# 工具列表
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="搜索最新信息和事实"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="执行数学计算"
    ),
    Tool(
        name="Database",
        func=database_query_tool,
        description="查询结构化数据"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 使用
query = "2024 年中国 GDP 增长率是多少？"
response = agent.run(query)
# Agent 会自动调用 Search 工具获取最新数据
```

#### 4. 约束解码

```python
from transformers import Constraint, ConstraintGeneration

# 定义约束：强制模型承认"不知道"
class AdmitUnknownConstraint(Constraint):
    """约束模型在不确定时承认不知道"""

    def __init__(self, uncertainty_threshold=0.3):
        self.uncertainty_threshold = uncertainty_threshold

    def apply(self, generated_tokens, logits):
        """应用约束"""
        # 检测不确定性（基于 logits 分布）
        probs = F.softmax(logits[0], dim=-1)
        max_prob = probs.max().item()

        # 如果最大概率低于阈值，强制选择"我不知道"类回答
        if max_prob < self.uncertainty_threshold:
            # 提升"不知道"、"抱歉"等 token 的概率
            unknown_tokens = ["我", "不", "知道", "抱歉", "没有"]
            for token in unknown_tokens:
                token_id = tokenizer.encode(token)[0]
                logits[0, token_id] += 5.0  # 提升 logit

        return logits

# 使用约束解码
def constrained_generation(prompt, uncertainty_threshold=0.3):
    """约束解码生成"""
    constraint = AdmitUnknownConstraint(uncertainty_threshold)

    inputs = tokenizer(prompt, return_tensors='pt')

    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,
        constraints=[constraint],
        temperature=0.7
    )

    return tokenizer.decode(outputs[0])
```

#### 5. 自一致性（Self-Consistency）

```python
def self_consistency_generation(query, llm, n_samples=5, temperature=0.8):
    """自一致性生成：多次采样 + 投票"""
    responses = []

    # 多次采样
    for i in range(n_samples):
        response = llm.generate(
            query,
            temperature=temperature,
            top_p=0.9
        )
        responses.append(response)

    # 提取关键答案
    answers = [extract_key_answer(r) for r in responses]

    # 投票
    answer_counts = {}
    for answer in answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    # 选择最常见的答案
    final_answer = max(answer_counts, key=answer_counts.get)
    confidence = answer_counts[final_answer] / n_samples

    return {
        'final_answer': final_answer,
        'confidence': confidence,
        'all_responses': responses,
        'vote_distribution': answer_counts
    }

# 案例：数学推理
query = "一个篮子里有 15 个苹果，拿走了 3 个，又放进去了 5 个，现在有几个苹果？"

result = self_consistency_generation(query, llm, n_samples=10)

# 输出示例：
# {
#   'final_answer': '17',
#   'confidence': 0.8,
#   'vote_distribution': {'17': 8, '13': 2}
# }
```

### 关键配置/参数

#### RAG 系统关键参数
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **文档切分大小** | 300-500 tokens | 平衡检索粒度与上下文完整性 |
| **切分重叠** | 50-100 tokens | 避免跨块信息丢失 |
| **检索 top_k** | 5-10 | 初始召回数量 |
| **重排 top_n** | 3-5 | 最终使用的文档数量 |
| **Embedding 模型** | BGE-large-zh / text-embedding-ada-002 | 平衡质量与成本 |
| **Reranker 模型** | BGE-reranker-large | Cross-Encoder，提升排序精度 |
| **温度** | 0.3-0.5 | RAG 场景较低温度，减少幻觉 |

#### 幻觉检测阈值
| 指标 | 推荐阈值 | 说明 |
|------|----------|------|
| **NLI 矛盾分数** | > 0.7 | 高矛盾表示可能幻觉 |
| **检索支持度** | < 0.5 | 低支持度表示回答缺乏依据 |
| **置信度分数** | < 0.6 | 低置信度需要人工审核 |
| **自一致性投票率** | < 0.6 | 低一致性表示模型不确定 |

## 权衡分析

| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| **RAG** | 显著降低事实性幻觉，支持最新知识，可溯源 | 检索延迟（50-200ms），知识库维护成本，检索错误会引入新幻觉 | 事实性问答、知识密集型任务、需要溯源场景 |
| **事实监督** | 训练阶段注入知识，推理无额外成本 | 需要高质量标注数据，知识更新需重新训练 | 知识相对稳定、推理延迟敏感场景 |
| **工具调用** | 实时数据、精确计算、动态查询 | 依赖外部服务，工具调用失败率，增加推理延迟 | 实时数据查询（天气、股票）、数学计算、数据库查询 |
| **约束解码** | 强制格式、强制承认无知 | 可能过度约束，降低流畅度 | 结构化输出、高风险场景、法律/医疗领域 |
| **自一致性** | 提升推理准确性，提供置信度 | 推理成本增加 N 倍 | 数学推理、逻辑推理、需要置信度估计的场景 |
| **混合方案** | 综合各方案优势 | 实现复杂，调试困难 | 生产环境、高准确性要求场景 |

### RAG 与非 RAG 场景对比
| 任务类型 | 是否适合 RAG | 原因 |
|----------|--------------|------|
| 事实性问答（如"爱因斯坦哪年提出相对论"） | ✅ 适合 | 需要准确事实，RAG 提供可靠依据 |
| 创意写作（如"写一首诗"） | ❌ 不适合 | 不需要事实依据，RAG 反而限制创造性 |
| 推理任务（如数学题） | ⚠️ 可选 | RAG 提供例题有帮助，但核心靠模型推理能力 |
| 代码生成 | ⚠️ 可选 | RAG 提供文档/API 有帮助，但可能引入噪声 |
| 翻译 | ❌ 不适合 | 模型已掌握翻译能力，RAG 无意义 |
| 实时信息查询（如"今天天气"） | ❌ 不适合 | 应使用工具调用，RAG 知识库无法实时更新 |

## 高频追问（至少 5 个）

### 1. Q: 怎么缓解大模型的幻觉问题？（阿里通义）

**A**: 幻觉缓解需要**分层治理**：

**第一层：训练阶段**
- **数据清洗**：去噪、去重、事实校验训练数据
- **事实监督**：引入事实性标注，训练模型区分事实与编造
- **RLHF 对齐**：通过偏好学习，惩罚编造行为

**第二层：推理阶段**
- **RAG**：检索外部知识库，提供事实依据
- **工具调用**：对实时数据、精确计算使用工具
- **约束解码**：强制模型承认"不知道"
- **自一致性**：多次采样投票，提升稳定性

**第三层：后处理阶段**
- **事实校验**：使用 NLI 模型检测矛盾
- **引用验证**：检查回答是否基于检索内容
- **人工审核**：高风险场景人工介入

**优先级**：RAG > 工具调用 > 约束解码 > 事实监督 > 自一致性

### 2. Q: RAG 是否能彻底解决幻觉问题？（阿里追问）

**A**: **不能彻底解决**，原因如下：

**1. 检索质量问题**
- 召回不准：检索到的文档不相关或低质量
- 知识库不全：长尾知识、最新知识缺失
- 检索延迟：实时性要求高的场景无法满足

**2. 模型能力问题**
- 模型可能**忽略检索内容**，仍依赖内部错误知识
- 多文档冲突时，模型可能选择错误信息
- 复杂推理时，模型可能错误整合检索内容

**3. 幻觉类型问题**
- RAG 只能缓解**事实性幻觉**
- 无法缓解**推理幻觉**（逻辑错误）
- 无法缓解**忠实性幻觉**（忽略 prompt 约束）

**4. 实践证据**
- 研究表明：RAG 能降低 30-50% 幻觉，但无法归零
- 真实场景：检索错误 + 模型编造 = 更隐蔽的幻觉

**结论**：RAG 是"缓解"而非"根治"幻觉的工具，需结合其他策略。

### 3. Q: 大语言模型在推理时出现幻觉现象的原因是什么？有哪些缓解方法？（阿里）

**A**: 详见"为什么"部分的三个因果链分析。简述：

**根因**：
1. **训练分布**：数据噪声、知识缺失、分布偏移
2. **推理机制**：next-token 自回归、曝光偏差、概率采样
3. **对齐副作用**：过度迎合、奖励黑客、SFT 模式记忆

**缓解方法**：
1. **RAG**：检索外部知识，提供事实依据
2. **工具调用**：实时数据、精确计算
3. **约束解码**：强制承认无知
4. **事实监督**：训练阶段注入知识
5. **自一致性**：多次采样投票
6. **RLHF 对齐**：惩罚编造行为

### 4. Q: RAG 的检索质量如何评估？有哪些关键指标？

**A**: RAG 检索质量评估分为**离线评估**和**在线评估**：

**离线评估指标**：
```python
# 1. 召回指标
- Recall@K: 召回率（top-K 中相关文档占比）
- Precision@K: 精确率（top-K 中相关文档占比）
- MRR: 第一个相关文档的排名倒数
- NDCG: 考虑排序位置的归一化折损累计增益

# 2. 检索延迟
- 平均检索时间（ms）
- P99 延迟

# 3. 相关性判断
- 人工标注相关性（1-5 分）
- LLM 辅助相关性判断
```

**在线评估指标**：
```python
# 1. 端到端指标
- 回答准确率（人工/LLM 评估）
- 幻觉率（事实错误占比）
- 引用覆盖率（回答中有引用的比例）

# 2. 业务指标
- 用户满意度
- 拒答率（无检索支持的拒答比例）
- 点击率/采纳率
```

**关键阈值**：
- Recall@5 > 0.8：召回质量良好
- Precision@3 > 0.7：排序质量良好
- 检索延迟 < 200ms：用户体验良好

### 5. Q: 如何设计 RAG 的 Prompt 才能最大化降低幻觉？

**A**: RAG Prompt 设计的关键原则：

**原则 1：明确指示"基于检索内容回答"**
```
✅ 好 Prompt:
"基于以下参考资料回答问题。如果参考资料中没有相关信息，请明确说明'参考资料中未找到相关信息'，不要编造答案。"

❌ 差 Prompt:
"回答以下问题。"  # 模型可能忽略检索内容
```

**原则 2：要求引用来源**
```
"如果使用参考资料中的信息，请在回答中标注引用编号，如[文档1]。"
```

**原则 3：设置置信度门槛**
```
"只有当你有 80% 以上把握时才回答，否则请说明不确定。"
```

**原则 4：提供拒绝模板**
```
"如果无法基于参考资料回答，请使用以下格式：
'抱歉，参考资料中没有关于XXX的信息，无法准确回答该问题。'"
```

**原则 5：限制回答长度**
```
"回答要简洁（不超过 200 字），避免冗余信息。"
```

**完整 Prompt 模板**：
```
你是一个专业的知识助手。请基于以下参考资料回答问题。

【重要规则】
1. 回答必须基于参考资料，不能编造信息
2. 如果使用参考资料中的信息，请标注引用编号，如[文档1]
3. 如果参考资料中没有相关信息，请如实说明"参考资料中未找到相关信息"
4. 回答要简洁、准确、逻辑清晰，不超过 200 字

参考资料：
{context}

问题：{query}

回答：
```

### 6. Q: 如果检索到的多个文档存在冲突信息，模型应该如何处理？

**A**: 多文档冲突处理策略：

**策略 1：显式展示冲突（透明性优先）**
```python
# Prompt 示例
"检测到参考资料中存在不同说法：
- 文档1认为...
- 文档2认为...

综合分析后，我的理解是..."
```

**策略 2：优先级排序（准确性优先）**
```python
# 文档优先级规则
优先级：官方文档 > 新闻报道 > 用户评论
时效性：最新文档 > 旧文档
权威性：专家观点 > 普通观点

# Prompt 示例
"根据权威性和时效性，我优先采用文档1的信息..."
```

**策略 3：拒答（安全性优先）**
```python
# 检测冲突后拒答
if conflict_detected and confidence < 0.6:
    return "参考资料中存在冲突信息，无法确定准确答案，建议查阅更权威的来源。"
```

**策略 4：标注不确定性**
```python
# 明确标注不确定性
"根据部分参考资料（文档1、3），答案是... 但也有资料（文档2）持不同观点。建议进一步核实。"
```

**推荐做法**：
- **高风险场景**：拒答或显式展示冲突
- **一般场景**：优先级排序 + 标注来源
- **用户可自行判断场景**：显式展示冲突，由用户决策

### 7. Q: RAG 与微调（SFT/RLHF）的关系是什么？能否结合？

**A**: RAG 与微调是**互补关系**，而非替代关系：

**对比**：
| 维度 | RAG | 微调（SFT/RLHF） |
|------|-----|------------------|
| **知识来源** | 外部知识库 | 模型内部参数 |
| **知识更新** | 实时更新知识库 | 需重新训练 |
| **推理成本** | 额外检索延迟 | 无额外成本 |
| **知识容量** | 理论无限 | 受模型参数限制 |
| **适用场景** | 事实性知识、动态知识 | 推理能力、风格对齐、专业能力 |

**结合策略**：

**策略 1：微调提升 RAG 效果**
```python
# 微调目标：让模型更好地利用检索内容
训练数据：
- 包含检索上下文的对齐数据
- 标注"基于检索"vs"忽略检索"的偏好
- 奖励正确引用、惩罚编造

效果：
- 模型学会优先使用检索内容
- 提升引用准确性
- 降低忽略检索的概率
```

**策略 2：RAG 补充微调不足**
```python
# 微调后仍使用 RAG
场景：
- 微调注入领域知识，但无法覆盖所有知识
- RAG 补充最新知识、长尾知识

示例：
- 医疗模型：微调学习医学推理能力，RAG 提供最新药物信息
- 法律模型：微调学习法律推理，RAG 提供最新法条
```

**策略 3：混合知识注入**
```python
# 部分知识通过微调注入，部分通过 RAG 提供
微调注入：
- 领域核心知识（相对稳定）
- 推理模式、分析框架
- 专业术语、概念定义

RAG 提供：
- 最新动态知识
- 长尾知识
- 案例库、数据集
```

**最佳实践**：
- **通用模型**：RAG 为主，微调对齐风格
- **领域模型**：微调注入领域知识 + RAG 补充最新信息
- **高准确性场景**：微调 + RAG + 工具调用 + 约束解码

## 常见错误（至少 3 个）

### 1. 错误：RAG 检索不准，但强制模型使用检索内容

**现象**：检索到不相关文档，模型被迫基于错误信息回答，导致幻觉

**错误代码**：
```python
# ❌ 错误：直接使用检索结果，不检查相关性
retrieved = vector_store.retrieve(query, top_k=5)
context = "\n".join([doc.content for doc in retrieved])

prompt = f"基于以下信息回答：\n{context}\n\n问题：{query}"
response = llm.generate(prompt)
```

**正确做法**：
```python
# ✅ 正确：检索 + 重排 + 相关性过滤
retrieved = vector_store.retrieve(query, top_k=10)
reranked = rerank(query, retrieved, top_n=5)

# 过滤低相关性文档
relevant_docs = [doc for doc, score in reranked if score > 0.5]

if len(relevant_docs) == 0:
    # 无相关文档，拒答
    return "抱歉，没有找到相关信息。"

context = "\n".join([doc.content for doc in relevant_docs])
```

### 2. 错误：RAG Prompt 设计不当，模型忽略检索内容

**现象**：模型仍依赖内部知识，忽略检索到的内容

**错误 Prompt**：
```python
# ❌ 错误：指令不明确
prompt = f"""参考资料：
{context}

{query}
"""
# 问题：未明确要求"基于参考资料回答"，模型可能忽略
```

**正确 Prompt**：
```python
# ✅ 正确：明确指令 + 引用要求 + 拒答模板
prompt = f"""你是一个专业的知识助手。请基于以下参考资料回答问题。

【重要规则】
1. 回答必须基于参考资料，不能编造信息
2. 如果使用参考资料中的信息，请标注引用编号，如[文档1]
3. 如果参考资料中没有相关信息，请如实说明"参考资料中未找到相关信息"

参考资料：
{context}

问题：{query}

回答："""
```

### 3. 错误：幻觉检测过度依赖单一指标

**现象**：使用单一指标（如 NLI 矛盾分数）判断幻觉，误判率高

**错误做法**：
```python
# ❌ 错误：仅用 NLI 矛盾分数判断
nli_result = nli_model.predict(premise=context, hypothesis=response)
if nli_result['contradiction'] > 0.5:
    return "检测到幻觉"
```

**正确做法**：
```python
# ✅ 正确：多指标综合判断
def detect_hallucination(response, retrieved_docs):
    """多维度幻觉检测"""

    # 1. NLI 矛盾检测
    nli_score = nli_model.predict(retrieved_docs, response)

    # 2. 检索支持度
    support_score = compute_retrieval_support(response, retrieved_docs)

    # 3. 置信度估计
    confidence = compute_confidence(response)

    # 4. 事实校验（可选，调用外部 API）
    fact_check = external_fact_check(response)

    # 综合判断
    hallucination_risk = (
        0.3 * nli_score['contradiction'] +
        0.3 * (1 - support_score) +
        0.2 * (1 - confidence) +
        0.2 * fact_check['error_rate']
    )

    return hallucination_risk > 0.5
```

### 4. 错误：混淆幻觉类型，使用不当策略

**现象**：对推理幻觉使用 RAG，对事实性幻觉使用约束解码，效果差

**错误案例**：
```python
# ❌ 错误：对推理任务使用 RAG
query = "如果 A > B, B > C, 那么 A 和 C 的关系是什么？"
# 这是推理任务，RAG 无意义，应提升模型推理能力

# ❌ 错误：对事实性问答仅用约束解码
query = "爱因斯坦哪一年提出相对论？"
# 这是事实性问题，约束解码无法解决知识缺失，应使用 RAG
```

**正确做法**：
```python
# ✅ 正确：根据幻觉类型选择策略
def select_strategy(query):
    """根据问题类型选择策略"""

    # 事实性问答 → RAG
    if is_factual_question(query):
        return "RAG"

    # 推理任务 → 提升推理能力 / 自一致性
    elif is_reasoning_task(query):
        return "self_consistency"

    # 实时数据 → 工具调用
    elif is_realtime_query(query):
        return "tool_calling"

    # 创意写作 → 降低约束
    elif is_creative_task(query):
        return "no_constraint"

    # 高风险场景 → 多策略组合
    elif is_high_risk(query):
        return "RAG + constraint + fact_check"
```

### 5. 错误：RAG 知识库不更新，知识过时

**现象**：RAG 知识库长期不更新，模型提供过时信息

**错误做法**：
```python
# ❌ 错误：知识库从未更新
vector_store = FAISS.load_local("vector_store_2023_01")  # 2023 年 1 月的知识库
# 2024 年查询时，返回过时信息
```

**正确做法**：
```python
# ✅ 正确：定期更新知识库 + 标注时效性
def update_knowledge_base(vector_store, new_docs, update_frequency="daily"):
    """定期更新知识库"""

    # 1. 添加新文档
    for doc in new_docs:
        doc.metadata['timestamp'] = datetime.now()
        doc.metadata['source_date'] = extract_date(doc)

    vector_store.add_documents(new_docs)

    # 2. 清理过时文档
    outdated = get_outdated_docs(vector_store, threshold_days=365)
    vector_store.delete(outdated)

    # 3. 重新索引（可选）
    if len(new_docs) > 1000:
        vector_store = rebuild_index(vector_store)

    return vector_store

# 检索时优先返回最新文档
def retrieve_with_recency(query, vector_store, top_k=5):
    retrieved = vector_store.similarity_search_with_score(query, k=top_k * 2)

    # 按时效性 + 相关性排序
    scored = []
    for doc, score in retrieved:
        recency = compute_recency_score(doc.metadata['timestamp'])
        combined_score = 0.7 * score + 0.3 * recency
        scored.append((doc, combined_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

## 反问面试官的问题

### 1. 技术深度类
- "团队在实际业务中，幻觉率能控制在什么水平？主要瓶颈在哪里？"
- "RAG 的检索质量在你们场景中，Recall@5 一般能做到多少？有什么优化经验？"
- "你们有尝试过事实监督训练吗？数据是怎么构造的？效果如何？"
- "对于多文档冲突的场景，你们是怎么处理的？用户接受度如何？"

### 2. 业务场景类
- "业务中对幻觉的容忍度如何？是宁可拒答也不能错，还是允许一定错误率？"
- "RAG 的延迟开销在你们场景中是否可接受？有没有做过延迟优化？"
- "知识库的更新频率是怎么定的？全量更新还是增量更新？"
- "有没有遇到过 RAG 反而引入新幻觉的情况？怎么分析和解决的？"

## 自测题

### 口述（能流畅讲清楚的知识点）
1. 幻觉的三层根因（训练分布、推理机制、对齐副作用）及其因果链
2. RAG 为什么不能"彻底解决"幻觉的四个局限
3. RAG 完整流程（检索→重排→生成→校验）及每个环节的关键参数
4. 不同缓解策略（RAG/事实监督/工具调用/约束解码/自一致性）的适用边界
5. 如何设计 RAG Prompt 才能最大化降低幻觉

### 手写（5 分钟能写出的代码/公式）
1. **KV Cache 显存估算**（复习）
2. **RAG 检索质量评估函数**：
```python
def evaluate_retrieval(retrieved_docs, relevant_docs):
    """计算 Recall@K, Precision@K, MRR"""
    # 实现计算逻辑
    pass
```

3. **简单的 RAG 流程代码**：
```python
def simple_rag(query, vector_store, llm):
    """实现：检索 → 构建 prompt → 生成"""
    # 实现完整流程
    pass
```

4. **幻觉检测函数**：
```python
def detect_hallucination(response, retrieved_docs):
    """多维度幻觉检测"""
    # 实现：NLI + 检索支持度 + 置信度
    pass
```

## 标签
#幻觉 #RAG #推理 #检索增强 #事实监督 #工具调用 #约束解码 #自一致性 #阿里 #字节 #腾讯

## 相关文档
- [[02-系统能力]] - 单 LLM 的能力边界与需要的系统能力
- [[../04-RLHF对齐/01-RLHF总览]] - RLHF 对齐如何影响幻觉
- [[../03-SFT与LoRA/01-SFT目标与边界]] - SFT 的局限性与对齐需求
- [[../09-推理Infra/02-KV Cache核心]] - 推理优化与延迟权衡
