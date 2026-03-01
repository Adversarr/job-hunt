# 任务：构建 LLM 算法面试知识库

## 输入文件
- `overview.md` - 知识库大纲（15 章主题域）
- `questions.md` - 真实面试问题收集（字节/阿里/腾讯/美团等）

## 输出目标
在 `docs/` 目录下创建具体的markdown 文档，覆盖所有面试考点。

## 文档结构规范

### 目录组织
```
docs/
├── 01-transformer-base/        # 第 1 章：Transformer 基础
│   ├── transformer-overview.md
│   ├── attention-mechanism.md
│   ├── ffn-and-norm.md
│   ├── tokenizer-embedding.md
│   └── handcode-mha.md
├── 02-training-pipeline/       # 第 2 章：训练流水线
│   ├── data-processing.md
│   ├── tokenize-packing.md
│   ├── training-loop.md
│   └── debug-loss-nan.md
├── 03-sft-lora/                # 第 3 章：SFT 与参数高效微调
├── 04-rlhf-alignment/          # 第 4 章：RLHF/对齐
├── 05-long-context/            # 第 5 章：长上下文
├── 06-model-architecture/      # 第 6 章：模型架构对比
├── 07-distributed-training/    # 第 7 章：分布式训练
├── 08-precision-quant/         # 第 8 章：数值精度与量化
├── 09-inference-infra/         # 第 9 章：推理与 Infra
├── 10-flash-attention/         # 第 10 章：FlashAttention
├── 11-evaluation/              # 第 11 章：评估体系
├── 12-hallucination-rag/       # 第 12 章：幻觉与 RAG
├── 13-handcode-algo/           # 第 13 章：手撕算法题
└── 14-project-interview/       # 第 14 章：项目拷打
```

### 单文件模板
每个文档必须包含以下结构：

```markdown
# {标题}

## 一句话结论
<15 秒面试回答版本>

## 核心定义/公式
<关键公式、代码、配置>

## 为什么（2-3 个因果链）
1. ...
2. ...
3. ...

## 怎么做（可落地步骤）
<步骤 + 伪代码/配置/指标>

## 权衡分析
| 方案 | 收益 | 代价 | 适用边界 |
|------|------|------|----------|
| ...  | ...  | ...  | ...      |

## 高频追问（至少 5 个）
1. Q: ...
   A: ...
2. ...

## 常见错误（至少 3 个）
1. ...
2. ...
3. ...

## 反问面试官的问题
1. ...
2. ...

## 自测题
- 口述：...
- 手写：...

## 标签
#标签 1 #标签 2 #题型

## 相关文档
- [[链接到关联文档]]
```

## 执行步骤
技术栈假设
- 后端：CUDA + PyTorch + NumPy
- 框架：Transformer/DeepSpeed/FlashAttention/vLLM
- 代码示例使用 Python + PyTorch 语法

内容要求
- **精确**：公式、代码、配置参数必须准确
- **详细**：覆盖"是什么、为什么、怎么做、best-practice"
- **关联**：添加文档间交叉引用
- **可检索**：标签体系强制执行



### 第 1 步：解析映射
1. 读取 `overview.md` 提取 15 章主题域
2. 读取 `questions.md` 提取所有面试问题
3. 建立"问题→章节"映射表

### 第 2 步：创建/更新文档
对每个主题域：
1. 创建对应目录 `docs/{序号}-{主题名}/`
2. 根据 overview 中的子主题，拆分独立文档
3. 每个文档填充模板各字段

### 第 3 步：进一步思考
多想一步：
1. 参数设置，why？
2. 实现的坑，how？
3. 低层逻辑，what？

## 验收标准
- [ ] 每个 overview 子主题都有对应文档
- [ ] 每个 questions.md 问题都能映射到具体文档
- [ ] 文档遵循统一模板
- [ ] 标签体系完整
- [ ] 文档间有交叉引用
