# 项目：LLM 算法面试知识库

## 概述
本项目用于构建 LLM 算法面试知识库，包含面试问答文档。

## 输入文件
- `overview.md` - 知识库大纲（14 章主题域）
- `questions.md` - 真实面试问题收集（字节/阿里/腾讯/美团等）

## 输出目录
`docs/` - 按 chapter 组织的 markdown 文档

## 文档模板
每个文档必须包含：
- 一句话结论
- 核心定义/公式
- 为什么（2-3 个因果链）
- 怎么做（可落地步骤）
- 权衡分析
- 高频追问（至少 5 个）
- 常见错误（至少 3 个）
- 反问面试官的问题
- 自测题
- 标签
- 相关文档

详细模板见 `.opencode/agents/write-document.md`

## 技术栈
- CUDA + PyTorch + NumPy
- Transformers / DeepSpeed / FlashAttention / vLLM
- 代码示例使用 Python + PyTorch

## 内容要求
- 精确：公式、代码、配置参数必须准确
- 详细：覆盖"是什么、为什么、怎么做、best-practice"
- 关联：添加文档间交叉引用
- 可检索：标签体系强制执行

## 验收标准
- 每个 overview 子主题都有对应文档
- 每个 questions.md 问题都能映射到具体文档
- 文档遵循统一模板
- 标签体系完整
- 文档间有交叉引用