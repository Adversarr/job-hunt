# Open-Instruct Code Map

> **项目**: Open-Instruct - AllenAI 开源指令微调与后训练框架
> **版本**: 0.1.0
> **最后更新**: 2026-03-02

---

## 1. 项目概述

Open-Instruct 是一个用于大型语言模型指令微调和后训练的完整框架，支持 SFT、DPO、PPO、GRPO 等多种训练方法。该项目是 Tulu 系列模型的训练基础，由 AllenAI 维护。

### 1.1 核心能力

| 能力领域 | 支持内容 |
|---------|---------|
| **监督微调 (SFT)** | 全参数微调、LoRA、QLoRA、无 padding 训练 |
| **偏好学习** | DPO、SimPO、WPO 等多种损失函数 |
| **强化学习** | PPO、GRPO、RLVR (带可验证奖励的 RL) |
| **工具使用** | MCP 工具、自定义环境、代码执行 |
| **分布式训练** | DeepSpeed ZeRO、FSDP、TP/PP、Ray |
| **推理优化** | vLLM 集成、FlashAttention-2/3 |

### 1.2 技术栈

```
PyTorch >= 2.9.0
Transformers >= 4.57.0
DeepSpeed >= 0.18.3
vLLM == 0.14.1
Ray >= 2.49.2
OLMo-core == 2.3.0
Flash-Attention >= 2.8.3
```

---

## 2. 目录结构

```
open-instruct/
├── open_instruct/                 # 核心源代码 (64+ 文件)
│   ├── 训练脚本
│   │   ├── finetune.py           # SFT 训练主入口
│   │   ├── dpo.py                # DPO 训练 (OLMo-core 版本)
│   │   ├── dpo_tune_cache.py     # DPO 训练 (Accelerate 版本)
│   │   ├── grpo_fast.py          # GRPO/RLVR 训练主入口
│   │   └── reward_modeling.py    # 奖励模型训练
│   ├── 数据处理
│   │   ├── data_loader.py        # 数据加载器 (HF + Ray)
│   │   ├── dataset_processor.py  # 数据集预处理器
│   │   ├── dataset_transformation.py  # 数据转换与缓存
│   │   ├── mix_data.py           # 数据混合工具
│   │   └── padding_free_collator.py   # 无 padding 数据整理器
│   ├── RL 组件
│   │   ├── rl_utils.py           # RL 通用工具
│   │   ├── dpo_utils.py          # DPO 损失与配置
│   │   ├── grpo_utils.py         # GRPO 配置与工具
│   │   ├── actor_manager.py      # Ray Actor 管理器
│   │   └── vllm_utils.py         # vLLM 推理工具
│   ├── 环境与工具
│   │   ├── environments/         # RL 环境框架
│   │   │   ├── base.py          # 环境基类
│   │   │   ├── tools/           # 工具实现
│   │   │   │   ├── tools.py     # 工具注册与定义
│   │   │   │   ├── parsers.py   # 工具调用解析
│   │   │   │   └── generic_mcp.py  # MCP 工具支持
│   │   │   └── pool.py          # 环境池管理
│   │   └── ground_truth_utils.py  # 奖励验证器
│   ├── 模型工具
│   │   ├── model_utils.py        # 模型加载与保存
│   │   ├── merge_lora.py         # LoRA 合并
│   │   └── olmo_core_utils.py    # OLMo-core 适配
│   ├── 评估
│   │   ├── benchmark_generators.py  # 推理基准测试
│   │   ├── judge_utils.py        # LLM 评判工具
│   │   └── rubrics/              # 评分标准
│   └── 工具模块
│       ├── utils.py              # 通用工具函数
│       ├── data_types.py         # 数据类型定义
│       ├── logger_utils.py       # 日志工具
│       └── tensor_utils.py       # 张量操作
├── configs/                       # 配置文件
│   ├── beaker_configs/           # AI2 Beaker 云配置
│   ├── ds_configs/               # DeepSpeed 配置
│   ├── judge_configs/            # 评判模型配置
│   └── train_configs/            # 训练配置
│       ├── dpo/                  # DPO 配置
│       ├── olmo2/                # OLMo-2 配置
│       ├── sft/                  # SFT 配置
│       └── tulu3/                # Tulu-3 配置
├── scripts/                       # 训练和评估脚本
│   ├── train/                    # 训练脚本
│   │   ├── tulu3/                # Tulu-3 训练
│   │   ├── olmo2/                # OLMo-2 训练
│   │   ├── rlvr/                 # RLVR 训练
│   │   └── debug/                # 调试脚本
│   ├── eval/                     # 评估脚本
│   └── data/                     # 数据准备脚本
├── decontamination/              # 数据去重工具
├── docs/                         # 文档
├── tests/                        # 测试用例
└── human_eval/                   # 人工评估界面
```

---

## 3. 核心模块详解

### 3.1 训练模块

#### 3.1.1 SFT 训练 (`finetune.py`)

**文件路径**: `open_instruct/finetune.py` (959 行)

**功能**: 监督微调主入口，支持全参数、LoRA、QLoRA 训练

**关键类**:
```python
@dataclass
class FlatArguments:
    """SFT 训练参数配置"""
    exp_name: str                    # 实验名称
    model_name_or_path: str          # 模型路径
    dataset_mixer_list: list[str]    # 数据集混合列表
    # ... 100+ 配置项
```

**核心流程**:
1. 解析配置参数
2. 初始化 Accelerator (分布式)
3. 加载模型和 tokenizer
4. 应用 PEFT (LoRA/QLoRA)
5. 数据加载与预处理
6. 训练循环
7. 模型保存与上传

**依赖关系**:
```
finetune.py
├── dataset_transformation.py    # 数据转换
├── model_utils.py               # 模型工具
├── padding_free_collator.py     # 无 padding 整理
└── utils.py                     # 通用工具
```

#### 3.1.2 DPO 训练 (`dpo.py`)

**文件路径**: `open_instruct/dpo.py` (435 行)

**功能**: 使用 OLMo-core 进行 DPO 训练

**关键类**:
```python
# DPO 训练模块 (OLMo-core 集成)
class DPOTrainModule(TransformerTrainModule):
    """DPO 训练模块"""
    def train_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        # 前向传播 chosen + rejected
        # 计算 DPO 损失
        # 反向传播与优化
```

**支持的损失函数** (`dpo_utils.py`):
```python
class DPOLossType(enum.StrEnum):
    dpo = "dpo"                      # 标准 DPO
    dpo_norm = "dpo_norm"            # 归一化 DPO
    simpo = "simpo"                  # SimPO (无需参考模型)
    wpo = "wpo"                      # 加权偏好优化
```

#### 3.1.3 GRPO/RLVR 训练 (`grpo_fast.py`)

**文件路径**: `open_instruct/grpo_fast.py` (2395 行)

**功能**: 大规模强化学习训练，支持工具使用和可验证奖励

**架构特点**:
- **分布式推理**: 使用 Ray + vLLM
- **异步数据流**: Queue-based 数据管道
- **工具集成**: 支持 MCP 工具和自定义环境
- **奖励验证**: 内置数学、代码、LLM 评判验证器

**关键组件**:
```python
# 主训练流程
class TrainingWorkflow:
    - 初始化 Ray cluster
    - 启动 vLLM inference actors
    - 数据准备 actors
    - 训练循环: generation -> reward -> update

# Actor 管理
class ActorManager:
    - 管理 Ray queues
    - 监控训练状态
    - 提供 Dashboard
```

**配置类** (`grpo_utils.py`):
```python
@dataclass
class ExperimentConfig:
    # 训练配置
    learning_rate: float = 2e-5
    num_epochs: int = 1
    num_mini_batches: int = 1
    
    # RL 超参
    beta: float = 0.05              # KL 系数
    clip_lower: float = 0.2         # PPO 裁剪下限
    clip_higher: float = 0.2        # PPO 裁剪上限
    
    # 损失函数
    loss_fn: GRPOLossType = GRPOLossType.dapo
    
    # 工具使用
    stop_at_tool_calls: bool = True
```

### 3.2 数据处理模块

#### 3.2.1 数据转换与缓存 (`dataset_transformation.py`)

**文件路径**: `open_instruct/dataset_transformation.py` (2116 行)

**功能**: 数据集转换、混合、缓存管理

**核心特性**:
- **智能缓存**: 基于配置哈希的自动缓存
- **数据集混合**: 支持多数据集按比例混合
- **Tokenizer 配置**: 统一 tokenization 配置

**关键函数**:
```python
# 获取缓存数据集
def get_cached_dataset_tulu(
    dataset_mixer_list: list[str],
    tc: TokenizerConfig,
    dataset_transform_fn: str,
    transform_fn_args: list[dict],
    # ... 更多参数
) -> Dataset:
    """
    根据配置哈希自动缓存/加载数据集
    支持分布式环境下的缓存同步
    """

# 配置哈希
def compute_config_hash(*args) -> str:
    """计算配置的 SHA256 哈希用于缓存键"""
```

**关键常量**:
```python
# 数据集列名
INPUT_IDS_KEY = "input_ids"
LABELS_KEY = "labels"
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
TOOLS_COLUMN_KEY = "tools"
```

#### 3.2.2 数据加载器 (`data_loader.py`)

**文件路径**: `open_instruct/data_loader.py` (1306 行)

**功能**: 多模态数据加载，支持 HuggingFace 和 Ray Streaming

**关键类**:
```python
# HuggingFace 数据加载器
class HFDataLoader(DataLoaderBase):
    """适配 olmo_core 的 HF 数据集加载器"""
    
# 流式数据加载配置
@dataclass
class StreamingDataLoaderConfig:
    num_prefetch_batches: int = 2
    inference_batch_size: int = 32
    
# vLLM 配置
@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
```

**数据准备 Actor**:
```python
class DataPreparationActor:
    """Ray Actor 用于异步数据准备"""
    - 接收 generation 结果
    - 计算奖励
    - 打包序列
    - 推送到训练队列
```

### 3.3 RL 组件

#### 3.3.1 DPO 工具 (`dpo_utils.py`)

**文件路径**: `open_instruct/dpo_utils.py` (1219 行)

**核心功能**:
- DPO 损失计算
- 参考模型 logprobs 缓存
- 多种 DPO 变体实现

**关键函数**:
```python
# DPO 损失计算
def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    loss_type: DPOLossType = DPOLossType.dpo,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算 DPO 损失和指标
    
    Returns:
        loss: 标量损失值
        metrics: 包含 accuracy, margin 等指标的字典
    """

# 批次 logprobs 计算
def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """计算每个样本的平均 log probability"""
```

#### 3.3.2 GRPO 工具 (`grpo_utils.py`)

**文件路径**: `open_instruct/grpo_utils.py` (378 行)

**支持的损失函数**:
```python
class GRPOLossType(enum.StrEnum):
    dapo = "dapo"        # DAPO (Dr. GRPO variant)
    cispo = "cispo"      # CISPO 损失
```

**关键配置**:
```python
@dataclass
class ExperimentConfig:
    # 采样配置
    num_samples_per_prompt: int = 4    # 每个 prompt 的样本数
    num_episodes_per_sync: int = 8     # 每次同步的 episode 数
    
    # KL 控制
    kl_estimator: Literal[0, 1, 2, 3] = 2
    ref_policy_update_freq: int | None = None  # 参考模型更新频率
    
    # 奖励配置
    reward_model_multiplier: float = 1.0
    apply_verifier_reward: bool = True
```

### 3.4 环境与工具系统

#### 3.4.1 环境基类 (`environments/base.py`)

**文件路径**: `open_instruct/environments/base.py` (131 行)

**核心抽象**:
```python
class RLEnvironment(ABC):
    """RL 环境和工具的抽象基类"""
    
    @abstractmethod
    async def reset(self, **kwargs) -> tuple[StepResult, list[dict]]:
        """初始化 episode，返回初始观察和工具定义"""
        
    @abstractmethod
    async def step(self, call: EnvCall) -> StepResult:
        """执行动作，返回观察、奖励、是否结束"""
        
    @abstractmethod
    def state(self) -> State:
        """返回当前 episode 状态"""

class TextRLEnvironment(RLEnvironment):
    """文本环境基类（用于对话式环境）"""
    response_role: str = "user"
```

#### 3.4.2 工具实现 (`environments/tools/`)

**目录结构**:
```
environments/tools/
├── tools.py          # 工具注册与基础实现
├── parsers.py        # 工具调用解析器
├── generic_mcp.py    # MCP (Model Context Protocol) 支持
├── utils.py          # 工具辅助函数
└── servers/          # 工具服务器实现
```

**工具注册**:
```python
TOOL_REGISTRY: dict[str, type[Tool]] = {}

class Tool(ABC):
    """工具基类"""
    config_name: str = ""
    
    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """返回 OpenAI 格式的工具定义"""
        
    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """执行工具"""
```

### 3.5 奖励验证系统 (`ground_truth_utils.py`)

**文件路径**: `open_instruct/ground_truth_utils.py` (1423 行)

**支持的验证器类型**:

| 验证器 | 用途 | 示例数据集 |
|-------|------|----------|
| `MathVerifier` | 数学问题验证 | GSM8K, MATH |
| `CodeVerifier` | 代码执行验证 | HumanEval, MBPP |
| `LMJudgeVerifier` | LLM 评判 | 开放式生成 |
| `RegexVerifier` | 正则匹配 | 特定格式输出 |
| `IFEvalVerifier` | 指令遵循验证 | IFEval |

**验证器基类**:
```python
class VerifierFunction(ABC):
    """奖励验证函数基类"""
    
    @abstractmethod
    def __call__(self, response: str, ground_truth: Any) -> VerificationResult:
        """
        验证模型输出
        
        Args:
            response: 模型生成的响应
            ground_truth: 参考答案
            
        Returns:
            VerificationResult(score, cost)
        """
```

**自动注册机制**:
```python
# 所有子类自动注册到 REWARD_FN_MAPPING
REWARD_FN_MAPPING: dict[str, type[VerifierFunction]] = {}

# 使用示例
verifier = build_verifier("math", config)
result = verifier("模型输出", "参考答案")
```

### 3.6 推理与生成

#### 3.6.1 vLLM 集成 (`vllm_utils.py`)

**文件路径**: `open_instruct/vllm_utils.py` (1296 行)

**核心功能**:
- Ray Actor 管理的 vLLM 推理
- 分布式张量并行
- 权重热更新 (weight update)
- OpenAI API 兼容服务器

**关键类**:
```python
class LLMRayActor:
    """vLLM Ray Actor，管理模型推理"""
    
    def init_model(self):
        """初始化 vLLM 引擎"""
        
    def generate(self, prompts: list[str], sampling_params) -> list[str]:
        """批量生成"""
        
    def update_weights(self, new_weights: dict):
        """热更新模型权重"""
```

**数据类型** (`data_types.py`):
```python
@dataclass
class GenerationResult:
    responses: list[list[int]]           # 生成的 token IDs
    finish_reasons: list[str]            # 结束原因
    masks: list[list[int]]               # 注意力掩码
    token_statistics: TokenStatistics    # Token 统计
    logprobs: list[list[float]] | None   # Log probabilities
    reward_scores: list[float] | None    # 奖励分数

@dataclass
class PromptRequest:
    prompt: list[int]                    # Tokenized prompt
    generation_config: Any               # 生成配置
    active_tools: list[str] | None       # 激活的工具列表
```

### 3.7 模型工具 (`model_utils.py`)

**文件路径**: `open_instruct/model_utils.py` (768 行)

**核心功能**:
```python
# 模型配置
@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"

# 张量缓存
@dataclass
class TensorCache:
    """用于缓存 reference model 的 logprobs"""
    tensors: dict[str, torch.Tensor]
    
    def to_disk(self, path: str) -> None:
        """原子保存到磁盘"""
        
    @classmethod
    def from_disk(cls, path: str, device: torch.device) -> "TensorCache":
        """从磁盘加载"""

# 批次容器
@dataclass
class Batch:
    queries: list[list[int]]             # Prompt token IDs
    ground_truths: list[list[int]]       # 参考答案
    datasets: list[str]                  # 数据集来源
    scores: list[float] | None           # 奖励分数
```

---

## 4. 模块依赖关系

### 4.1 训练流程依赖图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Flow                           │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   finetune   │      │     dpo      │      │  grpo_fast   │
  │    (SFT)     │      │  (DPO/OLMo)  │      │   (GRPO)     │
  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │              dataset_transformation.py                    │
  │         (数据转换、混合、缓存)                              │
  └──────────────────────────────────────────────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  data_loader │      │  data_loader │      │  data_loader │
  │   (HFData    │      │   (HFData    │      │ (Streaming   │
  │   Loader)    │      │   Loader)    │      │  Config)     │
  └──────────────┘      └──────────────┘      └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │  vllm_utils  │
                                              │ (LLMRayActor)│
                                              └──────┬───────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              │                      │                      │
                              ▼                      ▼                      ▼
                       ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
                       │ground_truth_ │      │  vllm_utils  │      │  rl_utils    │
                       │   utils      │      │  (inference) │      │ (rollout)    │
                       │ (verifiers)  │      └──────────────┘      └──────────────┘
                       └──────────────┘
```

### 4.2 数据流图 (GRPO)

```
┌────────────────────────────────────────────────────────────────────────┐
│                          GRPO Data Flow                                │
└────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │   Dataset    │
  └──────┬───────┘
         │ Tokenize
         ▼
  ┌──────────────┐
  │ PromptQueue  │ ◄────┐
  └──────┬───────┘      │
         │              │
         ▼              │
  ┌──────────────┐      │
  │ vLLM Actors  │      │
  │ (inference)  │      │
  └──────┬───────┘      │
         │ Generate     │
         ▼              │
  ┌──────────────┐      │
  │  Response    │      │
  │    Queue     │      │
  └──────┬───────┘      │
         │              │
         ▼              │
  ┌──────────────┐      │
  │DataPreparation│     │
  │    Actor     │      │
  │(reward calc) │      │
  └──────┬───────┘      │
         │              │
         ▼              │
  ┌──────────────┐      │
  │ Training     │      │
  │   Queue      │      │
  └──────┬───────┘      │
         │              │
         ▼              │
  ┌──────────────┐      │
  │  DeepSpeed   │      │
  │  (training)  │      │
  └──────┬───────┘      │
         │ Update       │
         │ weights      │
         └──────────────┘
```

### 4.3 核心工具依赖

```
utils.py (基础工具)
    ├── logger_utils.py (日志)
    ├── tensor_utils.py (张量)
    ├── launch_utils.py (启动)
    └── data_types.py (类型)

model_utils.py (模型)
    ├── utils.py
    └── ground_truth_utils.py

dataset_transformation.py (数据)
    ├── utils.py
    ├── launch_utils.py
    └── logger_utils.py

grpo_fast.py (GRPO)
    ├── utils.py
    ├── model_utils.py
    ├── data_loader.py
    ├── vllm_utils.py
    ├── rl_utils.py
    ├── ground_truth_utils.py
    ├── actor_manager.py
    └── environments/

dpo.py (DPO)
    ├── utils.py
    ├── dpo_utils.py
    ├── model_utils.py
    ├── olmo_core_utils.py
    └── olmo_core_train_modules.py
```

---

## 5. 关键配置与常量

### 5.1 训练配置常量

```python
# 文件: open_instruct/grpo_utils.py

# 损失函数类型
class GRPOLossType(enum.StrEnum):
    dapo = "dapo"
    cispo = "cispo"

# 文件: open_instruct/dpo_utils.py
class DPOLossType(enum.StrEnum):
    dpo = "dpo"
    dpo_norm = "dpo_norm"
    simpo = "simpo"
    wpo = "wpo"

# 文件: open_instruct/utils.py
INVALID_LOGPROB = 1.0  # 无效 logprob 的哨兵值
```

### 5.2 数据集列名常量

```python
# 文件: open_instruct/dataset_processor.py

# SFT 数据集
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
SFT_MESSAGE_KEY = "messages"

# 偏好数据集
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
ATTENTION_MASK_CHOSEN_KEY = "attention_mask_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
ATTENTION_MASK_REJECTED_KEY = "attention_mask_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"

# 特殊列
GROUND_TRUTHS_KEY = "ground_truth"
VERIFIER_SOURCE_KEY = "dataset"
TOOLS_COLUMN_KEY = "tools"
ENV_CONFIG_KEY = "env_config"
RAW_PROMPT_KEY = "raw_prompt"
```

### 5.3 超时与性能常量

```python
# 文件: open_instruct/vllm_utils.py
NUM_PREFETCH_WORKERS = 2
DRAIN_ACTIVE_TASKS_SLEEP_S = 1
SHOULD_STOP_TIMEOUT_S = 0.1
INFERENCE_INIT_TIMEOUT_S = 1200  # 20 分钟
VLLM_HEALTH_CHECK_TIMEOUT_S = 600.0  # 10 分钟

# 文件: open_instruct/rl_utils.py
ROLLOUT_SHARD_SIZE = 10000
```

---

## 6. 扩展点与开发指南

### 6.1 添加新的验证器

```python
# 在 ground_truth_utils.py 中添加

class MyVerifier(VerifierFunction):
    """自定义验证器"""
    
    def __call__(self, response: str, ground_truth: Any) -> VerificationResult:
        # 实现验证逻辑
        score = self.compute_score(response, ground_truth)
        return VerificationResult(score=score, cost=0.0)

# 自动注册，无需额外代码
# 可以通过名称 "my_verifier" 使用
```

### 6.2 添加新的工具

```python
# 在 environments/tools/tools.py 中添加

@register_tool
class MyTool(Tool):
    config_name = "my_tool"
    
    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Tool description",
                "parameters": {...}
            }
        }]
    
    async def execute(self, param1: str, param2: int) -> ToolResult:
        # 工具逻辑
        return ToolResult(output="result")
```

### 6.3 添加新的环境

```python
# 在 environments/ 下创建新文件

class MyEnvironment(RLEnvironment):
    config_name = "my_env"
    
    async def reset(self, **kwargs) -> tuple[StepResult, list[dict]]:
        # 初始化环境
        return StepResult(result="", reward=0.0), []
    
    async def step(self, call: EnvCall) -> StepResult:
        # 执行动作
        return StepResult(result="", reward=1.0)
```

---

## 7. 测试结构

```
open_instruct/test_*.py          # 各模块的单元测试
tests/                           # 集成测试
├── test_*.py
```

**主要测试文件**:
- `test_data_loader_gpu.py` - 数据加载器 GPU 测试
- `test_grpo_fast_gpu.py` - GRPO GPU 测试
- `test_dpo_utils_gpu.py` - DPO GPU 测试
- `test_olmo_core_train_modules.py` - OLMo-core 模块测试

---

## 8. 脚本说明

### 8.1 训练脚本

```bash
# SFT 训练
scripts/train/tulu3/finetune_8b.sh
scripts/train/tulu3/finetune_70b.sh

# DPO 训练
scripts/train/tulu3/dpo_8b.sh

# GRPO/RLVR 训练
scripts/train/rlvr/tulu_rlvr.sh
scripts/train/debug/single_gpu_on_beaker.sh
```

### 8.2 启动工具

```bash
# 构建镜像并启动
scripts/train/build_image_and_launch.sh $SCRIPT

# 调试脚本
scripts/train/debug/tools/olmo_3_parser_multigpu.sh
```

---

## 9. 项目路线图

### 9.1 已支持模型

- ✅ Llama 3/3.1 (8B, 70B, 405B)
- ✅ OLMo 2 (7B, 13B)
- ✅ Qwen 2.5/3
- ✅ Mixtral (MoE)

### 9.2 已支持训练方法

- ✅ SFT (全参数、LoRA、QLoRA)
- ✅ DPO (标准、归一化、SimPO、WPO)
- ✅ PPO (Ray + vLLM)
- ✅ GRPO (DAPO、CISPO)
- ✅ 奖励模型训练

### 9.3 已支持特性

- ✅ FlashAttention-2/3
- ✅ 无 padding 训练
- ✅ 工具使用 (MCP)
- ✅ 代码执行环境
- ✅ 数学验证器
- ✅ LLM 评判

---

## 10. 相关链接

- **项目主页**: https://github.com/allenai/open-instruct
- **Tulu 3 论文**: https://arxiv.org/abs/2411.15124
- **文档**: https://github.com/allenai/open-instruct/tree/main/docs
- **模型**: https://huggingface.co/collections/allenai/tulu-3-suite-67413dad47ab46c5c2babaf0

---

## 附录: 文件索引

### 核心训练文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `finetune.py` | 959 | SFT 训练主脚本 |
| `dpo.py` | 435 | DPO 训练 (OLMo-core) |
| `dpo_tune_cache.py` | ~800 | DPO 训练 (Accelerate) |
| `grpo_fast.py` | 2395 | GRPO/RLVR 训练 |
| `reward_modeling.py` | 424 | 奖励模型训练 |

### 核心数据文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `dataset_transformation.py` | 2116 | 数据转换与缓存 |
| `data_loader.py` | 1306 | 数据加载器 |
| `dataset_processor.py` | 369 | 数据集预处理 |
| `padding_free_collator.py` | 262 | 无 padding 数据整理 |

### 核心 RL 文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `vllm_utils.py` | 1296 | vLLM 推理工具 |
| `rl_utils.py` | 448 | RL 通用工具 |
| `dpo_utils.py` | 1219 | DPO 工具与损失 |
| `grpo_utils.py` | 378 | GRPO 配置 |
| `actor_manager.py` | 257 | Ray Actor 管理 |

### 核心工具文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `ground_truth_utils.py` | 1423 | 奖励验证器 |
| `model_utils.py` | 768 | 模型工具 |
| `utils.py` | 2623 | 通用工具 |
| `data_types.py` | 98 | 数据类型定义 |

### 环境相关文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `environments/base.py` | 131 | 环境基类 |
| `environments/tools/tools.py` | ~500 | 工具注册与实现 |
| `environments/tools/parsers.py` | ~300 | 工具解析器 |
| `environments/pool.py` | ~200 | 环境池管理 |

---

*文档生成时间: 2026-03-02*
*基于 Open-Instruct commit: main branch*
