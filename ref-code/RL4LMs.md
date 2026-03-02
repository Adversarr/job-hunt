# RL4LMs Code Map

> **项目**: RL4LMs - A modular RL library to fine-tune language models to human preferences  
> **版本**: 0.2.2  
> **论文**: https://arxiv.org/abs/2210.01241  

---

## 1. 项目整体架构

RL4LMs 是一个基于 PyTorch 和 Stable-Baselines3 的强化学习框架，专门用于优化语言模型。它将 NLP 任务建模为 RL 环境，支持多种 on-policy 算法（PPO、NLPO、A2C、TRPO）来微调语言模型。

### 1.1 核心设计理念

- **环境即任务**: 每个 NLP 任务被建模为 Gym 风格的 RL 环境 (`TextGenEnv`)
- **策略即模型**: Actor-Critic 策略封装 Hugging Face Transformers 模型
- **奖励即指标**: NLG 评估指标（ROUGE、BLEU、BERTScore 等）作为奖励函数
- **配置驱动**: YAML 配置文件定义完整的训练流程

### 1.2 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Scripts                         │
│  (train_text_generation.py, SupervisedTrainer)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Trainer Layer                             │
│  (OnPolicyTrainer, SupervisedTrainer)                      │
│  - 构建所有组件 (tokenizer, env, policy, alg)              │
│  - 训练和评估循环                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 Algorithm Wrappers                           │
│  (wrap_onpolicy_alg)                                       │
│  - 包装 SB3 算法以支持文本生成                              │
│  - 处理 rollout、KL 散度、奖励计算                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              On-Policy Algorithms                            │
│  (PPO, NLPO, A2C, TRPO)                                    │
│  - 来自 stable-baselines3 的修改版本                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               Actor-Critic Policies                          │
│  (CausalLM/Seq2SeqLM + Maskable variants)                  │
│  - 包装 Hugging Face Transformers                          │
│  - 提供 policy/value/ref 模型                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Text Generation Environment                     │
│  (TextGenEnv)                                              │
│  - Gym 环境接口                                            │
│  - 管理 episode、observation、reward                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Data & Reward Layer                         │
│  (DataPools, RewardFunctions, Metrics)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
RL4LMs/
├── rl4lms/                          # 主库代码
│   ├── __init__.py
│   ├── algorithms/                  # RL 算法实现
│   │   ├── a2c/                     # A2C 算法
│   │   ├── common/                  # 共享组件
│   │   │   ├── algo_utils.py        # 算法工具
│   │   │   └── maskable/            # NLPO maskable 支持
│   │   │       ├── buffers.py       # Maskable rollout buffer
│   │   │       ├── distributions.py # Maskable 分布
│   │   │       └── policies.py      # Maskable 策略基类
│   │   ├── nlpo/                    # NLPO (Natural Language Policy Optimization)
│   │   ├── ppo/                     # PPO 算法
│   │   └── trpo/                    # TRPO 算法
│   ├── core_components/             # 核心组件
│   │   ├── sampler.py               # 优先级采样器
│   │   └── sweep.py                 # 超参搜索
│   ├── data_pools/                  # 数据集管理
│   │   ├── custom_text_generation_pools.py  # 预定义数据集
│   │   ├── text_generation_pool.py  # 数据集基类
│   │   └── task_utils/              # 任务特定工具
│   │       └── totto/               # ToTTo 数据集工具
│   └── envs/                        # 环境定义
│       ├── common/                  # 通用环境接口
│       │   ├── action_space.py
│       │   ├── base_env.py
│       │   ├── observation.py
│       │   └── reward.py
│       └── text_generation/         # 文本生成环境
│           ├── alg_wrappers.py      # 算法包装器 ⭐核心
│           ├── caption_metrics/     # 图像描述指标 (CIDER, SPICE)
│           ├── env.py               # TextGenEnv 环境 ⭐核心
│           ├── evaluation_utils.py  # 评估工具
│           ├── hf_generation_utils.py  # HF 生成工具
│           ├── kl_controllers.py    # KL 散度控制器
│           ├── logging_utils.py     # 日志和追踪
│           ├── metric.py            # 评估指标 ⭐核心
│           ├── observation.py       # Observation 类 ⭐核心
│           ├── policy/              # 策略子模块
│           │   ├── base_policy.py   # 基础策略 (旧版)
│           │   ├── causal_policy.py # Causal LM 策略
│           │   └── seq2seq_policy.py# Seq2Seq 策略
│           ├── policy.py            # 策略类 ⭐核心
│           ├── post_processors.py   # 后处理器
│           ├── preference_reward.py # 偏好奖励
│           ├── registry.py          # 注册表 ⭐核心
│           ├── reward.py            # 奖励函数 ⭐核心
│           ├── summ_metrics/        # 摘要指标 (SummaC)
│           ├── training_utils.py    # 训练工具 ⭐核心
│           ├── utils_supervised.py  # 监督学习工具
│           └── warm_start.py        # 热启动支持
├── scripts/                         # 脚本
│   └── training/
│       ├── train_text_generation.py # 主训练脚本 ⭐入口
│       └── task_configs/            # YAML 配置文件
│           ├── summarization/       # 摘要任务
│           ├── common_gen/          # 常识生成
│           ├── dialog/              # 对话生成
│           └── ...
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. 核心模块详解

### 3.1 数据层 (Data Layer)

#### 3.1.1 Sample 数据类
**文件**: `rl4lms/data_pools/text_generation_pool.py:7-13`

```python
@dataclass(init=True)
class Sample:
    id: str                          # 样本唯一ID
    prompt_or_input_text: str        # 输入提示/文本
    references: List[str]            # 参考文本列表
    meta_data: Dict[str, Any] = None # 元数据
```

#### 3.1.2 TextGenPool 基类
**文件**: `rl4lms/data_pools/text_generation_pool.py:15-47`

- **职责**: 所有数据集的抽象基类
- **关键方法**:
  - `prepare(cls, split: str, **args)`: 工厂方法，子类必须实现
  - `__getitem__`: 返回 (Sample, weight) 元组
  - `sample()`: 随机采样

#### 3.1.3 预定义数据集
**文件**: `rl4lms/data_pools/custom_text_generation_pools.py`

| 数据集 | 类名 | 任务类型 |
|--------|------|----------|
| CNN/DailyMail | `CNNDailyMail` | 摘要 |
| CommonGen | `CommonGen` | 常识生成 |
| IMDB | `IMDB` | 情感续写 |
| ToTTo | `ToTTo` | 表格到文本 |
| NarrativeQA | `NarrativeQA` | 抽象问答 |
| WMT16 | `WMT` | 机器翻译 |
| DailyDialog | `DailyDialog` | 对话生成 |

---

### 3.2 环境层 (Environment Layer)

#### 3.2.1 TextGenEnv
**文件**: `rl4lms/envs/text_generation/env.py:15-179`

**核心职责**: 将文本生成建模为 MDP (Markov Decision Process)

**状态空间** (`observation_space`):
```python
{
    "prompt_or_input_encoded_pt": 编码后的输入 (max_prompt_length,)
    "prompt_or_input_attention_mask_pt": 输入 attention mask
    "context_encoded_pt": 已生成文本的编码 (max_episode_length,)
    "context_attention_mask_pt": 上下文 attention mask
    "input_encoded_pt": 拼接后的完整输入
    "input_attention_mask_pt": 完整 attention mask
}
```

**动作空间** (`action_space`): 
- 离散空间，大小为词表大小 (通常 32128 for T5, 50257 for GPT2)

**核心方法**:
- `reset(sample: Sample)`: 开始新 episode，从样本初始化 observation
- `step(action: int)`: 执行动作（生成 token），返回 (obs, reward, done, info)

#### 3.2.2 Observation 类
**文件**: `rl4lms/envs/text_generation/observation.py:10-178`

**职责**: 封装环境状态，管理输入/上下文编码

**关键属性**:
- `prompt_or_input_*`: 原始输入
- `context_*`: 当前已生成内容
- `input_*`: prompt + context 拼接

**关键方法**:
- `init_from_sample()`: 从 Sample 初始化 observation
- `update(action, tokenizer)`: 执行动作后更新状态
- `to_dict()`: 转换为 numpy 数组（供 SB3 使用）

---

### 3.3 奖励层 (Reward Layer)

#### 3.3.1 RewardFunction 基类
**文件**: `rl4lms/envs/text_generation/reward.py:23-45`

```python
class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(self, 
                 current_observation: Observation,
                 action: int,
                 next_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
```

#### 3.3.2 BatchedRewardFunction
**文件**: `rl4lms/envs/text_generation/reward.py:48-66`

用于批量计算奖励，提高效率。

#### 3.3.3 预定义奖励函数

| 奖励函数 | 类名 | 说明 |
|----------|------|------|
| Rouge | `RougeRewardFunction` | ROUGE-1/2/L |
| BERTScore | `BERTScoreRewardFunction` | 语义相似度 |
| BLEU | `BLEURewardFunction` | BLEU 分数 |
| Meteor | `MeteorRewardFunction` | Meteor 指标 |
| BLEURT | `BLEURTRewardFunction` | 学习评估指标 |
| Learned | `LearnedRewardFunction` | 分类器作为奖励 |
| Spider | `SpiderRewardFunction` | CIDER + SPICE |
| PARENT | `PARENTRewardFunction` | ToTTo 任务 |
| IntentAccuracy | `IntentAccuracy` | 对话任务 |

**注册表**: `rl4lms/envs/text_generation/registry.py:115-146`

---

### 3.4 策略层 (Policy Layer)

#### 3.4.1 策略类层次

```
BasePolicy (SB3)
    └── LMActorCriticPolicy
            ├── CausalLMActorCriticPolicy       # GPT 风格
            ├── Seq2SeqLMActorCriticPolicy      # T5/BART 风格
            ├── MaskedCausalLMActorCriticPolicy # NLPO + GPT
            └── MaskedSeq2SeqLMActorCriticPolicy # NLPO + T5
```

**文件**: `rl4lms/envs/text_generation/policy.py`

#### 3.4.2 LMActorCriticPolicy 核心组件

**初始化** (`__init__`):
- `model_name`: Hugging Face 模型名
- `_policy_model`: 策略模型（用于生成）
- `_value_model`: Value 模型（用于估计状态价值）
- `_ref_model`: 参考模型（用于 KL 散度计算）
- `_value_head`: 价值头（线性层输出标量价值）

**关键方法**:

1. **`forward_policy()`**: 策略前向传播
   - 输入: observation, actions
   - 输出: actions, log_prob, entropy, outputs, past_model_kwargs

2. **`forward_value()`**: Value 估计
   - 输入: observation
   - 输出: values, past_model_kwargs

3. **`get_log_probs_ref_model()`**: 参考模型 log prob
   - 用于计算 KL 散度

4. **`generate()`**: 生成文本
   - 使用 HF 的 `generate()` 方法
   - 返回生成的 tokens、log probs、文本

5. **`evaluate_actions()`**: 评估动作（训练时使用）
   - 返回: values, log_prob, entropy

#### 3.4.3 Seq2SeqLMActorCriticPolicy

专为 encoder-decoder 模型设计（T5, BART）：
- 分别编码 encoder 和 decoder
- 管理 `decoder_attention_mask`
- 支持 encoder-decoder 架构的特殊处理

#### 3.4.4 Maskable Policies (NLPO)

支持动作掩码，限制可选 token：
- `top_mask`: 只保留 top-k 概率的 token
- `mask_type`: 'learned_top_k' 或 'ref_top_k'
- 通过 `MaskLogitsProcessor` 实现

---

### 3.5 算法层 (Algorithm Layer)

#### 3.5.1 算法实现

| 算法 | 文件 | 说明 |
|------|------|------|
| PPO | `rl4lms/algorithms/ppo/ppo.py` | Proximal Policy Optimization |
| NLPO | `rl4lms/algorithms/nlpo/nlpo.py` | PPO + Invalid Action Masking |
| A2C | `rl4lms/algorithms/a2c/a2c.py` | Advantage Actor-Critic |
| TRPO | `rl4lms/algorithms/trpo/trpo.py` | Trust Region Policy Optimization |

#### 3.5.2 PPO 核心逻辑

**文件**: `rl4lms/algorithms/ppo/ppo.py:179-328`

**训练步骤** (`train()`):
1. 从 rollout buffer 采样 batch
2. 重新计算 log probs 和 values
3. 计算 PPO 损失:
   - Policy loss (clipped surrogate)
   - Value loss (MSE)
   - Entropy loss
4. 反向传播和梯度裁剪
5. 记录训练指标

**关键公式**:
```
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clamp(ratio, 1 - clip_range, 1 + clip_range)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages)
```

#### 3.5.3 NLPO 特点

**文件**: `rl4lms/algorithms/nlpo/nlpo.py`

- 使用 `MaskableRolloutBuffer` 存储 action masks
- 支持动作掩码约束
- 定期更新 mask model (`update_mask_model()`)

---

### 3.6 算法包装层 (Wrapper Layer)

#### 3.6.1 wrap_onpolicy_alg
**文件**: `rl4lms/envs/text_generation/alg_wrappers.py:90-408`

**职责**: 将 SB3 算法适配到文本生成任务

**包装功能**:
1. **Rollout 生成** (`generate_batch()`):
   - 使用 policy 生成文本序列
   - 逐 token 计算 log probs、values、KL 散度
   - 存储到 rollout buffer

2. **KL 散度控制**:
   - 计算与参考模型的 KL 散度
   - 作为惩罚加入奖励
   - 使用 `KLController` 自适应调整系数

3. **奖励计算**:
   - 支持 step-wise 和 episode-end 奖励
   - 支持 BatchedRewardFunction 批量计算

4. **Advantage 计算**:
   - 使用 GAE (Generalized Advantage Estimation)
   - `compute_returns_and_advantage()`

**TransitionInfo**:
```python
@dataclass
class TransitionInfo:
    observation: TensorDict
    action: np.ndarray
    task_reward: np.ndarray
    total_reward: np.ndarray  # task + kl
    kl_div: np.ndarray
    episode_start: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    done: np.ndarray
    ref_log_prob: torch.Tensor
    kl_reward: np.ndarray
    action_mask: np.ndarray
    info: Dict[str, Any]
```

---

### 3.7 KL 控制器

**文件**: `rl4lms/envs/text_generation/kl_controllers.py:5-54`

```python
class KLController:
    def __init__(self, kl_coeff: float, target_kl: Optional[float] = None)
    
    def step(self, kl_div: torch.tensor):
        # 自适应调整 kl_coeff
        # 如果 kl_div > target_kl: 增加惩罚
        # 如果 kl_div < target_kl: 减少惩罚
```

---

### 3.8 训练层 (Training Layer)

#### 3.8.1 OnPolicyTrainer
**文件**: `rl4lms/envs/text_generation/training_utils.py:126-223`

**职责**: 完整的 RL 训练流程

**初始化流程** (`_setup()`):
1. 加载训练状态（checkpoint）
2. 构建 tokenizer
3. 构建 reward function
4. 构建 metrics
5. 构建 datapool (train/val/test)
6. 构建 vectorized environment
7. 构建 algorithm

**训练循环** (`train_and_eval()`):
```python
for epoch in range(n_iters):
    # 1. RL 训练
    alg.learn(n_steps_per_iter)
    
    # 2. 保存 checkpoint
    if (epoch + 1) % save_every == 0:
        save_trainer_state()
    
    # 3. 评估
    if (epoch + 1) % eval_every == 0:
        evaluate_on_datapools()
```

#### 3.8.2 SupervisedTrainer
**文件**: `rl4lms/envs/text_generation/training_utils.py:225-321`

使用 Hugging Face Trainer 进行监督学习：
- 支持 Causal LM 和 Seq2Seq LM
- 自动处理数据预处理
- 集成评估回调

#### 3.8.3 构建函数

| 函数 | 职责 |
|------|------|
| `build_tokenizer()` | 加载 tokenizer，配置 padding/truncation |
| `build_reward_fn()` | 从注册表获取奖励函数 |
| `build_metrics()` | 构建评估指标列表 |
| `build_datapool()` | 加载 train/val/test 数据 |
| `build_env()` | 构建 vectorized environment |
| `build_alg()` | 构建 RL 算法 |

---

### 3.9 评估层 (Evaluation Layer)

#### 3.9.1 BaseMetric
**文件**: `rl4lms/envs/text_generation/metric.py:21-42`

```python
class BaseMetric(ABC):
    def compute(self, 
                prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, Tuple]:
        # 返回: {metric_name: (individual_scores, corpus_score)}
```

#### 3.9.2 预定义指标

**文件**: `rl4lms/envs/text_generation/metric.py`

- **Lexical**: ROUGE, BLEU, METEOR, chrF, TER
- **Semantic**: BERTScore, BLEURT
- **Caption**: CIDER, SPICE
- **Summary**: SummaC (ZS/Conv)
- **Diversity**: Distinct-N, MSTTR
- **Task-specific**: PARENT (ToTTo), IntentAccuracy (DailyDialog)

---

### 3.10 注册表系统 (Registry)

**文件**: `rl4lms/envs/text_generation/registry.py`

统一注册和管理所有可配置组件：

| 注册表 | 管理对象 |
|--------|----------|
| `DataPoolRegistry` | 数据集 |
| `RewardFunctionRegistry` | 奖励函数 |
| `MetricRegistry` | 评估指标 |
| `PolicyRegistry` | 策略类 |
| `AlgorithmRegistry` | RL 算法 |
| `WrapperRegistry` | 算法包装器 |
| `PostProcessorRegistry` | 后处理器 |

**使用示例**:
```python
# 从配置动态获取组件
reward_fn = RewardFunctionRegistry.get("rouge", {"rouge_type": "rouge1"})
policy_cls = PolicyRegistry.get("causal_lm_actor_critic_policy")
```

---

## 4. 关键文件索引

### 4.1 核心入口

| 文件 | 说明 |
|------|------|
| `scripts/training/train_text_generation.py` | 主训练脚本 |
| `rl4lms/envs/text_generation/training_utils.py` | Trainer 实现 |

### 4.2 环境相关

| 文件 | 说明 |
|------|------|
| `rl4lms/envs/text_generation/env.py` | TextGenEnv 环境 |
| `rl4lms/envs/text_generation/observation.py` | Observation 类 |
| `rl4lms/envs/text_generation/alg_wrappers.py` | 算法包装器 |

### 4.3 策略相关

| 文件 | 说明 |
|------|------|
| `rl4lms/envs/text_generation/policy.py` | 策略基类和实现 |
| `rl4lms/envs/text_generation/policy/causal_policy.py` | Causal LM 策略 |
| `rl4lms/envs/text_generation/policy/seq2seq_policy.py` | Seq2Seq 策略 |

### 4.4 算法相关

| 文件 | 说明 |
|------|------|
| `rl4lms/algorithms/ppo/ppo.py` | PPO 实现 |
| `rl4lms/algorithms/nlpo/nlpo.py` | NLPO 实现 |
| `rl4lms/algorithms/common/maskable/buffers.py` | Maskable Rollout Buffer |

### 4.5 奖励和评估

| 文件 | 说明 |
|------|------|
| `rl4lms/envs/text_generation/reward.py` | 奖励函数 |
| `rl4lms/envs/text_generation/metric.py` | 评估指标 |
| `rl4lms/envs/text_generation/kl_controllers.py` | KL 控制器 |

### 4.6 数据

| 文件 | 说明 |
|------|------|
| `rl4lms/data_pools/text_generation_pool.py` | 数据集基类 |
| `rl4lms/data_pools/custom_text_generation_pools.py` | 预定义数据集 |

### 4.7 配置和注册

| 文件 | 说明 |
|------|------|
| `rl4lms/envs/text_generation/registry.py` | 组件注册表 |

---

## 5. 模块依赖关系

```
train_text_generation.py
    ├── Tracker (logging_utils.py)
    ├── OnPolicyTrainer/SupervisedTrainer (training_utils.py)
    │   ├── build_tokenizer() → AutoTokenizer
    │   ├── build_reward_fn() → RewardFunctionRegistry
    │   ├── build_metrics() → MetricRegistry
    │   ├── build_datapool() → DataPoolRegistry
    │   ├── build_env() → TextGenEnv (env.py)
    │   │   ├── Observation (observation.py)
    │   │   └── RewardFunction (reward.py)
    │   └── build_alg() → AlgorithmRegistry + WrapperRegistry
    │       └── wrap_onpolicy_alg() (alg_wrappers.py)
    │           ├── OnPolicyAlgorithm (ppo.py, nlpo.py)
    │           ├── LMActorCriticPolicy (policy.py)
    │           │   ├── AutoModelForCausalLM/Seq2SeqLM
    │           │   └── CategoricalDistribution
    │           ├── KLController (kl_controllers.py)
    │           └── MaskableRolloutBuffer (buffers.py)
    └── evaluate_on_samples() (evaluation_utils.py)
        └── BaseMetric (metric.py)
```

---

## 6. 配置系统

### 6.1 YAML 配置结构

```yaml
# 1. 数据集配置
datapool:
  id: cnn_daily_mail          # DataPoolRegistry key
  args:
    prompt_prefix: "Summarize: "

# 2. Tokenizer 配置
tokenizer:
  model_name: t5-base
  padding_side: left
  truncation_side: left

# 3. 奖励函数配置
reward_fn:
  id: rouge                   # RewardFunctionRegistry key
  args:
    rouge_type: "rouge1"

# 4. 环境配置
env:
  n_envs: 10                  # 并行环境数
  args:
    max_prompt_length: 512
    max_episode_length: 100
    terminate_on_eos: True

# 5. 算法配置
alg:
  id: ppo                     # AlgorithmRegistry key
  args:
    n_steps: 512
    batch_size: 64
    n_epochs: 5
    learning_rate: 0.000002
  kl_div:
    coeff: 0.001              # KL 惩罚系数
    target_kl: 0.2            # 自适应目标 KL
  policy:
    id: seq2seq_lm_actor_critic_policy  # PolicyRegistry key
    args:
      model_name: t5-base
      generation_kwargs:
        do_sample: True
        top_k: 50
        max_new_tokens: 100

# 6. 训练评估配置
train_evaluation:
  n_iters: 100
  eval_every: 10
  save_every: 1
  eval_batch_size: 100
  metrics:
    - id: rouge
    - id: meteor
    - id: bert_score
  generation_kwargs:
    temperature: 0.7
```

---

## 7. 训练流程时序

```
1. 初始化阶段
   ├── 加载配置文件
   ├── 初始化 Tracker
   └── 构建所有组件

2. 训练循环 (for epoch in n_iters)
   │
   ├── 2.1 Collect Rollouts
   │   ├── env.reset() → 获取初始 observation
   │   ├── policy.generate() → 生成文本序列
   │   ├── for each token:
   │   │   ├── policy.forward_policy() → log_prob
   │   │   ├── policy.forward_value() → value
     │   │   ├── policy.get_log_probs_ref_model() → ref_log_prob
   │   │   ├── compute KL reward
   │   │   ├── env.step(action) → task_reward
   │   │   └── store to rollout_buffer
   │   └── compute returns and advantages (GAE)
   │
   ├── 2.2 Train Algorithm
   │   ├── for epoch in n_epochs:
   │   │   └── for batch in rollout_buffer:
   │   │       ├── evaluate_actions() → new_log_prob, value, entropy
   │   │       ├── compute PPO loss
   │   │       ├── backward() + clip_grad_norm()
   │   │       └── optimizer.step()
   │   └── log training metrics
   │
   ├── 2.3 Save Checkpoint (可选)
   │
   └── 2.4 Evaluation (可选)
       ├── policy.generate() on val set
       ├── compute metrics
       └── log eval results

3. 结束阶段
   └── 保存最终模型
```

---

## 8. 扩展点

### 8.1 添加新数据集

```python
from rl4lms.data_pools.text_generation_pool import Sample, TextGenPool

class MyDataPool(TextGenPool):
    @classmethod
    def prepare(cls, split: str, **args) -> 'TextGenPool':
        samples = []
        # 加载数据
        for ix, item in enumerate(data):
            sample = Sample(
                id=f"{split}_{ix}",
                prompt_or_input_text=item["input"],
                references=[item["target"]]
            )
            samples.append(sample)
        return cls(samples)

# 注册
from rl4lms.envs.text_generation.registry import DataPoolRegistry
DataPoolRegistry.add("my_dataset", MyDataPool)
```

### 8.2 添加新奖励函数

```python
from rl4lms.envs.text_generation.reward import RewardFunction

class MyRewardFunction(RewardFunction):
    def __call__(self, prev_obs, action, current_obs, done, meta_info):
        if done:
            # 计算最终奖励
            reward = compute_reward(current_obs.context_text, 
                                    current_obs.target_or_reference_texts)
            return reward
        return 0  # 中间步骤奖励为0

# 注册
from rl4lms.envs.text_generation.registry import RewardFunctionRegistry
RewardFunctionRegistry.add("my_reward", MyRewardFunction)
```

### 8.3 添加新指标

```python
from rl4lms.envs.text_generation.metric import BaseMetric

class MyMetric(BaseMetric):
    def compute(self, prompts, gen_texts, ref_texts, meta_infos, model, split):
        # 计算指标
        scores = [compute_score(gen, refs) for gen, refs in zip(gen_texts, ref_texts)]
        corpus_score = sum(scores) / len(scores)
        return {"my_metric": (scores, corpus_score)}

# 注册
from rl4lms.envs.text_generation.registry import MetricRegistry
MetricRegistry.add("my_metric", MyMetric)
```

---

## 9. 重要类和方法速查

### 9.1 Observation

```python
# 创建
obs = Observation.init_from_sample(sample, tokenizer, max_input, max_context, 
                                    prompt_truncation_side, context_start_token)

# 更新
new_obs = obs.update(action_token_id, tokenizer)

# 转字典
dict_obs = obs.to_dict()  # 用于 SB3
```

### 9.2 TextGenEnv

```python
# 初始化
env = TextGenEnv(tokenizer, reward_fn, samples, max_episode_length, 
                 priority_scale, max_prompt_length, terminate_on_eos)

# 标准 Gym 接口
obs = env.reset(sample)  # 或 env.reset() 随机采样
obs, reward, done, info = env.step(action)
```

### 9.3 Policy

```python
# 生成
gen_output = policy.generate(tokenizer, texts, max_prompt_length, gen_kwargs)
# 返回: gen_texts, gen_tokens, step_wise_logprobs, step_wise_actions

# 训练前向
actions, log_prob, entropy, outputs, past_kwargs = policy.forward_policy(
    obs, actions, past_model_kwargs
)

# Value 估计
values, past_kwargs = policy.forward_value(obs, past_model_kwargs)

# 评估（训练）
values, log_prob, entropy = policy.evaluate_actions(obs, actions)
```

### 9.4 Trainer

```python
# 构建
trainer = OnPolicyTrainer(
    tokenizer_config, datapool_config, reward_config,
    env_config, on_policy_alg_config, train_eval_config, tracker
)

# 训练
trainer.train_and_eval()
```

---

## 10. 依赖库版本

```
torch==1.11.0
transformers==4.18.0
stable-baselines3==1.5.1a5
gym==0.21.0
datasets==2.5.1
bert-score==0.3.11
wandb==0.12.15
```

---

## 11. 参考资料

- **论文**: [Is Reinforcement Learning (Not) for Natural Language Processing?](https://arxiv.org/abs/2210.01241)
- **项目主页**: https://rl4lms.apps.allenai.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
