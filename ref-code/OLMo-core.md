# OLMo-core 代码结构映射 (Code Map)

## 项目概述

**OLMo-core** 是 AI2 开发的用于 OLMo 系列语言模型训练的核心库。它提供了模块化的组件用于 Transformer 架构、分布式训练、数据加载和评估。

- **仓库地址**: https://github.com/allenai/OLMo-core
- **版本**: v1.9.0
- **Python 包名**: `ai2-olmo-core`
- **代码文件总数**: 174+ Python 文件

---

## 目录结构

```
OLMo-core/
├── docs/                          # 文档 (Sphinx 自动生成)
├── src/
│   ├── olmo_core/                # 核心库代码
│   │   ├── data/                 # 数据加载模块
│   │   ├── distributed/          # 分布式训练
│   │   ├── eval/                 # 评估工具
│   │   ├── float8/               # FP8 训练支持
│   │   ├── generate/             # 文本生成
│   │   ├── internal/             # 内部工具
│   │   ├── kernels/              # CUDA 内核
│   │   ├── launch/               # Beaker 启动工具
│   │   ├── model_ladder/         # 模型规模实验
│   │   ├── nn/                   # 神经网络组件
│   │   ├── ops/                  # 自定义算子
│   │   ├── optim/                # 优化器和学习率调度
│   │   ├── testing/              # 测试工具
│   │   ├── train/                # 训练循环
│   │   ├── aliases.py            # 类型别名
│   │   ├── config.py             # 配置基类
│   │   ├── doc_utils.py          # 文档工具
│   │   ├── exceptions.py         # 异常定义
│   │   ├── fs_cache.py           # 文件系统缓存
│   │   ├── io.py                 # IO 工具
│   │   ├── script_utils.py       # 脚本工具
│   │   ├── utils.py              # 通用工具
│   │   └── version.py            # 版本信息
│   ├── examples/                 # 示例代码
│   ├── scripts/                  # 训练脚本
│   │   ├── official/             # 官方模型训练脚本
│   │   ├── train/                # 训练脚本模板
│   │   └── ...
│   └── test/                     # 测试代码
└── ...
```

---

## 核心模块详解

### 1. 配置系统 (`src/olmo_core/config.py`)

**核心设计理念**: 一切通过 dataclass 配置

| 关键类/函数 | 作用 |
|------------|------|
| `Config` | 配置基类，支持 YAML/JSON 序列化、命令行覆盖 (dot notation)、merge 操作 |
| `StrEnum` | 字符串枚举，兼容 Python 3.11+ |
| `DType` | 支持的数据类型枚举 (float32, bfloat16, float16, float8) |
| `UNSET` | 未设置值的标记 |

**特性**:
- 配置类支持 `Registrable` mixin，实现多态配置（通过 `type` 字段解析子类）
- 嵌套配置支持模块化组合
- 命令行覆盖: `--train_module.optim.lr=6e-3`

---

### 2. 训练流水线 (`src/olmo_core/train/`)

#### 2.1 核心训练器

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `trainer.py` | `Trainer` / `TrainerConfig` | 核心训练循环，支持 checkpoint、评估、回调系统 |
| `common.py` | `TrainingProgress`, `Duration` | 训练状态和持续时间定义 |
| `checkpoint.py` | `Checkpointer` | 分布式 checkpoint 管理 |
| `utils.py` | `reduce_metrics`, `move_metrics` | 训练工具函数 |

#### 2.2 训练模块 (`train_module/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `train_module.py` | `TrainModule`, `BasicTrainModule` | 训练模块基类，封装模型和优化器 |
| `transformer/train_module.py` | `TransformerTrainModule` | Transformer 专用训练模块 |
| `transformer/pipeline_train_module.py` | `TransformerPipelineTrainModule` | Pipeline Parallel 训练模块 |
| `transformer/config.py` | `TransformerTrainModuleConfig` | Transformer 训练配置 |

**关键配置**:
- `TransformerDataParallelConfig` - 数据并行 (FSDP/HSDP/DDP)
- `TransformerTensorParallelConfig` - 张量并行 (TP)
- `TransformerPipelineParallelConfig` - 流水线并行 (PP)
- `TransformerContextParallelConfig` - 上下文并行 (CP/Rring Attention)
- `TransformerExpertParallelConfig` - 专家并行 (EP)

#### 2.3 回调系统 (`callbacks/`)

| 回调类 | 功能 |
|--------|------|
| `Callback` / `CallbackConfig` | 回调基类 |
| `CheckpointerCallback` | Checkpoint 保存策略 |
| `ConsoleLoggerCallback` | 控制台日志输出 |
| `EvaluatorCallback` | 评估执行 |
| `SpeedMonitorCallback` | 训练速度监控 |
| `WandBCallback` / `CometCallback` | 实验追踪 |
| `SlackNotifierCallback` | Slack 通知 |
| `BeakerCallback` | Beaker 平台集成 |
| `ModelMergeCallback` | 模型合并 (Model Soups) |
| `SequenceLengthSchedulerCallback` | 序列长度调度 |
| `BatchSizeSchedulerCallback` | 批量大小调度 |
| `StabilityMonitorCallback` | 训练稳定性监控 |

---

### 3. 神经网络架构 (`src/olmo_core/nn/`)

#### 3.1 Transformer 核心 (`transformer/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `config.py` | `TransformerConfig` | Transformer 配置，含工厂方法 `olmo2_32B()` 等 |
| `model.py` | `Transformer`, `NormalizedTransformer`, `MoETransformer` | Transformer 模型实现 |
| `block.py` | `TransformerBlock`, `MoETransformerBlock`, `NormalizedTransformerBlock` | Transformer Block 变体 |
| `init.py` | `InitMethod` | 初始化方法枚举 |

**支持的 Block 类型**:
- `default` - 标准 Transformer Block
- `reordered_norm` - Norm 重排序 Block
- `normalized` - nGPT 风格归一化 Block
- `moe` - MoE Block
- `moe_hybrid` - 混合 MoE Block
- `peri_norm` - Peri-Norm Block

#### 3.2 注意力机制 (`attention/`)

| 文件 | 关键类/函数 | 作用 |
|------|-------------|------|
| `__init__.py` | `Attention`, `AttentionConfig` | 注意力机制主类 |
| `backend.py` | `AttentionBackend`, `FlashAttention2Backend`, `TEAttentionBackend` | 注意力后端抽象 |
| `flash_attn_api.py` | Flash Attention 封装 | Flash Attention 2/3/4 API |
| `ring.py` | `RingAttentionLoadBalancer`, `UlyssesLoadBalancer` | Ring Attention 负载均衡 |
| `kv_cache.py` | `KVCacheManager` | KV Cache 管理 |
| `recurrent.py` | `GatedDeltaNet` | 循环注意力 (Gated DeltaNet) |

**支持的注意力后端**:
- `torch` - PyTorch 原生实现
- `flash_attention_2` - Flash Attention 2
- `flash_attention_3` - Flash Attention 3
- `flash_attention_4` - Flash Attention 4
- `transformer_engine` - NVIDIA Transformer Engine

#### 3.3 MoE (混合专家) (`moe/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `moe.py` | `MoEConfig`, `MoELayer` | MoE 层实现 |
| `router.py` | `MoERouterConfig` | 专家路由 |
| `mlp.py` | `MoEMLP` | 专家 MLP |
| `parallel_mlp.py` | `ParallelMLP` | 并行 MLP (专家并行) |
| `loss.py` | `MoELoadBalancingLoss` | 负载均衡损失 |

#### 3.4 其他神经网络组件

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `layer_norm.py` | `LayerNorm`, `RMSNorm`, `L2Norm` | 归一化层变体 |
| `feed_forward.py` | `FeedForwardConfig`, `SwiGLU`, `GeGLU` | FFN 变体 |
| `rope.py` | `RoPEConfig`, `RotaryEmbedding` | RoPE 位置编码 |
| `lm_head.py` | `LMHeadConfig` | 语言模型头部 |
| `cross_entropy_loss.py` | `CrossEntropyLoss` | 交叉熵损失 |

---

### 4. 数据加载 (`src/olmo_core/data/`)

#### 4.1 可组合数据加载 API (`composable/`)

**推荐使用的数据加载架构**:
```
TokenSource -> InstanceSource -> ComposableDataLoader
```

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `__init__.py` | 文档和示例 | 数据加载流程说明 |
| `source_abc.py` | `Source` | 数据源抽象基类 |
| `token_source.py` | `TokenSource` | Token 级数据源 |
| `instance_source.py` | `InstanceSource` | Instance 级数据源 |
| `data_loader.py` | `ComposableDataLoader` | 可组合数据加载器 |
| `mixing_*.py` | `MixingTokenSource`, `MixingInstanceSource` | 数据源混合 |
| `sampling_*.py` | `SamplingTokenSource`, `SamplingInstanceSource` | 采样数据源 |
| `packing_instance_source.py` | `PackingInstanceSource` | Packing 数据源 |

#### 4.2 传统数据加载

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `numpy_dataset.py` | `NumpyFSLDataset`, `NumpyVSLDataset` | NumPy 数据集 |
| `data_loader.py` | `NumpyDataLoaderConfig` | 数据加载器配置 |
| `collator.py` | `DataCollator` | 数据 Collator |
| `tokenizer.py` | `TokenizerConfig` | Tokenizer 配置 |
| `mixes/` | 预定义数据混合配置 | Dolma17, OLMoE-mix 等 |

#### 4.3 数据源混合 (`source_mixtures/`)

包含 AI2 内部数据源混合配置文件。

---

### 5. 分布式训练 (`src/olmo_core/distributed/`)

#### 5.1 并行策略 (`parallel/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `__init__.py` | `DataParallel`, `TensorParallel`, `PipelineParallel` | 并行策略主类 |
| `data_parallel.py` | FSDP/HSDP/DDP 配置 | 数据并行策略 |
| `tensor_parallel.py` | `TensorParallel` | 张量并行 |
| `pipeline_parallel.py` | `PipelineParallel` | 流水线并行 |
| `context_parallel.py` | `ContextParallel` | 上下文并行 (Ring Attention) |
| `expert_parallel.py` | `ExpertParallel` | 专家并行 |

#### 5.2 Checkpoint (`checkpoint/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `__init__.py` | `load_state_dict`, `save_state_dict` | Checkpoint 加载/保存 |
| `filesystem.py` | 文件系统后端 | 支持本地/云存储 |

#### 5.3 分布式工具 (`utils.py`)

分布式通信和工具函数:
- `all_reduce_value`, `all_gather`
- `get_world_size`, `get_rank`, `get_local_rank`
- `barrier`, `broadcast_object`

---

### 6. 优化器 (`src/olmo_core/optim/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `config.py` | `OptimConfig`, `MatrixAwareOptimConfig` | 优化器配置基类 |
| `adamw.py` | `AdamWConfig`, `SkipStepAdamW` | AdamW 及变体 |
| `lion.py` | `LionConfig`, `SkipStepLion` | Lion 优化器 |
| `muon.py` | `MuonConfig` | Muon 优化器 |
| `dion.py` | `DionConfig` | Dion 优化器 |
| `scheduler.py` | `CosWithWarmup`, `WSD`, `WSDS` | 学习率调度器 |
| `skip_step_optimizer.py` | `SkipStepOptimizer` | 跳过不稳定 step 的包装器 |

**支持的调度器**:
- `ConstantScheduler`
- `ConstantWithWarmup`
- `CosWithWarmup`
- `CosWithWarmupAndLinearDecay`
- `LinearWithWarmup`
- `InvSqrtWithWarmup`
- `ExponentialScheduler`
- `WSD` (Warmup-Stable-Decay)
- `WSDS` (WSD with S-curve)

---

### 7. 生成/推理 (`src/olmo_core/generate/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `chat.py` | Chat 循环实现 | 交互式聊天 demo |
| `sampling.py` | 采样策略 | Temperature, Top-p 等 |
| `generation_module/transformer/` | `TransformerGenerator` | Transformer 生成模块 |

---

### 8. 评估 (`src/olmo_core/eval/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `evaluator.py` | `Evaluator` | 评估器基类 |
| `lm_evaluator.py` | `LMEvaluator` | 语言模型评估 |
| `metrics.py` | 评估指标 | Perplexity 等 |
| `task_groups.py` | 任务分组 | 下游任务分组 |

---

### 9. 启动工具 (`src/olmo_core/launch/`)

| 文件 | 关键类 | 作用 |
|------|--------|------|
| `beaker.py` | `BeakerLaunchConfig`, `OLMoCoreBeakerImage` | Beaker 平台启动配置 |
| `utils.py` | `prepare_cli_environment` | CLI 环境准备 |

**启动流程**:
1. 创建 Gantry recipe
2. 从预构建镜像启动容器
3. 克隆 git 仓库
4. 安装 OLMo-core
5. 运行训练命令

---

### 10. FP8 支持 (`src/olmo_core/float8/`)

| 文件 | 作用 |
|------|------|
| `ao.py` | torchao FP8 训练集成 |
| `__init__.py` | FP8 配置和工具 |

---

## 模块依赖关系

```
olmo_core/
├── config.py (基类，被所有模块依赖)
├── utils.py (通用工具)
├── io.py (IO 操作)
│
├── nn/ (神经网络组件)
│   ├── transformer/ (依赖 attention/, feed_forward/, layer_norm/, rope/)
│   ├── attention/ (依赖 backend/, rope/)
│   ├── moe/ (依赖 router/, mlp/, parallel_mlp/)
│   └── ...
│
├── data/ (数据加载)
│   ├── composable/ (推荐 API)
│   └── numpy_dataset.py (传统 API)
│
├── optim/ (优化器)
│   └── scheduler.py (学习率调度)
│
├── distributed/ (分布式)
│   ├── parallel/ (并行策略)
│   └── checkpoint/ (Checkpoint)
│
└── train/ (训练)
    ├── train_module/
    │   └── transformer/ (依赖 nn/, distributed/)
    ├── callbacks/ (依赖 train_module/)
    └── trainer.py (依赖 train_module/, callbacks/, data/)
```

---

## 训练脚本结构

### 官方训练脚本 (`src/scripts/official/`)

```
official/
├── OLMo2/              # OLMo-2 官方模型
│   ├── OLMo-2-0325-32B-train.py   # 32B 模型预训练
│   └── OLMo-2-0325-32B-anneal.py  # 32B 模型退火
│
└── OLMo3/              # OLMo-3 官方模型
    ├── OLMo-3-1025-7B-pretrain-*.py
    ├── OLMo-3-1025-32B-pretrain.py
    └── OLMo-3-1025-*-midtrain.py
```

**启动方式**:
```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints \
  --train_module.optim.lr=6e-3
```

### 实验脚本 (`src/scripts/train/`)

```
train/
├── OLMo2/              # OLMo-2 实验配置
├── OLMo3/              # OLMo-3 实验配置
├── sft/                # SFT 训练脚本
├── ladder/             # 模型规模实验 (Scaling Laws)
│   └── 2026Q1/         # 2026 Q1 实验
└── template.py         # 训练脚本模板
```

---

## 示例代码 (`src/examples/`)

| 目录 | 内容 |
|------|------|
| `llm/` | LLM 训练示例 |
| `moe/` | MoE 模型示例 |
| `ngpt/` | nGPT (归一化 Transformer) 示例 |
| `huggingface/` | Hugging Face 集成示例 |
| `llama_lns/` | LLaMA LayerNorm Scaling 示例 |

---

## 重要工具脚本 (`src/scripts/`)

| 脚本 | 功能 |
|------|------|
| `unshard.py` | 合并分片 checkpoint |
| `merge_core_checkpoints.py` | 合并 Core checkpoint |
| `merge_hf_checkpoints.py` | 合并 HuggingFace checkpoint |
| `reshard_core_checkpoint.py` | 重新分片 checkpoint |
| `dump_training_batch.py` | 导出训练 batch |

---

## 测试结构 (`src/test/`)

测试结构与源代码结构镜像对应:

```
test/
├── data/               # 数据模块测试
├── distributed/        # 分布式测试
│   ├── checkpoint/     # Checkpoint 测试
│   └── parallel/       # 并行策略测试
├── nn/                 # NN 模块测试
│   ├── attention/      # 注意力测试
│   ├── transformer/    # Transformer 测试
│   └── ...
├── optim/              # 优化器测试
├── train/              # 训练模块测试
└── ...
```

---

## 配置系统详解

### 配置继承关系

```
Config (基类)
├── ExperimentConfig
│   ├── model: TransformerConfig
│   ├── dataset: NumpyDatasetConfig
│   ├── data_loader: NumpyDataLoaderConfig
│   ├── train_module: TransformerTrainModuleConfig
│   └── trainer: TrainerConfig
│
├── TransformerConfig
│   ├── block: TransformerBlockConfig
│   ├── attention: AttentionConfig
│   ├── feed_forward: FeedForwardConfig
│   ├── layer_norm: LayerNormConfig
│   └── rope: RoPEConfig
│
├── TransformerTrainModuleConfig
│   ├── dp_config: TransformerDataParallelConfig
│   ├── tp_config: TransformerTensorParallelConfig
│   ├── pp_config: TransformerPipelineParallelConfig
│   ├── cp_config: TransformerContextParallelConfig
│   ├── ep_config: TransformerExpertParallelConfig
│   ├── optim: OptimConfig
│   └── scheduler: Scheduler
│
└── TrainerConfig
    ├── callbacks: List[CallbackConfig]
    └── checkpointer: CheckpointerConfig
```

### 配置覆盖示例

```python
# 命令行覆盖
--train_module.optim.lr=6e-3
--model.n_layers=32
--trainer.max_steps=100000

# Python 代码覆盖
config = config.merge(["train_module.optim.lr=6e-3"])
```

---

## 关键设计模式

### 1. Registrable 多态配置

```python
@dataclass
class OptimConfig(Config, Registrable):
    lr: float

@dataclass
class AdamWConfig(OptimConfig):
    betas: Tuple[float, float] = (0.9, 0.999)
    
# 注册
OptimConfig.register("adamw", AdamWConfig)

# 运行时解析
config = OptimConfig.from_dict({"type": "adamw", "lr": 1e-4})
```

### 2. Builder 模式

```python
config = TransformerConfig.olmo2_7B()  # 工厂方法
model = config.build(init_device="meta")  # 构建模型
train_module = train_module_config.build(model)  # 构建训练模块
trainer = trainer_config.build(train_module, data_loader)  # 构建训练器
```

### 3. 回调系统

```python
class MyCallback(Callback):
    def pre_step(self, trainer: Trainer, ...):
        pass
    
    def post_step(self, trainer: Trainer, ...):
        pass
    
    def pre_train(self, trainer: Trainer):
        pass
```

---

## 依赖的第三方库

### 核心依赖
- **PyTorch**: 深度学习框架
- **dataclass-extensions**: 提供 `Registrable` mixin
- **cached-path**: 文件缓存和下载
- **rich**: 美观的控制台输出
- **pyyaml**: YAML 配置支持

### 可选依赖
- **flash-attn**: Flash Attention 2/3/4
- **ring-flash-attn**: Ring Attention (上下文并行)
- **transformer-engine**: NVIDIA FP8 训练
- **torchao**: PyTorch AO (Float8 训练)
- **liger-kernel**: 低内存 fused loss
- **grouped_gemm**: Dropless MoE
- **vllm**: 推理加速

---

## 总结

OLMo-core 是一个设计精良的大规模语言模型训练框架，其核心特点包括:

1. **模块化架构**: 清晰的模块划分和依赖关系
2. **灵活配置系统**: 基于 dataclass 的配置，支持命令行覆盖
3. **多维度并行**: 支持 DP/TP/PP/CP/EP 任意组合
4. **多种注意力后端**: Flash Attention, TransformerEngine 等
5. **丰富的模型变体**: Standard, Normalized (nGPT), MoE
6. **可组合数据加载**: 灵活的数据混合和采样策略
7. **完整的训练工具**: Checkpoint, 回调, 监控, 评估

该框架特别适合用于 7B-100B+ 规模的语言模型预训练和微调。
