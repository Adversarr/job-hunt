# OlmOCR 代码地图 (Code Map)

> **版本**: v0.4.0  
> **项目**: Allen AI 开源 OCR 工具包  
> **定位**: 基于 7B 参数 VLM 的 PDF/图像文档转 Markdown 工具

---

## 1. 项目整体架构

### 1.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           olmOCR 系统架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────┐ │
│  │   输入层     │   │   处理层     │   │         输出层               │ │
│  │  (PDF/图片)  │──▶│  (VLM推理)   │──▶│    (Markdown/Dolma格式)      │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────────┘ │
│         │                  │                        │                  │
│         ▼                  ▼                        ▼                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────┐ │
│  │  数据流水线  │   │  训练系统    │   │       评测系统               │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 目录结构

```
olmocr/
├── olmocr/                          # 核心源代码
│   ├── pipeline.py                  # 主推理流水线 (1,526行)
│   ├── metrics.py                   # 指标统计
│   ├── s3_utils.py                  # S3/GCS/云存储工具 (409行)
│   ├── work_queue.py                # 分布式任务队列 (473行)
│   ├── check.py                     # 环境检查工具
│   ├── image_utils.py               # 图像处理工具
│   ├── repeatdetect.py              # 重复内容检测
│   ├── datatypes.py                 # 数据类型定义
│   ├── version.py                   # 版本信息
│   ├── bench/                       # 评测基准系统
│   │   ├── benchmark.py             # 评测主入口 (457行)
│   │   ├── tests.py                 # 测试定义 (1,109行)
│   │   ├── convert.py               # 格式转换
│   │   ├── report.py                # 报告生成
│   │   ├── utils.py                 # 工具函数
│   │   ├── prompts.py               # 评测提示词
│   │   ├── review_app.py            # 评测审核Web应用
│   │   ├── review_app_latex.py      # LaTeX评测审核
│   │   ├── runners/                 # 不同OCR引擎运行器
│   │   │   ├── run_olmocr_pipeline.py
│   │   │   ├── run_chatgpt.py
│   │   │   ├── run_claude.py
│   │   │   ├── run_gemini.py
│   │   │   ├── run_marker.py
│   │   │   ├── run_mineru.py
│   │   │   └── ... (共15+个运行器)
│   │   ├── miners/                  # 测试数据挖掘脚本 (24个)
│   │   │   ├── mine_tables_gpt.py
│   │   │   ├── mine_math.py
│   │   │   ├── mine_old_scans.py
│   │   │   └── ...
│   │   ├── katex/                   # KaTeX渲染支持
│   │   ├── scripts/                 # 辅助脚本
│   │   ├── synth/                   # 合成数据生成
│   │   └── templates/               # 评测模板
│   ├── data/                        # 数据处理模块
│   │   ├── buildsilver.py           # 银标数据生成 (250行)
│   │   ├── renderpdf.py             # PDF渲染 (129行)
│   │   ├── prepare_olmocrmix.py     # 数据集准备
│   │   ├── clean_olmocrmix.py       # 数据清洗
│   │   └── ... (共13个处理脚本)
│   ├── filter/                      # 内容过滤
│   │   ├── filter.py                # PDF过滤器 (203行)
│   │   └── coherency.py             # 文本连贯性检测
│   ├── prompts/                     # 提示词工程
│   │   ├── prompts.py               # 主提示词定义 (183行)
│   │   └── anchor.py                # Anchor文本生成 (359行)
│   ├── train/                       # 训练系统
│   │   ├── train.py                 # SFT微调 (732行)
│   │   ├── grpo_train.py            # GRPO强化学习 (1,067行)
│   │   ├── dataloader.py            # 数据加载器 (1,747行)
│   │   ├── config.py                # 训练配置 (507行)
│   │   ├── muon.py                  # Muon优化器
│   │   └── ... (配置和工具脚本)
│   └── viewer/                      # 可视化工具
│       └── dolmaviewer.py           # Dolma文档查看器
├── scripts/                         # 运维脚本
│   ├── data/                        # 数据处理脚本
│   ├── eval/                        # 评测脚本
│   ├── pii/                         # PII检测脚本
│   └── train/                       # 训练提交脚本
├── tests/                           # 单元测试
├── docs/                            # 文档
├── Dockerfile                       # Docker镜像定义
└── pyproject.toml                   # 项目配置
```

---

## 2. 核心模块详解

### 2.1 主推理流水线 (olmocr/pipeline.py)

**功能**: PDF/图像 → Markdown 的核心转换流程

**关键组件**:

| 类/函数 | 行号 | 功能描述 |
|---------|------|----------|
| `PageResult` | 94 | 单页OCR结果数据类 |
| `build_page_query()` | 106 | 构建VLM查询请求 |
| `try_single_page()` | 149 | 单页OCR推理(重试机制) |
| `process_pdf()` | 800+ | PDF整体处理流程 |
| `vllm_server_task()` | 1000+ | vLLM服务管理 |
| `main()` | 1200+ | 命令行入口 |

**核心流程**:
```python
PDF输入 → 渲染为图像 → 构建Prompt → VLM推理 → 解析输出 → Markdown输出
   │           │              │            │           │
   │           │              │            │           └── olmocr/prompts/prompts.py
   │           │              │            └── vLLM/OpenAI API
   │           │              └── olmocr/prompts/prompts.py
   │           └── olmocr/data/renderpdf.py
   └── olmocr/filter/filter.py (可选过滤)
```

**关键特性**:
- 支持本地/远程 vLLM 服务
- 支持 S3 分布式处理
- 温度退火重试策略 (TEMPERATURE_BY_ATTEMPT)
- 工作队列并发控制
- 页面组批量处理

### 2.2 提示词工程 (olmocr/prompts/)

#### prompts.py

**核心数据结构**:
```python
@dataclass(frozen=True)
class PageResponse:
    primary_language: Optional[str]      # 主语言
    is_rotation_valid: bool              # 旋转是否有效
    rotation_correction: int             # 旋转角度 (0/90/180/270)
    is_table: bool                       # 是否表格页
    is_diagram: bool                     # 是否图表页
    natural_text: Optional[str]          # 提取的自然文本
```

**主要提示词函数**:

| 函数 | 用途 |
|------|------|
| `build_openai_silver_data_prompt()` | GPT-4o银标数据生成 |
| `build_no_anchoring_v4_yaml_prompt()` | 无Anchor推理提示词 |
| `build_finetuning_prompt()` | 微调训练提示词 |
| `openai_response_format_schema()` | 结构化输出Schema |

#### anchor.py

**Anchor文本生成引擎**:

```python
def get_anchor_text(
    local_pdf_path: str,
    page: int,
    pdf_engine: Literal["pdftotext", "pdfium", "pypdf", "topcoherency", "pdfreport"],
    target_length: int = 4000
) -> str
```

**PDF引擎对比**:

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| `pdftotext` | 速度快，纯文本 | 快速预览 |
| `pdfium` | 位置信息准确 | 需要坐标时 |
| `pypdf` | Python原生 | 简单提取 |
| `topcoherency` | 自动选择最优 | 默认推荐 |
| `pdfreport` | 结构化报告 | 训练数据生成 |

### 2.3 评测系统 (olmocr/bench/)

#### benchmark.py

**核心函数**:

```python
def evaluate_candidate(
    candidate_folder: str,
    all_tests: List[BasePDFTest],
    pdf_basenames: List[str],
    force: bool = False
) -> Tuple[float, int, List[str], List[str], Dict, List[float], Dict]
```

**返回结果**:
- overall_score: 总体得分
- total_tests: 测试总数
- candidate_errors: 候选错误列表
- test_failures: 测试失败详情
- test_type_breakdown: 按类型分组得分
- all_test_scores: 所有测试分数（用于Bootstrap CI）
- test_results: 详细测试结果

#### tests.py

**测试类型定义** (TestType Enum):

```python
class TestType(str, Enum):
    BASELINE = "baseline"      # 基线测试
    PRESENT = "present"        # 内容存在测试
    ABSENT = "absent"          # 内容缺失测试
    ORDER = "order"            # 顺序测试
    TABLE = "table"            # 表格测试
    REPEAT = "repeat"          # 重复检测测试
    LATEX = "latex"            # LaTeX公式测试
```

**核心测试类**:

| 类名 | 功能 | 测试内容 |
|------|------|----------|
| `BaselineTest` | 基线测试 | 基础文本提取质量 |
| `PresentTest` | 存在测试 | 特定文本/元素是否存在 |
| `AbsentTest` | 缺失测试 | 确保某些内容不出现 |
| `OrderTest` | 顺序测试 | 文本阅读顺序正确性 |
| `TableTest` | 表格测试 | 表格结构和内容 |
| `RepeatTest` | 重复测试 | 检测重复/幻觉内容 |
| `LatexTest` | LaTeX测试 | 数学公式准确性 |

**表格解析**:
```python
def parse_html_tables(html_content: str) -> List[TableData]:
    """解析HTML表格，处理 colspan/rowspan"""
    
@dataclass
class TableData:
    data: np.ndarray              # 表格数据
    header_rows: Set[int]         # 表头行索引
    header_cols: Set[int]         # 表头列索引
    col_headers: dict             # 列头映射
    row_headers: dict             # 行头映射
```

### 2.4 数据流水线 (olmocr/data/)

#### buildsilver.py - 银标数据生成

**核心流程**:
```python
PDF输入 → 过滤 → 页面采样 → 渲染图像 → GPT-4o标注 → 结构化输出
```

**关键配置**:
```python
TARGET_IMAGE_DIM = 2048  # 渲染图像长边尺寸
first_n_pages = 3        # 始终采样前N页
max_sample_pages = 5     # 最大采样页数
```

**OpenAI Batch API 优化技巧**:
1. 使用 Batch API（价格减半）
2. 使用结构化输出（Structured Outputs）
3. 利用 Schema 字段顺序引导推理
4. 请求 logprobs 用于后续质量筛选

#### renderpdf.py - PDF渲染

```python
def render_pdf_to_base64png(
    local_pdf_path: str,
    page_num: int,
    target_longest_image_dim: int = 2048
) -> str:
    """使用 pdftoppm 将PDF页面渲染为PNG"""

def get_pdf_media_box_width_height(
    local_pdf_path: str,
    page_num: int
) -> tuple[float, float]:
    """使用 pdfinfo 获取页面尺寸"""
```

### 2.5 训练系统 (olmocr/train/)

#### train.py - SFT微调

**核心组件**:

```python
class QwenDataCollator:
    """处理Vision-Language模型的数据对齐"""
    # 支持字段: input_ids, attention_mask, labels, pixel_values, image_grid_thw

def prepare_lora_model(model, model_cfg) -> torch.nn.Module:
    """使用PEFT包装LoRA适配器"""
    # 支持 rank, alpha, dropout, target_modules 配置

def save_checkpoint(...):
    """保存模型、优化器、学习率调度器状态"""
```

**训练流程**:
```
配置加载 → 模型初始化 → LoRA适配 → 数据加载 → 训练循环 → 检查点保存
    │              │          │          │         │          │
    │              │          │          │         │          └── 每N步保存
    │              │          │          │         └── autocast混合精度
    │              │          │          └── QwenDataCollator
    │              │          └── get_peft_model
    │              └── Qwen2_5_VLForConditionalGeneration
    └── OmegaConf YAML配置
```

#### grpo_train.py - GRPO强化学习

**数据集类**:
```python
class OlmOCRBenchDataset(Dataset):
    """从olmocr-bench格式加载PDF页面"""
    # 加载 claude_original 作为参考答案
    # 支持多页PDF采样
```

**奖励函数**:
```python
def compute_olmocr_rewards(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    bench_data_folder: str
) -> List[float]:
    """计算基于单元测试的奖励分数"""
    # 使用 fuzzy matching 比较输出
    # 基于 olmocr-bench 测试规则评分
```

**GRPO配置**:
```python
GRPOConfig(
    output_dir=output_dir,
    num_generations=8,              # 每组样本数
    per_device_train_batch_size=1,  # 每设备批次
    gradient_accumulation_steps=8,  # 梯度累积
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    bf16=True,
    max_prompt_length=8192,
    max_completion_length=4096,
)
```

#### dataloader.py - 数据加载器

**Pipeline架构**:
```python
@dataclass(frozen=True, slots=True)
class PipelineStep(ABC):
    """流水线步骤抽象基类"""
    @abstractmethod
    def __call__(self, sample: Sample) -> Optional[Sample]:
        """处理样本，返回None表示跳过"""
```

**标准Pipeline步骤**:

| 步骤类 | 功能 |
|--------|------|
| `FrontMatterParser` | 解析前置元数据 |
| `PDFRenderer` | 渲染PDF为图像 |
| `StaticLengthDocumentAnchoring` | 生成定长Anchor文本 |
| `FinetuningPrompt` | 构建微调提示词 |
| `FrontMatterOutputFormat` | 格式化输出 |
| `LatexBracketNormalizer` | LaTeX括号规范化 |

**数据集类**:
```python
class BaseMarkdownPDFDataset(Dataset):
    """Markdown-PDF配对数据集基类"""
    # 支持路径验证、符号链接解析、页面数检查

class MarkdownPDFDataset(BaseMarkdownPDFDataset):
    """标准训练数据集"""
    # 支持多进程验证、缓存、增量加载
```

#### config.py - 训练配置

**配置层次结构**:
```python
@dataclass
class Config:
    model: ModelConfig              # 模型配置
    datasets: List[DatasetConfig]   # 数据集配置
    training: TrainingConfig        # 训练超参
    data_processing: DataProcessingConfig  # 数据处理
    wandb: Optional[WandbConfig]    # 实验追踪
```

### 2.6 内容过滤 (olmocr/filter/)

#### filter.py

```python
class PdfFilter:
    """PDF内容过滤器"""
    
    def filter_out_pdf(self, local_pdf_path: str) -> bool:
        """返回True表示应该过滤掉该PDF"""
        # 检查1: 是否为表单
        # 检查2: 是否SEO垃圾内容
        # 检查3: 语言检测 (Lingua)
        # 检查4: 文本连贯性
```

**过滤规则**:

| 检查项 | 阈值/条件 | 说明 |
|--------|----------|------|
| Form检测 | `get_form_text_fields()` | PDF内置表单 |
| SEO垃圾词 | >0.4%词频 | download, free, casino等 |
| 语言过滤 | 非目标语言 | 使用Lingua检测 |
| 字母比例 | <50% | 防止扫描件OCR质量差 |
| 文本长度 | <200字符 | 保留短文档 |

#### coherency.py

```python
def get_document_coherency(text: str) -> float:
    """计算文档连贯性分数"""
    # 基于句子分割和语义连贯性
    # 返回0-1之间的分数，越高表示越连贯
```

### 2.7 分布式基础设施

#### work_queue.py

**架构设计**:
```
┌─────────────────────────────────────────────────────────┐
│                     WorkQueue                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Local     │  │     S3      │  │   Future Backends│  │
│  │   Backend   │  │   Backend   │  │                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│         │                │                               │
│         └────────────────┴──▶ Backend (Abstract)         │
│                                   - load_index_lines()   │
│                                   - save_index_lines()   │
│                                   - get_completed_hashes()│
│                                   - create_worker_lock() │
└─────────────────────────────────────────────────────────┘
```

**关键机制**:
- **工作锁(Worker Lock)**: 防止多节点重复处理 (默认超时30分钟)
- **完成标记(Done Flag)**: 原子性完成状态标记
- **索引文件(Index)**: CSV格式存储工作项 (zstd压缩)

#### s3_utils.py

**核心功能**:

```python
# S3路径解析
def parse_s3_path(s3_path: str) -> tuple[str, str]:
    # 支持 s3://, gs://, weka://

# 通配符扩展
def expand_s3_glob(s3_client, s3_glob: str) -> dict[str, str]:
    # 返回 {s3_path: etag} 映射

# 流式下载
def get_s3_bytes(s3_client, s3_path: str, 
                 start_index: Optional[int] = None,
                 end_index: Optional[int] = None) -> bytes:

# zstd压缩CSV
def download_zstd_csv(...) -> List[Dict]
def upload_zstd_csv(...) -> None
```

---

## 3. 模块依赖关系

### 3.1 依赖图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         依赖关系图                                   │
└──────────────────────────────────────────────────────────────────────┘

pipeline.py (主入口)
    ├── prompts/prompts.py
    │   └── (数据类定义，无依赖)
    ├── prompts/anchor.py
    │   └── filter/coherency.py
    ├── data/renderpdf.py
    │   └── (系统调用: pdftoppm, pdfinfo)
    ├── filter/filter.py
    │   └── lingua (第三方)
    ├── s3_utils.py
    │   └── boto3, google-cloud-storage
    ├── work_queue.py
    │   └── s3_utils.py
    ├── metrics.py
    │   └── (标准库)
    └── train/dataloader.py (用于FrontMatterParser)

train/train.py (SFT训练)
    ├── train/config.py
    │   └── omegaconf
    ├── train/dataloader.py
    │   ├── data/renderpdf.py
    │   ├── prompts/prompts.py
    │   └── prompts/anchor.py
    └── train/muon.py

train/grpo_train.py (RL训练)
    ├── train/dataloader.py
    ├── bench/tests.py (加载单元测试)
    ├── data/renderpdf.py
    └── prompts/prompts.py

bench/benchmark.py (评测)
    ├── bench/tests.py
    ├── bench/report.py
    └── bench/utils.py

bench/tests.py
    ├── bench/katex/render.py (LaTeX渲染)
    └── repeatdetect.py
```

### 3.2 外部依赖

**核心依赖** (pyproject.toml):

| 包名 | 用途 | 关键模块 |
|------|------|----------|
| `torch>=2.7.0` | 深度学习框架 | train/ |
| `transformers==4.57.3` | 模型加载 | train/, pipeline |
| `vllm==0.11.2` | 推理加速 | pipeline |
| `trl` | RL训练 | train/grpo_train.py |
| `peft` | LoRA微调 | train/train.py |
| `pypdf>=5.2.0` | PDF解析 | data/, filter/ |
| `pypdfium2` | PDF渲染 | prompts/anchor.py |
| `boto3` | AWS S3 | s3_utils.py |
| `google-cloud-storage` | GCS | s3_utils.py |
| `lingua-language-detector` | 语言检测 | filter/filter.py |
| `Pillow` | 图像处理 | data/renderpdf.py |
| `beautifulsoup4` | HTML解析 | bench/tests.py |
| `rapidfuzz` | 模糊匹配 | bench/, grpo_train.py |
| `fuzzysearch` | 模糊搜索 | bench/tests.py |

---

## 4. 关键配置参数

### 4.1 Pipeline配置

```python
# 图像渲染
TARGET_IMAGE_DIM = 2048          # 渲染图像长边像素

# 温度退火策略 (8次重试)
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

# 并发控制
pdf_render_max_workers_limit = BoundedSemaphore(CPU_COUNT - 2)
max_concurrent_requests_limit = BoundedSemaphore(1)  # 动态设置

# 重试策略
max_page_retries = 3             # 单页最大重试
max_page_error_rate = 1/250      # 文档最大错误率
pages_per_group = 8              # 每工作组页数
```

### 4.2 训练配置

```yaml
# 模型配置 (v0.4.0)
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
  lora_rank: 64
  lora_alpha: 128
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# 训练超参
training:
  per_device_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  num_epochs: 3
  save_steps: 500
  
# 数据处理
data_processing:
  target_longest_image_dim: 1024
  target_anchor_text_len: 6000
```

### 4.3 评测配置

```python
# 测试类别权重
CATEGORY_WEIGHTS = {
    "arxiv": 1.0,
    "old_scans_math": 1.0,
    "tables": 1.0,
    "old_scans": 1.0,
    "headers_footers": 1.0,
    "multi_column": 1.0,
    "long_tiny_text": 1.0,
    "base": 1.0,
}

# Bootstrap置信区间
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# 重复检测阈值
REPEAT_DETECTOR_THRESHOLD = 0.7
```

---

## 5. 重要类和函数清单

### 5.1 核心类

| 类名 | 文件 | 职责 |
|------|------|------|
| `PageResponse` | prompts/prompts.py | OCR结果数据结构 |
| `PdfFilter` | filter/filter.py | PDF内容过滤 |
| `WorkQueue` | work_queue.py | 分布式任务队列 |
| `MetricsKeeper` | metrics.py | 性能指标统计 |
| `Backend` | work_queue.py | 存储后端抽象 |
| `LocalBackend` | work_queue.py | 本地文件存储 |
| `S3Backend` | work_queue.py | S3存储后端 |
| `PageResult` | pipeline.py | 单页处理结果 |
| `TableData` | bench/tests.py | 表格数据结构 |
| `BasePDFTest` | bench/tests.py | 测试基类 |
| `Config` | train/config.py | 训练配置 |
| `QwenDataCollator` | train/train.py | 数据对齐 |
| `BaseMarkdownPDFDataset` | train/dataloader.py | 数据集基类 |

### 5.2 关键函数

**推理流水线**:
- `build_page_query()` - 构建VLM查询
- `try_single_page()` - 单页推理(含重试)
- `process_pdf()` - PDF整体处理
- `render_pdf_to_base64png()` - PDF渲染

**数据处理**:
- `get_anchor_text()` - 生成Anchor文本
- `build_silver_data()` - 银标数据生成
- `filter_out_pdf()` - PDF过滤
- `parse_html_tables()` - HTML表格解析

**评测**:
- `evaluate_candidate()` - 评测候选系统
- `load_tests()` - 加载测试定义
- `run_test()` - 执行单个测试
- `generate_html_report()` - 生成评测报告

**训练**:
- `prepare_lora_model()` - LoRA模型准备
- `save_checkpoint()` - 保存检查点
- `compute_olmocr_rewards()` - GRPO奖励计算
- `build_finetuning_prompt()` - 构建训练提示词

---

## 6. 数据流分析

### 6.1 推理数据流

```
PDF文件路径
    │
    ▼
┌─────────────────┐
│   S3/本地加载    │◄── s3_utils.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PDF过滤       │◄── filter/filter.py (可选)
│ - 表单检测      │
│ - SEO垃圾检测   │
│ - 语言检测      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   页面渲染      │◄── data/renderpdf.py
│ - pdftoppm     │
│ - 尺寸调整      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Anchor生成    │◄── prompts/anchor.py (旧版)
│ - pdfium       │
│ - pdftotext    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prompt构建    │◄── prompts/prompts.py
│ - 系统提示词    │
│ - 图像编码      │
│ - 结构化输出Schema│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VLM推理       │◄── vLLM / OpenAI API
│ - 温度退火      │
│ - 重试机制      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   输出解析      │◄── prompts/prompts.py
│ - JSON解析      │
│ - PageResponse  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   结果存储      │
│ - Dolma格式     │
│ - Markdown格式  │
└─────────────────┘
```

### 6.2 训练数据流

```
Markdown-PDF配对
    │
    ▼
┌─────────────────┐
│   路径验证      │◄── dataloader.py
│ - PDF存在性     │
│ - 页数检查      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pipeline处理  │◄── dataloader.py
│ - 渲染图像      │
│ - 生成Anchor    │
│ - 构建Prompt    │
│ - 格式化输出    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   数据对齐      │◄── train.py (QwenDataCollator)
│ - padding      │
│ - tensor转换    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   模型训练      │◄── train.py / grpo_train.py
│ - SFT微调      │
│ - GRPO强化学习  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   检查点保存    │
│ - 模型权重      │
│ - 优化器状态    │
│ - 训练状态      │
└─────────────────┘
```

---

## 7. 扩展点与二次开发

### 7.1 添加新测试类型

在 `olmocr/bench/tests.py` 中:

```python
class MyCustomTest(BasePDFTest):
    type: TestType = TestType.CUSTOM
    
    def run(self, md_content: str) -> Tuple[bool, str]:
        # 实现测试逻辑
        passed = self.check_condition(md_content)
        return passed, "explanation"
```

### 7.2 添加新PDF引擎

在 `olmocr/prompts/anchor.py` 中:

```python
def get_anchor_text(..., pdf_engine: Literal[..., "myengine"]):
    if pdf_engine == "myengine":
        return _get_myengine(local_pdf_path, page)

def _get_myengine(local_pdf_path: str, page: int) -> str:
    # 实现提取逻辑
    pass
```

### 7.3 自定义Pipeline步骤

在 `olmocr/train/dataloader.py` 中:

```python
@dataclass(frozen=True, slots=True)
class MyCustomStep(PipelineStep):
    def __call__(self, sample: Sample) -> Optional[Sample]:
        # 处理逻辑
        return modified_sample
```

---

## 8. 典型使用场景

### 8.1 本地单PDF转换

```bash
python -m olmocr.pipeline ./workspace \
    --markdown \
    --pdfs sample.pdf \
    --model allenai/olmOCR-2-7B-1025-FP8
```

### 8.2 批量S3处理

```bash
# 启动主节点
python -m olmocr.pipeline s3://bucket/workspace \
    --pdfs s3://bucket/pdfs/*.pdf

# 启动工作节点
python -m olmocr.pipeline s3://bucket/workspace
```

### 8.3 自定义训练

```bash
python -m olmocr.train.train \
    --config configs/v0.4.0/qwen2.5-7b.yaml \
    --output_dir ./checkpoints
```

### 8.4 运行评测

```bash
python -m olmocr.bench.benchmark \
    --input_dir ./olmocr-bench \
    --candidates ./outputs/markdown
```

---

## 9. 性能优化要点

### 9.1 推理优化

1. **FlashInfer**: 安装后可显著加速vLLM推理
2. **FP8量化**: 默认使用FP8减少显存占用
3. **页面分组**: `--pages_per_group` 控制批处理大小
4. **并发控制**: `--workers` 和 `--max_concurrent_requests` 协调

### 9.2 训练优化

1. **LoRA**: 只训练适配器，大幅减少参数量
2. **梯度累积**: 模拟大批量训练
3. **混合精度**: `torch.amp.autocast` 自动启用
4. **Muon优化器**: 可选的内存高效优化器

### 9.3 数据处理优化

1. **zstd压缩**: S3传输使用zstd压缩减少带宽
2. **多进程渲染**: PDF渲染使用进程池并行
3. **缓存机制**: 数据集支持本地缓存

---

## 10. 调试与故障排查

### 10.1 日志级别

```python
# pipeline.py 日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 调整为INFO减少输出
```

### 10.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `too many open files` | 文件描述符限制 | `ulimit -n 65536` |
| vLLM启动失败 | 显存不足 | 降低 `--max_model_len` |
| S3访问失败 | 凭证配置 | 检查AWS_PROFILE环境变量 |
| 页面渲染失败 | PDF损坏 | 调整 `--max_page_retries` |
| 重复内容 | 模型幻觉 | 启用 `--guided_decoding` |

### 10.3 调试工具

```bash
# 检查环境
python -m olmocr.check

# 查看统计数据
python -m olmocr.pipeline ./workspace --stats

# 单步调试
python -m olmocr.pipeline ./workspace --workers 1 --pdfs test.pdf
```

---

## 11. 附录

### 11.1 文件大小统计

| 文件 | 行数 | 模块 |
|------|------|------|
| pipeline.py | 1,526 | 主推理 |
| dataloader.py | 1,747 | 数据加载 |
| tests.py | 1,109 | 评测测试 |
| grpo_train.py | 1,067 | RL训练 |
| train.py | 732 | SFT训练 |
| review_app_latex.py | 716 | 评测审核 |
| prepare_checkpoint.py | 603 | 检查点处理 |
| config.py | 507 | 训练配置 |
| benchmark.py | 457 | 评测主入口 |
| work_queue.py | 473 | 任务队列 |
| mine_html_templates.py | 1,597 | 合成数据 |

**总计**: ~16,687 行 Python 代码

### 11.2 版本演进

- **v0.1.x**: 初始发布，基础PDF转Markdown
- **v0.2.x**: 训练代码重构，支持LoRA微调
- **v0.3.x**: 自动旋转检测，减少幻觉
- **v0.4.x**: RL训练(GRPO)，合成数据增强

### 11.3 相关资源

- **论文**: arXiv:2502.18443 (v1), arXiv:2510.19817 (v2)
- **模型**: https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8
- **Demo**: https://olmocr.allenai.org/
- **Docker**: https://hub.docker.com/r/alleninstituteforai/olmocr

---

*文档生成时间: 2026-03-02*  
*基于代码版本: v0.4.0*
