# SERA Code Map

## 项目概述

**SERA (Soft-Verified Efficient Repository Agents)** 是一个用于生成合成代码修复数据的框架。它通过从真实代码库中提取函数，并使用 LLM 生成模拟的 PR 问题描述，然后运行 SWE-agent 来生成修复轨迹，从而创建训练数据。

## 整体架构

```
SERA/
├── sera/                          # 主代码包
│   ├── main.py                    # 实验入口点
│   ├── config_schema.py           # 配置数据结构定义
│   ├── utils.py                   # 通用工具函数
│   ├── constants.py               # 常量定义
│   ├── configs/                   # 配置文件
│   │   ├── pipeline/              # 数据处理流水线配置
│   │   ├── sweagent/              # SWE-agent 配置
│   │   └── *.yaml                 # 各种实验配置
│   └── datagen/                   # 数据生成模块
│       ├── data/                  # 数据处理
│       │   ├── generate/          # 数据生成功能
│       │   ├── distill/           # 蒸馏/合成 PR 生成
│       │   ├── eval/              # 评估模块
│       │   └── postprocess/       # 后处理
│       ├── inference/             # 推理服务脚本
│       └── train/                 # 训练脚本和配置
├── modules/                       # 外部模块
│   ├── code2flow/                 # 代码调用图分析工具
│   └── SERA-SWE-Agent/            # SWE-agent 定制版本
└── assets/                        # 静态资源
```

## 核心模块

### 1. 配置系统 (`sera/config_schema.py`)

**职责**: 使用 Python dataclass 定义所有配置参数，确保类型安全。

**主要类**:

| 类名 | 说明 |
|------|------|
| `SeraConfig` | 主配置对象，包含所有子配置 |
| `GenerateConfig` | 数据生成配置 |
| `DistillConfig` | 蒸馏/合成 PR 生成配置 |
| `EvalConfig` | 评估配置 |
| `PostprocessConfig` | 后处理配置 |
| `ModelConfig` | LLM 模型配置 |
| `SWEAgentWrapperConfig` | SWE-agent 包装器配置 |
| `PersonalRepoConfig` | 个人代码库配置 |
| `ExistingRepoConfig` | 现有代码库配置 |

**关键配置参数**:

```python
# 生成配置
generate.fns_per_repo: int = 5000        # 每个仓库提取的最大函数数
generate.insts_per_fn: int = 1           # 每个函数处理次数
generate.personal_repos: List[PersonalRepoConfig]  # 个人代码库列表
generate.existing_repos: List[ExistingRepoConfig]  # 现有代码库列表

# 蒸馏配置
distill.model.name: str                  # 模型名称 (openai/xxx 或 anthropic/xxx)
distill.model.url: str                   # API 端点
distill.sweagent_wrapper_config.num_workers: int = 32  # 并发 workers
distill.shard: int = 0                   # 分片索引
distill.total_shards: int = 1            # 总分片数

# 评估配置
eval.compare_patch_threshold: float = 1  # 补丁验证阈值 (1=hard, 0-1=soft, 0=no verify)

# 后处理配置
postprocess.tool_call_format: str = "hermes"  # 工具调用格式 (hermes/xml/raw)
postprocess.add_think: bool = False       # 是否添加思考标签
postprocess.enforce_submit: bool = True   # 是否只处理成功提交的轨迹
```

### 2. 主流程 (`sera/main.py`)

**职责**: 实验编排和流水线执行。

**核心类**:

```python
class Experiment:
    """实验管理类，控制整个数据生成流水线"""
    
    stage_map = {
        "pipeline": -1,          # 运行所有阶段
        "generate": 0,           # 数据生成
        "distill_stage_one": 1,  # 第一阶段蒸馏
        "distill_stage_two": 2,  # 第二阶段蒸馏
        "eval": 3,               # 评估
        "postprocess": 4         # 后处理
    }
    
    def run(self, stage: str) -> None:
        """执行指定阶段"""
        
    def _run_generate(self, cfg: GenerateConfig, skip: bool = False) -> None:
        """生成阶段: 从代码库提取函数并创建实例"""
        
    def _run_distill_one(self, cfg: DistillConfig, skip: bool = False) -> None:
        """蒸馏阶段1: 生成合成 PR 问题"""
        
    def _run_distill_two(self, cfg: DistillConfig, skip: bool = False) -> None:
        """蒸馏阶段2: 使用合成 PR 生成修复轨迹"""
        
    def _run_eval(self, cfg: EvalConfig, skip: bool = False) -> None:
        """评估阶段: 验证补丁质量"""
        
    def _run_postprocess(self, cfg: PostprocessConfig, skip: bool = False) -> None:
        """后处理阶段: 格式化输出训练数据"""
```

**流水线流程**:

```
1. Generate → 2. Distill Stage One → 3. Distill Stage Two → 4. Eval → 5. Postprocess
```

### 3. 数据生成模块 (`sera/datagen/data/generate/`)

#### 3.1 数据生成入口 (`generate.py`)

**职责**: 协调数据生成流程。

**主要函数**:

```python
def main(config: GenerateConfig, metadata_dir: str, folder: ExperimentFolder) -> None:
    """
    主生成函数
    - 处理个人代码库 (LocalRepository)
    - 处理现有代码库 (ExistingRepository)
    - 构建 NoBugDataset
    """
```

#### 3.2 代码库类 (`classes.py`)

**职责**: 抽象和管理代码库实例。

**核心类层次**:

```python
@dataclass
class Repository:
    """代码库基类"""
    org_name: str                    # GitHub 组织名
    last_name: str                   # 仓库名
    top_level_folder: List[str]      # 顶层代码目录
    overwrite_cg: bool               # 是否覆盖调用图缓存
    repo_path: Optional[Path]        # 本地克隆路径
    instances: List[RepositoryInstance]

class LocalRepository(Repository):
    """需要创建 Docker 容器的新代码库"""
    python_version: str
    install_cmds: List[str]
    test_cmd: str
    skip_package_name: List[str]
    language: str
    commits: Optional[list[str]]     # 要处理的 commit 列表

class ExistingRepository(Repository):
    """已有 Docker 容器的代码库 (SWE-Bench/SWE-Smith)"""
    source: str                      # "swebench" | "swesmith"
    base_commit: Optional[str]
    instance_id: Optional[str]
    image_name: Optional[str]

@dataclass
class RepositoryInstance:
    """单个代码库实例 (特定 commit)"""
    parent: Repository
    base_commit: str
    image_name: str                  # Docker 镜像名
    call_graph: Optional[nx.DiGraph] # 调用图
    folders: List[str]               # 代码文件夹通配符
```

#### 3.3 Docker 容器管理 (`docker.py`)

**职责**: 构建和管理代码库的 Docker 容器。

**主要函数**:

```python
def build_container(
    org_dh: str,                    # Docker Hub 组织
    org_gh: str,                    # GitHub 镜像组织
    gh_owner: str,                  # GitHub 仓库所有者
    repo_name: str,                 # 仓库名
    commit: str,                    # Commit hash
    install_cmds: list,             # 安装命令
    test_cmd: str = None,           # 测试命令
    language: str = "python",       # 编程语言
    python_version: str = "3.10",   # Python 版本
    package_name: str = None        # 要跳过的包名
) -> str:
    """为代码库创建 Docker 容器"""

def create_profile_class(...) -> type[RepoProfile]:
    """创建 SWE-smith RepoProfile 类"""

def build_profile_image(...) -> tuple[bool, Optional[str]]:
    """构建 Docker 镜像"""

def dockerhub_tag_exists(image: str) -> bool:
    """检查 Docker Hub 上是否存在标签"""
```

#### 3.4 代码解析 (`codebase_parsing.py`)

**职责**: 解析代码库结构并生成调用图。

**主要函数**:

```python
def find_code_folders(
    repo_path: str,
    repo_last_name: str,
    base_commit: str,
    top_level_folder: List[str]
) -> List[str]:
    """自动发现代码库的顶层代码文件夹"""

def get_adj_list(
    repo_path: str,
    repo_last_name: str,
    base_commit: str,
    relevant_folders: List[str],
    metadata_dir: str,
    overwrite: bool = False
) -> Dict:
    """
    生成代码调用图的邻接表
    - 使用 code2flow 工具生成调用图
    - 缓存结果到 metadata_dir
    - 转换为 NetworkX DiGraph 格式
    """

def convert_code2flow_to_adj(loaded_json) -> tuple:
    """将 code2flow JSON 输出转换为邻接表"""

def convert_to_file_path(call_graph, folders, node_id_to_name, nodes) -> tuple:
    """将函数名映射到完整文件路径"""
```

#### 3.5 数据集构建 (`no_bug_dataset.py`)

**职责**: 从无 bug 代码生成训练实例。

**核心类**:

```python
class NoBugDataset(SyntheticDataset):
    """
    无 bug 数据集生成器
    - 从代码库调用图中提取函数
    - 为每个函数创建合成实例
    - 生成 stage_one_instances.yaml
    """
    
    def build_dataset(self) -> None:
        """构建并保存数据集"""
        
    def process_repo(self, repo: RepositoryInstance) -> List[SyntheticInstance]:
        """处理单个代码库实例"""
        
    def process_instance(self, fn_path: str, replicas: int, repo: RepositoryInstance) -> List[SyntheticInstance]:
        """处理单个函数，创建合成实例"""

@dataclass
class SyntheticInstance:
    """合成数据实例"""
    repo: RepositoryInstance        # 所属的代码库实例
    start_fn: str                   # 起始函数名
    start_fn_file: str              # 起始函数所在文件
```

### 4. 蒸馏模块 (`sera/datagen/data/distill/`)

#### 4.1 蒸馏入口 (`distill.py`)

**职责**: 运行 SWE-agent 生成合成 PR 和修复轨迹。

**核心类**:

```python
class DistillRunner:
    """
    SWE-agent 蒸馏运行器
    - 管理分片数据处理
    - 构建 SWE-agent 命令
    - 执行批量推理
    """
    
    def __init__(
        self,
        config: DistillConfig,
        folder: ExperimentFolder,
        instances_fp: Path,
        cfg_fp: Path,
        args: dict = {}
    ):
        """
        - config: 蒸馏配置
        - instances_fp: 实例 YAML 文件路径
        - cfg_fp: SWE-agent 配置文件路径
        - args: 额外参数
        """
    
    @property
    def name(self) -> str:
        """生成运行名称 (包含模型、温度、限制等信息)"""
        
    @property
    def output_dir(self) -> Path:
        """输出目录路径"""
        
    def run(self) -> Path:
        """
        执行 SWE-agent 批量运行
        - 支持本地模型和 API 模型
        - 支持 OpenAI 和 Anthropic API
        - 返回输出目录
        """

def scrape_synthetic_prs(
    instance_fp: Path,
    traj_dir: Path,
    remove_duplicates: bool = True
) -> List[Dict]:
    """
    从第一阶段输出中提取合成 PR
    - 读取 .pred 文件获取补丁
    - 读取 .synth 文件获取合成 PR 描述
    - 过滤无效补丁
    - 去重
    """

def get_dataset_shard(
    instances_fp: str,
    shard: int,
    total_shards: int
) -> str:
    """将数据集分片用于并行处理"""
```

**两阶段蒸馏**:

```
Stage One: 生成合成 PR
- 输入: stage_one_instances.yaml (包含 start_fn, start_fn_file)
- 使用 e2e.yaml 配置
- 输出: 合成 PR 描述 (.synth 文件) 和修复补丁 (.pred 文件)

Stage Two: 使用合成 PR 生成修复轨迹  
- 输入: stage_two_instances.yaml (包含合成 PR 作为 problem_statement)
- 使用 qwen.yaml 配置
- 输出: 完整修复轨迹 (.traj 文件) 和最终补丁 (.pred 文件)
```

### 5. 评估模块 (`sera/datagen/data/eval/`)

#### 5.1 评估入口 (`eval.py`)

**职责**: 验证生成的补丁是否有效。

**核心函数**:

```python
def eval_loop(
    config: EvalConfig,
    instances_fp: Path,
    second_stage_dir: Path
) -> List[str]:
    """
    评估第二阶段生成的补丁
    - 读取 instances 文件获取目标补丁
    - 读取第二阶段输出获取生成补丁
    - 使用 compare_patch_recall 比较补丁
    - 返回成功修复的实例 ID 列表
    """

def compare_patch_recall(
    target_patch: str,
    produced_patch: str,
    threshold: float
) -> Optional[bool]:
    """
    比较两个补丁的相似度
    - 提取添加/删除的行
    - 计算召回率 (匹配行数 / 目标总行数)
    - 根据阈值判断是否匹配
    
    Args:
        target_patch: 目标补丁 (第一阶段生成)
        produced_patch: 生成补丁 (第二阶段生成)
        threshold: 匹配阈值 (1.0=完全匹配, 0.5=50%匹配)
    
    Returns:
        True if 召回率 >= threshold, False if < threshold, None if 目标补丁为空
    """
```

### 6. 后处理模块 (`sera/datagen/data/postprocess/`)

#### 6.1 后处理入口 (`postprocess.py`)

**职责**: 将轨迹转换为训练数据格式。

**核心函数**:

```python
def format_and_save(
    config: PostprocessConfig,
    traj_dir: Path,
    report_path: Optional[Path],
    out_dir: Path
) -> Path:
    """
    格式化轨迹并保存为训练数据
    - 根据评估报告过滤成功实例
    - 转换工具调用格式 (hermes/xml/raw)
    - 添加 train key (用于 axolotl)
    - 重新格式化助手消息
    """

def get_raw_trajectories(
    traj_dir: Path,
    report: Dict,
    tool_call_format: str,
    add_think: bool,
    enforce_submit: bool,
    tool_json: bool
) -> List[Dict]:
    """
    从轨迹目录提取原始轨迹数据
    - 遍历所有实例文件夹
    - 读取 .traj 文件
    - 转换消息格式
    - 过滤未成功提交的轨迹
    """

def create_file_name(
    config: PostprocessConfig,
    traj_dir: Path,
    report_path: Path
) -> str:
    """生成输出文件名 (包含所有配置参数)"""
```

**格式映射**:

```python
MAP_TO_PARSER = {
    "xml": transform_traj_xml,      # XML 格式工具调用
    "hermes": transform_traj_hermes, # Hermes 格式 (默认)
    "raw": transform_traj_raw        # 原始格式
}

MAP_TO_SYSTEM_PROMPT = {
    "xml": XML_DEFAULT_SYSTEM_PROMPT,
    "hermes": HERMES_DEFAULT_SYSTEM_PROMPT,
    "raw": None
}
```

#### 6.2 工具函数 (`utils.py`)

**职责**: 轨迹转换的辅助函数。

```python
def transform_traj_hermes(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    """
    将轨迹转换为 Hermes 格式
    - 工具调用: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    - 工具响应: <tool_response>...</tool_response>
    - 支持添加思考标签: <think>...</think>
    """

def transform_traj_xml(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    """
    将轨迹转换为 XML 格式
    - 函数调用: <function=name><parameter=key>value</parameter></function>
    """

def transform_traj_raw(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    """保持原始格式不变"""

def add_train_key(dataset: List[Dict]) -> List[Dict]:
    """
    为消息添加 train key
    - assistant 消息: train=True
    - 其他消息: train=False
    """

def reformat_assistant_message(dataset: List[Dict], mode: str) -> List[Dict]:
    """
    重新格式化助手消息
    - mode="keep_only_think": 只保留思考内容
    - mode="keep_only_non_think": 只保留非思考内容
    """

def reformat_think_message(content: str) -> Optional[str]:
    """确保思考标签格式正确 (Qwen3 格式)"""

def parse_text_indexed(text: str) -> Optional[tuple]:
    """解析文本中的思考内容和工具调用"""
```

### 7. 数据过滤与采集 (`sera/datagen/data/`)

#### 7.1 GitHub Issue 采集 (`scrape_github.py`)

**职责**: 采集 GitHub issue 作为合成 PR 的示例。

```python
def scrape_issue_texts(
    org: str,
    repo: str,
    n: int,
    out_path: str = "issues.json",
    state: str = "all",
    skip_pr: bool = True
) -> List[Dict[str, Any]]:
    """
    从 GitHub 仓库采集 issue 文本
    - 使用 GitHub REST API v3
    - 需要 GITHUB_TOKEN 或 GH_TOKEN 环境变量
    - 过滤掉 PR (pull_request 字段存在)
    - 返回 issue body 列表
    """

# 使用示例:
# python scrape_github.py -o SWE-agent -n SWE-agent -c 100
```

#### 7.2 数据集过滤 (`filter.py`)

**职责**: 根据质量启发式规则过滤轨迹。

**过滤模式**:

```python
# long_edit: 移除补丁修改行数 > 40 的轨迹
PATCH_EDIT_MAX = 40

# user_length: 移除平均工具调用响应 token 数 > 600 的轨迹
TOOL_CALL_TOKEN_MAX = 600

def analyze_diff(patch_text: str) -> Dict:
    """
    分析补丁统计信息
    - added_lines: 添加行数
    - deleted_lines: 删除行数
    - new_files: 新文件数
    """
```

### 8. 训练模块 (`sera/datagen/train/`)

#### 8.1 Unsloth LoRA 训练 (`train_unsloth_lora.py`)

**职责**: 使用 Unsloth 进行高效的 LoRA/QLoRA 训练。

**特性**:
- 支持 MoE 模型 (如 Qwen3-30B-A3B)
- 4-bit 量化减少内存使用
- 8-bit AdamW 优化器
- FlexAttention 加速

**主要组件**:

```python
def main():
    """
    主训练流程:
    1. 解析命令行参数
    2. 初始化分布式训练
    3. 加载模型 (FastLanguageModel/FastModel)
    4. 添加 LoRA 适配器
    5. 加载数据集
    6. 配置训练参数
    7. 开始训练
    """

# 关键函数
def _maybe_init_torch_distributed():
    """初始化 torch 分布式训练"""

def _load_model_and_tokenizer(model_name: str, is_moe: bool, max_seq_length: int):
    """加载模型和分词器"""

def _setup_lora_config(model, lora_rank: int, lora_alpha: int, target_modules: List[str]):
    """配置 LoRA"""

def _load_training_dataset(
    train_file: str,
    eval_file: str,
    tokenizer,
    system_prompt: str,
    max_seq_length: int
):
    """加载训练数据集"""
```

#### 8.2 DeepSpeed 配置 (`deepspeed_configs/`)

- `zero1.json`: ZeRO Stage 1 (优化器状态分片)
- `zero2.json`: ZeRO Stage 2 (优化器状态 + 梯度分片)
- `zero3.json`: ZeRO Stage 3 (完整分片)
- `zero3_bf16_cpuoffload_params.json`: ZeRO-3 + CPU offload 参数
- `zero3_bf16_cpuoffload_all.json`: ZeRO-3 + CPU offload 全部

#### 8.3 Axolotl 配置 (`train_config/`)

- `axolotl_qwen25_32b.yaml`: Qwen2.5 32B 配置
- `axolotl_qwen3_32b.yaml`: Qwen3 32B 配置
- `axolotl_qwen3_8b.yaml`: Qwen3 8B 配置
- `llamafactory_qwen3_full_sft.yaml`: LLaMA-Factory 全参数微调
- `unsloth_qwen3_moe_qlora.yaml`: Unsloth MoE QLoRA 配置

### 9. 外部模块

#### 9.1 code2flow (`modules/code2flow/`)

**职责**: 多语言代码调用图生成工具。

**核心文件**:

| 文件 | 说明 |
|------|------|
| `engine.py` | 主引擎，处理命令行参数和流程控制 |
| `model.py` | 数据模型 (Node, Edge, Group, Variable) |
| `python.py` | Python 语言解析器 |
| `javascript.py` | JavaScript 解析器 |
| `php.py` | PHP 解析器 |
| `ruby.py` | Ruby 解析器 |

**工作原理**:
1. 解析源代码 AST
2. 识别函数/方法定义和调用
3. 构建调用关系图
4. 输出为 Graphviz DOT 或 JSON 格式

**关键类**:

```python
class Node:
    """表示代码中的函数/方法"""
    name: str
    token: str
    group: Group
    
class Edge:
    """表示调用关系"""
    node0: Node  # 调用者
    node1: Node  # 被调用者
    
class Group:
    """表示文件/类/模块"""
    name: str
    nodes: List[Node]
```

### 10. 工具函数 (`sera/utils.py`)

**职责**: 通用工具函数和数据结构。

**主要函数**:

```python
# JSON/YAML 操作
def dump_json(fp: str, data: Any, overwrite: bool = False) -> None:
def dump_jsonl(fp: str, data: List, overwrite: bool = False) -> None:
def save_yaml(fp: str, data: Any, overwrite: bool = False) -> None:
def load_yaml(fp: str) -> Any:

# 后处理查询
def pp_query(
    system: str,
    prompt: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    max_tokens: int = 4096,
    retries: int = 0,
    args: dict = {}
) -> str:
    """
    使用 OpenAI/Anthropic API 进行查询
    - 支持 OpenAI 兼容服务器
    - 支持 Anthropic API
    - 支持模板渲染 (Jinja2)
    """

def pp_regex(text: str, re_string: str) -> List[str]:
    """正则表达式提取"""

# 实验文件夹管理
class ExperimentFolder:
    """
    实验文件夹管理器
    
    创建结构:
    base_dir/name/
        configs/    # 配置文件
        data/       # 数据文件
        trajs/      # 轨迹文件
    """
    
    @classmethod
    def create(cls, base_dir: str, name: str, exist_ok: bool = True) -> 'ExperimentFolder':
        """创建实验文件夹结构"""
        
    def add_config(self, path: Path) -> None:
        """复制配置文件到实验目录"""
```

### 11. 常量定义 (`sera/constants.py`)

**职责**: 定义系统使用的常量。

**主要内容**:

```python
# 路径常量
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录

# 系统提示词
HERMES_DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant...
# Tools
...
"""

XML_DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant...
<IMPORTANT>...</IMPORTANT>
"""

# 评估提示词
CHECK_SYNTHETIC_TRAJ_PROMPT = """
<fix_steps>...</fix_steps>
<initial_prompt>...</initial_prompt>
Your task is to judge if the final fix is valid...
"""

SYNTHETIC_TRAJ_PROMPT = """
<fix_steps>...</fix_steps>
...generate synthetic PR...
"""

# SWE-Bench 和 SWE-Smith 镜像映射
SWEBENCH_IMAGES = {
    "django__django-7530": {"image_name": "swebench/sweb.eval.x86_64....", "base_commit": "..."},
    # ...
}

SWESMITH_IMAGES = {
    "org/repo": {"image_name": "swesmith/...", "base_commit": "..."},
    # ...
}

# SWE-agent 工具定义
SWEAGENT_TOOLS = [
    {"type": "function", "function": {"name": "bash", ...}},
    {"type": "function", "function": {"name": "str_replace_editor", ...}},
    {"type": "function", "function": {"name": "submit", ...}},
]
```

## 数据流

```
1. Generate 阶段:
   代码库 → Clone → Docker 容器 → 调用图提取 → stage_one_instances.yaml
                ↓
         (LocalRepository / ExistingRepository)

2. Distill Stage One:
   stage_one_instances.yaml + SWE-agent (e2e config) 
   → 生成合成 PR → stage_one_output/
   
3. Distill Stage Two:
   stage_one_output/ → scrape_synthetic_prs() → stage_two_instances.yaml
   → SWE-agent (qwen config) → stage_two_output/
   
4. Eval 阶段:
   stage_two_output/ + stage_two_instances.yaml
   → compare_patch_recall() → report.json (resolved_ids)
   
5. Postprocess 阶段:
   stage_one_output/ + stage_two_output/ + report.json
   → format_and_save() → training_data.jsonl
```

## 配置系统 (Hydra)

SERA 使用 Hydra 进行配置管理。

**配置层次**:

```
sera/configs/
├── config.yaml (默认配置)
├── specialization_django.yaml
├── specialization_sympy.yaml
├── specialization_personal.yaml
├── specialization_anthropic.yaml
├── swesmith_scaling.yaml
├── pipeline/
│   └── default_pipeline.yaml (数据处理流水线)
└── sweagent/
    ├── e2e.yaml (Stage One 配置)
    └── qwen.yaml (Stage Two 配置)
```

**配置继承**:

```yaml
# specialization_django.yaml
defaults:
  - sera                    # 继承默认配置
  - _self_

name: specialize_django_test

generate:
  fns_per_repo: 3
  existing_repos:
    - org_name: django
      last_name: django
      source: swebench
      instance_id: django__django-7530

distill:
  model:
    name: openai/GLM-4.5-Air
    url: null
  sweagent_wrapper_config:
    num_workers: 24
```

## 依赖关系

```
sera/
├── main.py
│   ├── config_schema.py
│   ├── utils.py
│   ├── datagen/data/generate/generate.py
│   ├── datagen/data/distill/distill.py
│   ├── datagen/data/eval/eval.py
│   └── datagen/data/postprocess/postprocess.py
│
├── datagen/data/generate/
│   ├── generate.py
│   │   ├── classes.py
│   │   ├── codebase_parsing.py → code2flow
│   │   ├── docker.py → swesmith
│   │   └── no_bug_dataset.py
│   └── classes.py
│       └── codebase_parsing.py
│
├── datagen/data/distill/
│   └── distill.py → sweagent (外部命令)
│
├── datagen/data/eval/
│   └── eval.py
│
└── datagen/data/postprocess/
    ├── postprocess.py
    └── utils.py
        └── constants.py
```

## 使用示例

### 1. 基础数据生成 (SWE-Bench 仓库)

```bash
python sera/main.py \
    --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=http://localhost:24444/v1
```

### 2. 个人仓库数据生成

```bash
# 1. 修改 specialization_personal.yaml 添加仓库信息
# 2. 创建 GitHub 镜像组织
# 3. 运行生成

python sera/main.py \
    --config-name=specialization_personal \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL \
    generate.docker.gh_mirror_org=your-org \
    generate.docker.docker_org=your-docker-org
```

### 3. 采集 GitHub Issues

```bash
python sera/datagen/data/scrape_github.py \
    -o SWE-agent \
    -n SWE-agent \
    -c 100
```

### 4. 过滤数据集

```bash
python sera/datagen/data/filter.py \
    -d training_data.jsonl \
    -f experiments/traj/ \
    -fm long_edit
```

### 5. 启动推理服务

```bash
bash sera/datagen/inference/launch_glm45.sh 8 24444 42
# 参数: TP大小, 端口, 随机种子
```

## 关键设计决策

### 1. 两阶段蒸馏

- **Stage One**: 从代码函数生成合成 PR 问题描述
- **Stage Two**: 使用合成 PR 作为输入，生成修复轨迹
- **优势**: 可以验证第二阶段是否能正确理解并修复第一阶段生成的问题

### 2. 软验证 (Soft Verification)

- **Hard Verification (threshold=1.0)**: 要求第二阶段补丁与第一阶段完全匹配
- **Soft Verification (0 < threshold < 1)**: 允许部分差异，通过补丁行召回率判断
- **No Verification (threshold=0)**: 接受所有提交

### 3. 分片并行

- 支持将数据集分片到多个推理服务器
- 每个分片独立运行，最后合并结果
- 适合大规模数据生成

### 4. Docker 容器复用

- 自动检测 Docker Hub 上已存在的镜像
- 支持镜像缓存，避免重复构建
- 支持推送到 Docker 组织实现持久化

### 5. 调用图缓存

- 使用 code2flow 生成调用图
- 缓存到 `metadata_dir`，避免重复解析
- 支持强制重新生成

## 扩展点

### 1. 添加新的语言支持

修改 `docker.py` 中的 `LANGUAGE_PROFILES`:

```python
LANGUAGE_PROFILES = {
    "python": PythonProfile,
    "go": GoProfile,
    "rust": RustProfile,
    "javascript": JavaScriptProfile,
    "java": JavaProfile,  # 新增
}
```

### 2. 自定义 SWE-agent 配置

在 `sera/configs/sweagent/` 创建新的 YAML 文件:

```yaml
agent:
  type: default
  templates:
    system_template: "..."
    instance_template: "..."
  tools:
    execution_timeout: 300
    enable_bash_tool: true
  model:
    name: openai/model-name
    temperature: 0.6
```

### 3. 添加新的后处理格式

在 `postprocess/utils.py` 添加转换函数:

```python
def transform_traj_custom(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    """自定义格式转换"""
    ...

# 在 postprocess.py 注册
MAP_TO_PARSER = {
    "xml": transform_traj_xml,
    "hermes": transform_traj_hermes,
    "raw": transform_traj_raw,
    "custom": transform_traj_custom,  # 新增
}
```

## 常见问题

### Q: 如何处理构建失败的仓库?

A: 检查 `install_cmds` 是否正确，或查看 `skip_package_name` 跳过冲突包。

### Q: 为什么第二阶段没有生成补丁?

A: 可能是第一阶段生成的 PR 描述不够清晰，或者代码库太复杂。尝试调整模型或温度参数。

### Q: 如何提高数据质量?

A: 
1. 提高 `eval.compare_patch_threshold` 进行更严格的验证
2. 使用 `filter.py` 过滤长补丁或高 token 消耗
3. 使用 GitHub issue 作为示例 PR (通过 `scrape_github.py`)

### Q: 如何恢复中断的运行?

A: 使用相同的 `name` 参数重新运行命令，或使用 `stage` 参数从特定阶段恢复:

```bash
python sera/main.py \
    --config-name=specialization_django \
    name=previous_run_name \
    stage=distill_stage_two
```
