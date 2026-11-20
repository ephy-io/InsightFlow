import importlib
import os
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from insightflow.catalog.factory import create_catalog_store
from insightflow.constants import EXTRACTION_MAX_TOOL_ROUNDS, SQL_RESULT_LIMIT
from insightflow.utils import log

_ENV_VALUE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _resolve_environment_values(value: Any) -> Any:
    """递归解析配置中的 $NAME / ${NAME}，避免把密钥写入 YAML。"""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(3)
            default = match.group(2)
            return os.getenv(name, default if default is not None else match.group(0))

        return _ENV_VALUE_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_resolve_environment_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_environment_values(item) for key, item in value.items()}
    return value


def _load_project_dotenv() -> None:
    """读取项目根目录 .env；仅补充未在进程中设置的变量，且不记录密钥。"""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


class LLMProviderConfig(BaseModel):
    """为单个提供者解析了 LLM 对象。"""

    model_config = {"arbitrary_types_allowed": True}

    default_llm: MagicMock | BaseChatModel
    embedding_model: MagicMock | BaseModel | None = None
    text2sql_llm: MagicMock | BaseChatModel | None = None
    analysis_llm: MagicMock | BaseChatModel | None = None


class TracingConfig(BaseModel):
    enabled: bool = False
    provider: str | None = None  # 'langfuse' | 'langsmith' | None
    langfuse_host: str | None = None
    sample_rate: float = 1.0


class MetricsConfig(BaseModel):
    enabled: bool = False
    prometheus_port: int | None = None


class AuditConfig(BaseModel):
    enabled: bool = False
    sink: str = "log"  # 'log' → 'file' 时间轴 :
    path: str | None = None
    mask_sql_literals: bool = True


class ObservabilityConfig(BaseModel):
    tracing: TracingConfig = TracingConfig()
    metrics: MetricsConfig = MetricsConfig()
    audit: AuditConfig = AuditConfig()


class Config(BaseModel):
    """InsightFlow 应用程序的配置模型。

    Attributes:
        organization (str): 组织名称，默认为"公司".
        dialect (str): 要使用的 SQL 方言。默认为 "presto"。
        default_llm (BaseChatModel): 一般任务的默认语言模型。
        embedding_model (BaseModel): 嵌入生成语言模型。
        text2sql_llm (Optional[BaseChatModel]): 专用于文本到 SQL 任务的语言模型。
        bi_config (Dict[str, Any]): 从 YAML 文件装入的 BI 配置。 默认为空。
        data_warehouse_config (Dict[str, Any]): 数据仓库配置 默认为空 dict。
    """

    model_config = {"arbitrary_types_allowed": True}

    # 常规配置
    organization: str = "The Company"
    dialect: str = "presto"

    # LLM 配置
    default_llm: MagicMock | BaseChatModel
    embedding_model: MagicMock | BaseModel | None = None
    text2sql_llm: MagicMock | BaseChatModel | None = None
    analysis_llm: MagicMock | BaseChatModel | None = None
    # 多个 LLM 提供者(可选)
    llm_provider: str | None = None
    llm_providers: dict[str, LLMProviderConfig] = {}

    # BI 配置
    bi_config: dict[str, Any] = {}

    # 数据仓库配置
    data_warehouse_config: dict[str, Any] = {}

    # 目录存储器
    catalog_store: Any = None

    # 向量库路径
    vector_db_path: str | None = None
    # 检索模式：baseline、bm25_dense、rrf、rrf_reranker。
    schema_retrieval_strategy: str = "rrf"
    reranker_config: dict[str, Any] = {}
    memory_db_path: str | None = None

    # MCP 服务器配置
    mcp_servers: list[dict[str, Any]] = []

    # 报表配置
    report_directory: str = "./data"

    # 代码执行配置
    python_executor: str = "docker"  # 默认使用容器隔离；local 必须由部署方显式启用。

    # 可视化配置
    visualization_mode: str | None = "rule"  # 选项: "rule" , "llm" , None (skip可视化)

    # SQL 执行结果限制配置
    enable_sql_result_limit: bool = True
    sql_result_limit: int = SQL_RESULT_LIMIT

    # 只读 SQL 防护默认开启；数据库账号仍必须具备只读权限。
    enable_fail_closed_sql_guard: bool = True

    # HITL 信任门配置( 零递归的默认 OFF)
    enable_confidence_gate: bool = False
    sql_confidence_threshold: float = 0.7
    confidence_gate_mode: str = "post_exec"  # 选项: "post_exec" (默认), "pre_exec" (阶段-2)
    confidence_evaluator_mode: str = "simple"

    # 将零行结果作为软失败处理( EmptyResultError). 默认 OFF :
    # 空结果保持 SQL_SUCCESS ,以保留现有的可视化-输入行为。
    enable_empty_result_error: bool = False

    # HITL Golden - SQL 抓取配置(零退缩默认为 OFF)
    enable_golden_sql: bool = False
    golden_sql_namespace: str = "global"

    # 信息提取:每个问题允许的工具回合( search_knowledge 或 AskHuman)
    extraction_max_tool_rounds: int = EXTRACTION_MAX_TOOL_ROUNDS

    # SQL 重试/ 恢复配置
    sql_max_retries: int = 3
    # 当 True 时，超时/ 连接失败可能被重试; 默认的 OFF 保存
    # 现有的"超时"行为立即结束。
    retry_on_timeout: bool = False
    # 可选的 per- error 类策略覆盖，如 { "SQLSyntaxError" : "retry" }.
    # 保留用于阶段2(例如启用 RETRY_WITH_NEW_TABLE);默认空 =无覆盖。
    retry_strategy_overrides: dict[str, Any] = {}

    # 上下文管理配置
    context_config: dict[str, Any] = {}

    # 记忆和模式学习配置，结构与 context_config 一致。参见 memory_config.py。
    memory_config: dict[str, Any] = {}

    # 可观察性配置 ( S1 –所有子旗默认 OFF)
    observability: ObservabilityConfig = ObservabilityConfig()

    # 时间序列服务配置
    timeseries_forecasting_service_url: str = "http://localhost:8765"
    # 预测模型最小输入长度的可选手动覆盖。当未设置( None)时，
    # 从预测服务(从模型配置读取)中获取该值。
    timeseries_forecasting_min_input_length: int | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        """从词典创建配置实例。

        Args:
            config (Dict[str, Any]): 包含配置值的词典。

        Returns:
            配置 : 一个新的配置实例，包含所提供的值。
        """
        return cls(**config)


class ConfigLoader:
    """Singleton类可以加载和管理 InsightFlow 的配置设置。

    此类提供了加载、获取和设置配置参数的方法
    用于应用，包括 LLM 模型，SQL 方言，以及其他设置。
    """

    _instance: "ConfigLoader | None" = None
    _config: Config | None = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    llm_configs = ["default_llm", "embedding_model", "text2sql_llm", "analysis_llm"]

    def get(self) -> Config:
        """获取当前配置。

        Returns:
            配置:当前配置实例。

        Raises:
            ValueError :如果配置尚未装入。
        """
        if self._config is None:
            raise ValueError("Configuration has not been loaded. Please call load() or set() first.")
        return self._config

    def load(self, config_file: str | None = None) -> None:
        """从 YAML 文件装入配置。

        Args:
            config_file (str, optional): 配置文件路径。 使用 CONFIG_FILE
                如果不提供环境变量或 'insightflow/config.yaml'。

        Raises:
            ImportError :如果pyyaml没有安装。
            FileNotFoundError :如果无法找到配置文件。
        """
        if config_file is None:
            config_file = os.getenv("CONFIG_FILE", "insightflow/config.yaml")

        _load_project_dotenv()

        if not find_spec("yaml"):
            raise ImportError("Please install pyyaml to use this feature.")

        import yaml

        try:
            with open(config_file, encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                if config_data is None:
                    config_data = {}
        except FileNotFoundError:
            log(f"Configuration file not found: {config_file}, leave config un-loaded.")
            return
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_file}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file {config_file}: {e}") from e

        self._process_config_dict(config_data)
        self._config = Config.from_dict(config_data)

    def _process_config_dict(self, config_data: dict[str, Any]) -> None:
        """处理配置词典。"""
        config_data["reranker_config"] = _resolve_environment_values(config_data.get("reranker_config", {}))
        self._process_llm_providers(config_data)

        providers = config_data.get("llm_providers", {})
        selected_provider = None

        default_llm_value = config_data.get("default_llm")
        if isinstance(default_llm_value, str):
            # 简化多提供者配置: default_llm :< provider_name >
            if not providers:
                raise ValueError("default_llm is a provider name but llm_providers is missing.")
            selected_provider = default_llm_value
        elif providers:
            # 后向兼容: 允许通过 llm_provider 选择提供者
            legacy_provider = config_data.get("llm_provider")
            if isinstance(legacy_provider, str):
                selected_provider = legacy_provider
            elif "default_llm" not in config_data:
                # 在配置顺序中选择第一个向后兼容的 YAML 行为提供商
                selected_provider = next(iter(providers.keys()), None)
            elif isinstance(default_llm_value, dict):
                raise ValueError(
                    "When using llm_providers, set default_llm to a provider name (e.g. default_llm: openai), "
                    "not a class config."
                )

        if providers:
            if not selected_provider or selected_provider not in providers:
                raise ValueError(f"Unknown LLM provider '{selected_provider}'. Available: {sorted(providers.keys())}")
            # 存储选中供运行时间浏览的提供者( UI / API 仍然可以覆盖每个请求)
            config_data["llm_provider"] = selected_provider
            # 为遗留呼叫站点传播顶级 LLM 对象
            config_data["default_llm"] = providers[selected_provider].default_llm
            config_data.setdefault("embedding_model", providers[selected_provider].embedding_model)
            config_data.setdefault("text2sql_llm", providers[selected_provider].text2sql_llm)
            config_data.setdefault("analysis_llm", providers[selected_provider].analysis_llm)
        elif "default_llm" not in config_data:
            raise ValueError("Missing LLM config key: default_llm")

        if not config_data.get("embedding_model"):
            log("WARN: Missing LLM config key: embedding_model, will use BM25 based retrival only")
        if "data_warehouse_config" not in config_data:
            raise ValueError("Missing Data Warehouse config key: data_warehouse_config")

        # 加载 BI 配置
        if "bi_config_file" in config_data:
            bi_config = self.load_bi_config(config_data["bi_config_file"])
            bi_config.update(config_data.get("bi_config", {}))
            config_data["bi_config"] = bi_config

        if "catalog_store" in config_data:
            if "store_type" not in config_data["catalog_store"]:
                raise ValueError("catalog_store must have a store_type field.")
            catalog_store_config = dict(config_data["catalog_store"])
            auto_load = catalog_store_config.pop("auto_load", True)
            catalog_store = create_catalog_store(
                **catalog_store_config,
                auto_load=auto_load,
                data_warehouse_config=config_data.get("data_warehouse_config"),
            )
        else:
            log("Catalog store config key `catalog_store` not found. Using default file system store.")
            catalog_store = create_catalog_store(
                store_type="file_system",
                auto_load=True,
                data_warehouse_config=config_data.get("data_warehouse_config"),
            )
        config_data["catalog_store"] = catalog_store

        for config_key in self.llm_configs:
            config_item = config_data.get(config_key)
            if not isinstance(config_item, dict) or "class" not in config_item:
                continue
            config_data[config_key] = self._instantiate_from_config_dict(config_item, config_key=config_key)

    def _instantiate_from_config_dict(self, config_item: dict[str, Any], *, config_key: str) -> Any:
        try:
            class_path = config_item["class"]
            if "." not in class_path:
                raise ValueError(f"Invalid class path format: {class_path}")
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            llm_cls = getattr(module, class_name)
            params = _resolve_environment_values(config_item.get("params", {}))
            return llm_cls(**params)
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to load {config_key} class '{config_item.get('class', '')}': {e}") from e

    def _process_llm_providers(self, config_data: dict[str, Any]) -> None:
        """将 llm_providers 解析为即时提供者配置(如果存在)."""
        raw_providers = config_data.get("llm_providers")
        if not raw_providers:
            return
        if not isinstance(raw_providers, dict):
            raise ValueError("llm_providers must be a mapping of provider_name -> config")

        providers: dict[str, LLMProviderConfig] = {}
        for provider_name, provider_cfg in raw_providers.items():
            if isinstance(provider_cfg, LLMProviderConfig):
                providers[str(provider_name)] = provider_cfg
                continue
            if not isinstance(provider_cfg, dict):
                raise ValueError(f"llm_providers.{provider_name} must be a mapping")

            resolved_cfg: dict[str, Any] = dict(provider_cfg)
            for config_key in self.llm_configs:
                config_item = resolved_cfg.get(config_key)
                if not isinstance(config_item, dict) or "class" not in config_item:
                    continue
                resolved_cfg[config_key] = self._instantiate_from_config_dict(
                    config_item, config_key=f"llm_providers.{provider_name}.{config_key}"
                )

            if "default_llm" not in resolved_cfg or resolved_cfg["default_llm"] is None:
                raise ValueError(f"llm_providers.{provider_name} missing default_llm")

            providers[str(provider_name)] = LLMProviderConfig(**resolved_cfg)

        config_data["llm_providers"] = providers

    def load_bi_config(self, bi_config_file: str) -> dict[str, Any]:
        """从一个 YAML 文件中加载 BI 配置。

        Args:
            bi_config_file (str): BI 配置文件路径。
                默认为 'example/bi.yaml'。

        Returns:
            Dict[str, Any]: 已装入的 BI 配置为字典。

        Raises:
            ImportError :如果pyyaml没有安装。
            FileNotFoundError :如果找不到 BI 配置文件。
        """
        if not find_spec("yaml"):
            raise ImportError("Please install pyyaml to use this feature.")

        import yaml

        bi_config_data: dict[str, Any] = {}

        try:
            with open(bi_config_file, encoding="utf-8") as file:
                bi_config_data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            log(f"Warning: BI config file '{bi_config_file}' not found. Ignore load BI config from yaml file.")
        except yaml.YAMLError as e:
            log(f"Warning: Invalid YAML in BI config file '{bi_config_file}': {e}. Using empty config.")
        except Exception as e:
            log(f"Warning: Failed to read BI config file '{bi_config_file}': {e}. Using empty config.")

        return bi_config_data

    def set(self, config: dict[str, Any]) -> None:
        """从字典中设定配置。

        Args:
            config (Dict[str, Any]): 包含配置值的词典。
        """
        self._process_config_dict(config)
        self._config = Config.from_dict(config)
