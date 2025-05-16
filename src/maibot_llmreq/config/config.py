from dataclasses import dataclass, field
from typing import List, Dict

from packaging.version import Version

NEWEST_VER = "0.1.0"  # 当前支持的最新版本


@dataclass
class APIProvider:
    name: str = ""  # API提供商名称
    base_url: str = ""  # API基础URL
    api_key: str = field(repr=False, default="")  # API密钥
    client_type: str = "openai"  # 客户端类型（如openai/google等，默认为openai）


@dataclass
class ModelInfo:
    model_identifier: str = ""  # 模型标识符（用于URL调用）
    name: str = ""  # 模型名称（用于模块调用）
    api_provider: str = ""  # API提供商（如OpenAI、Azure等）

    # 以下用于模型计费
    price_in: float = 0.0  # 每M token输入价格
    price_out: float = 0.0  # 每M token输出价格

    force_stream_mode: bool = False  # 是否强制使用流式输出模式


@dataclass
class RequestConfig:
    max_retry: int = 2  # 最大重试次数（单个模型API调用失败，最多重试的次数）
    timeout: int = (
        10  # API调用的超时时长（超过这个时长，本次请求将被视为“请求超时”，单位：秒）
    )
    retry_interval: int = 10  # 重试间隔（如果API调用失败，重试的间隔时间，单位：秒）
    default_temperature: float = 0.7  # 默认的温度（如果bot_config.toml中没有设置temperature参数，默认使用这个值）
    default_max_tokens: int = 1024  # 默认的最大输出token数（如果bot_config.toml中没有设置max_tokens参数，默认使用这个值）


@dataclass
class ModelUsageArgConfigItem:
    """模型使用的配置类
    该类用于加载和存储子任务模型使用的配置
    """

    name: str = ""  # 模型名称
    temperature: float | None = None  # 温度
    max_tokens: int | None = None  # 最大token数
    max_retry: int | None = None  # 调用失败时的最大重试次数


@dataclass
class ModelUsageArgConfig:
    """子任务使用模型的配置类
    该类用于加载和存储子任务使用的模型配置
    """

    name: str = ""  # 任务名称
    usage: List[ModelUsageArgConfigItem] = field(
        default_factory=lambda: []
    )  # 任务使用的模型列表


@dataclass
class ModuleConfig:
    INNER_VERSION: Version | None = None  # 配置文件版本

    req_conf: RequestConfig = field(default_factory=lambda: RequestConfig())  # 请求配置
    api_providers: Dict[str, APIProvider] = field(
        default_factory=lambda: {}
    )  # API提供商列表
    models: Dict[str, ModelInfo] = field(default_factory=lambda: {})  # 模型列表
    task_model_arg_map: Dict[str, ModelUsageArgConfig] = field(
        default_factory=lambda: {}
    )
