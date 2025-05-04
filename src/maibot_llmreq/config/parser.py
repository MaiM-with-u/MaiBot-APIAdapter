import os

import tomli
from packaging import version
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion

from .. import _logger as logger

from .config import (
    ModelUsageConfigItem,
    ModelUsageConfig,
    APIProvider,
    ModelInfo,
    NEWEST_VER,
    ModuleConfig,
)


def _get_config_version(toml: dict) -> Version:
    """提取配置文件的 SpecifierSet 版本数据
    Args:
        toml[dict]: 输入的配置文件字典
    Returns:
        Version
    """

    if "inner" in toml:
        try:
            config_version: str = toml["inner"]["version"]
        except KeyError as e:
            logger.error("配置文件中 inner 段 不存在 version 键")
            raise KeyError(f"配置文件中 inner 段 不存在 {e}, 这是错误的配置文件") from e
    else:
        toml["inner"] = {"version": "0.0.0"}
        config_version = toml["inner"]["version"]

    try:
        ver = version.parse(config_version)
    except InvalidVersion as e:
        logger.error(
            "配置文件中 inner 段 的 version 键是错误的版本描述\n"
            f"请检查配置文件，当前 version 键: {config_version}\n"
            f"错误信息: {e}"
        )
        raise InvalidVersion(
            "配置文件中 inner段 的 version 键是错误的版本描述\n"
        ) from e

    return ver


def _request_conf(parent: dict, config: ModuleConfig):
    request_conf_config = parent.get("request_conf")
    config.req_conf.max_retry = request_conf_config.get(
        "max_retry", config.req_conf.max_retry
    )
    config.req_conf.timeout = request_conf_config.get(
        "timeout", config.req_conf.timeout
    )
    config.req_conf.retry_interval = request_conf_config.get(
        "retry_interval", config.req_conf.retry_interval
    )
    config.req_conf.default_temperature = request_conf_config.get(
        "default_temperature", config.req_conf.default_temperature
    )
    config.req_conf.default_max_tokens = request_conf_config.get(
        "default_max_tokens", config.req_conf.default_max_tokens
    )


def _api_providers(parent: dict, config: ModuleConfig):
    api_providers_config = parent.get("api_providers")
    for provider in api_providers_config:
        name = provider.get("name", None)
        base_url = provider.get("base_url", None)
        api_key = provider.get("api_key", None)
        client_type = provider.get("client_type", "openai")

        if name in config.api_providers:  # 查重
            logger.error(f"重复的API提供商名称: {name}，请检查配置文件。")
            raise KeyError(f"重复的API提供商名称: {name}，请检查配置文件。")

        if name and base_url:
            config.api_providers[name] = APIProvider(
                name=name,
                base_url=base_url,
                api_key=api_key,
                client_type=client_type,
            )
        else:
            logger.error(f"API提供商 '{name}' 的配置不完整，请检查配置文件。")
            raise ValueError(f"API提供商 '{name}' 的配置不完整，请检查配置文件。")


def _models(parent: dict, config: ModuleConfig):
    models_config = parent.get("models")
    for model in models_config:
        model_identifier = model.get("model_identifier", None)
        name = model.get("name", model_identifier)
        api_provider = model.get("api_provider", None)
        price_in = model.get("price_in", 0.0)
        price_out = model.get("price_out", 0.0)
        force_stream_mode = model.get("force_stream_mode", False)

        if name in config.models:  # 查重
            logger.error(f"重复的模型名称: {name}，请检查配置文件。")
            raise KeyError(f"重复的模型名称: {name}，请检查配置文件。")

        if model_identifier and api_provider:
            # 检查API提供商是否存在
            if api_provider not in config.api_providers:
                logger.error(f"未声明的API提供商 '{api_provider}' ，请检查配置文件。")
                raise ValueError(
                    f"未声明的API提供商 '{api_provider}' ，请检查配置文件。"
                )
            config.models[name] = ModelInfo(
                name=name,
                model_identifier=model_identifier,
                api_provider=api_provider,
                price_in=price_in,
                price_out=price_out,
                force_stream_mode=force_stream_mode,
            )
        else:
            logger.error(f"模型 '{name}' 的配置不完整，请检查配置文件。")
            raise ValueError(f"模型 '{name}' 的配置不完整，请检查配置文件。")


def _task_model_usage(parent: dict, config: ModuleConfig):
    config.task_model_usage_map = {}

    model_usage_configs = parent.get("task_model_usage")

    for task_name, item in model_usage_configs.items():
        if task_name in config.task_model_usage_map:
            logger.error(f"子任务 {task_name} 已存在，请检查配置文件。")
            raise KeyError(f"子任务 {task_name} 已存在，请检查配置文件。")

        usage = []
        if isinstance(item, dict):
            if "model" in item:
                usage.append(
                    ModelUsageConfigItem(
                        name=item["model"],
                        temperature=item.get("temperature", None),
                        max_tokens=item.get("max_tokens", None),
                        max_retry=item.get("max_retry", None),
                    )
                )
            else:
                logger.error(f"子任务 {task_name} 的模型配置不合法，请检查配置文件。")
                raise ValueError(
                    f"子任务 {task_name} 的模型配置不合法，请检查配置文件。"
                )
        elif isinstance(item, list):
            for model in item:
                if isinstance(model, dict):
                    usage.append(
                        ModelUsageConfigItem(
                            name=model["model"],
                            temperature=model.get("temperature", None),
                            max_tokens=model.get("max_tokens", None),
                            max_retry=model.get("max_retry", None),
                        )
                    )
                elif isinstance(model, str):
                    usage.append(
                        ModelUsageConfigItem(
                            name=model,
                            temperature=None,
                            max_tokens=None,
                            max_retry=None,
                        )
                    )
                else:
                    logger.error(
                        f"子任务 {task_name} 的模型配置不合法，请检查配置文件。"
                    )
                    raise ValueError(
                        f"子任务 {task_name} 的模型配置不合法，请检查配置文件。"
                    )
        elif isinstance(item, str):
            usage.append(
                ModelUsageConfigItem(
                    name=item,
                    temperature=None,
                    max_tokens=None,
                    max_retry=None,
                )
            )

        config.task_model_usage_map[task_name] = ModelUsageConfig(
            name=task_name,
            usage=usage,
        )


def load_config(config_path: str) -> ModuleConfig:
    """从TOML配置文件加载配置"""
    config = ModuleConfig()

    include_configs = {
        "request_conf": {
            "func": _request_conf,
            "support": ">=0.0.0",
            "necessary": False,
        },
        "api_providers": {"func": _api_providers, "support": ">=0.0.0"},
        "models": {"func": _models, "support": ">=0.0.0"},
        "task_model_usage": {"func": _task_model_usage, "support": ">=0.0.0"},
    }

    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            try:
                toml_dict = tomli.load(f)
            except tomli.TOMLDecodeError as e:
                logger.critical(
                    f"配置文件model_list.toml填写有误，请检查第{e.lineno}行第{e.colno}处：{e.msg}"
                )
                exit(1)

        # 获取配置文件版本
        config.INNER_VERSION = _get_config_version(toml_dict)

        # 检查版本
        if config.INNER_VERSION > Version(NEWEST_VER):
            logger.warning(
                f"当前配置文件版本 {config.INNER_VERSION} 高于支持的最新版本 {NEWEST_VER}，可能导致异常，建议更新依赖。"
            )

        # 解析配置文件
        # 如果在配置中找到了需要的项，调用对应项的闭包函数处理
        for key in include_configs:
            if key in toml_dict:
                group_specifier_set: SpecifierSet = SpecifierSet(
                    include_configs[key]["support"]
                )

                # 检查配置文件版本是否在支持范围内
                if config.INNER_VERSION in group_specifier_set:
                    # 如果版本在支持范围内，检查是否存在通知
                    if "notice" in include_configs[key]:
                        logger.warning(include_configs[key]["notice"])
                    # 调用闭包函数处理配置
                    include_configs[key]["func"](toml_dict, config)
                else:
                    # 如果版本不在支持范围内，崩溃并提示用户
                    logger.error(
                        f"配置文件中的 '{key}' 字段的版本 ({config.INNER_VERSION}) 不在支持范围内。\n"
                        f"当前程序仅支持以下版本范围: {group_specifier_set}"
                    )
                    raise InvalidVersion(
                        f"当前程序仅支持以下版本范围: {group_specifier_set}"
                    )

            # 如果 necessary 项目存在，而且显式声明是 False，进入特殊处理
            elif (
                "necessary" in include_configs[key]
                and include_configs[key].get("necessary") is False
            ):
                # 通过 pass 处理的项虽然直接忽略也是可以的，但是为了不增加理解困难，依然需要在这里显式处理
                if key == "keywords_reaction":
                    pass
            else:
                # 如果用户根本没有需要的配置项，提示缺少配置
                logger.error(f"配置文件中缺少必需的字段: '{key}'")
                raise KeyError(f"配置文件中缺少必需的字段: '{key}'")

        logger.success(f"成功加载配置文件: {config_path}")

    return config
