from .config.config import ModuleConfig

import loguru

from .config.parser import load_config

_logger: loguru.Logger = loguru.logger
_module_config: ModuleConfig | None = None


def init(
    logger: loguru.Logger | None = None,
    module_config_path: str | None = None,
):
    """
    对LLMRequest模块进行配置
    :param logger: 日志对象
    :param module_config_path: 配置文件路径
    """
    global _logger, _module_config
    if _logger:
        _logger = logger
    else:
        _logger.warning("Warning: No logger provided, using default logger.")

    if module_config_path:
        _module_config = load_config(config_path=module_config_path, logger=logger)
    else:
        _logger.warning("Warning: No config path provided, using default config path.")
        _module_config = load_config(
            config_path="config/model_list.toml", logger=logger
        )
