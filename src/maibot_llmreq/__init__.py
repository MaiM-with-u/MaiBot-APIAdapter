import loguru

_logger: loguru.Logger = loguru.logger


def init_logger(
    logger: loguru.Logger | None = None,
):
    """
    对LLMRequest模块进行配置
    :param logger: 日志对象
    """
    global _logger  # 申明使用全局变量
    if logger:
        _logger = logger
    else:
        _logger.warning("Warning: No logger provided, using default logger.")
