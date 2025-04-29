import asyncio

from .base_client import BaseClient, APIResponse
from .. import _logger as logger
from ..config.config import ModelInfo, ModelUsageConfigItem, RequestConfig
from ..exceptions import (
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from ..model_manager import ModelManager
from ..payload_content.message import Message
from ..payload_content.tool_option import ToolOption
from ..usage_statistic import ModelUsageStatistic, UsageCallStatus
from ..utils import compress_messages


def default_exception_handler(
    e: Exception,
    task_name: str,
    model_name: str,
    remain_try: int,
    retry_interval: int = 10,
    messages: tuple[list[Message], bool] | None = None,
) -> tuple[int, list[Message] | None]:
    """
    默认异常处理函数
    :param e: 异常对象
    :param task_name: 任务名称
    :param model_name: 模型名称
    :param remain_try: 剩余尝试次数
    :param retry_interval: 重试间隔
    :param messages: (消息列表, 是否已压缩过)
    :return (等待间隔（如果为0则不等待，为-1则不再请求该模型）, 新的消息列表（适用于压缩消息）)
    """

    if isinstance(e, NetworkConnectionError):  # 网络连接错误
        if remain_try > 0:
            # 还有重试机会
            logger.warning(
                f"任务-'{task_name}' 模型-'{model_name}'\n"
                f"连接异常，将于{retry_interval}秒后重试"
            )
            return retry_interval, None
        else:
            # 达到最大重试次数
            logger.error(
                f"任务-'{task_name}' 模型-'{model_name}'"
                f"连接异常，超过最大重试次数，请检查网络连接状态或URL是否正确"
            )
            return -1, None  # 不再重试请求该模型
    elif isinstance(e, ReqAbortException):
        # 请求被中断
        # TODO: 流式输出模式适配
        logger.warning(f"任务-'{task_name}' 模型-'{model_name}'\n请求被中断")
        return -1, None  # 不再重试请求该模型
    elif isinstance(e, RespNotOkException):
        # 响应错误
        if e.status_code in [400, 401, 402, 403, 404]:
            # 客户端错误
            logger.error(
                f"任务-'{task_name}' 模型-'{model_name}'\n"
                f"请求失败，错误代码-{e.status_code}，错误信息-{e.message}"
            )
            return -1, None  # 不再重试请求该模型
        elif e.status_code == 413:
            # 请求体过大
            if messages:
                if not messages[1]:
                    # 尝试压缩消息
                    logger.warning(
                        f"任务-'{task_name}' 模型-'{model_name}'\n"
                        "请求体过大，尝试压缩消息后重试"
                    )
                    return 0, compress_messages(messages[0])
                else:
                    logger.error(
                        f"任务-'{task_name}' 模型-'{model_name}'\n"
                        "压缩后消息仍然过大，放弃请求。"
                    )
                    return -1, None  # 不再重试请求该模型
            else:
                # 没有消息可压缩
                logger.error(
                    f"任务-'{task_name}' 模型-'{model_name}'\n"
                    "请求体过大，无法压缩消息，放弃请求。"
                )
                return -1, None
        elif e.status_code == 429:
            # 请求过于频繁
            if remain_try > 0:
                # 还有重试机会
                logger.warning(
                    f"任务-'{task_name}' 模型-'{model_name}'\n"
                    f"请求过于频繁，将于{retry_interval}秒后重试"
                )
                return retry_interval, None
            else:
                # 达到最大重试次数
                logger.error(
                    f"任务-'{task_name}' 模型-'{model_name}'\n"
                    f"请求过于频繁，超过最大重试次数，请稍后再试"
                )
                return -1, None  # 不再重试请求该模型
        elif e.status_code >= 500:
            # 服务器错误
            if remain_try > 0:
                # 还有重试机会
                logger.warning(
                    f"任务-'{task_name}' 模型-'{model_name}'\n"
                    f"服务器错误，将于{retry_interval}秒后重试"
                )
                return retry_interval, None
            else:
                # 达到最大重试次数
                logger.error(
                    f"任务-'{task_name}' 模型-'{model_name}'\n"
                    f"服务器错误，超过最大重试次数，请稍后再试"
                )
                return -1, None  # 不再重试请求该模型
        else:
            # 未知错误
            logger.error(
                f"任务-'{task_name}' 模型-'{model_name}'\n"
                f"未知错误，错误代码-{e.status_code}，错误信息-{e.message}"
            )
            return -1, None
    elif isinstance(e, RespParseException):
        # 响应解析错误
        logger.error(
            f"任务-'{task_name}' 模型-'{model_name}'\n"
            f"响应解析错误，错误信息-{e.message}\n"
        )
        logger.debug(f"响应内容:\n{str(e.resp)}")
        return -1, None  # 不再重试请求该模型
    else:
        logger.error(
            f"任务-'{task_name}' 模型-'{model_name}'\n未知异常，错误信息-{str(e)}"
        )
        return -1, None  # 不再重试请求该模型


class ModelRequestHandler:
    """
    模型请求处理器
    """

    task_name: str  # 任务名称
    client_map: dict[str, BaseClient]  # 客户端列表
    configs: list[(ModelInfo, ModelUsageConfigItem)]  # 模型使用配置
    usage_statistic: ModelUsageStatistic  # 任务的使用统计信息
    req_conf: RequestConfig

    def __init__(
        self,
        task_name: str,
        manager: ModelManager,
    ):
        self.task_name = task_name
        self.client_map = {}
        self.configs = []
        self.usage_statistic = manager.usage_statistic
        self.req_conf = manager.config.req_conf

        # 获取模型与使用配置
        for model_usage in manager.config.task_model_usage_map[task_name].usage:
            if model_usage.name not in manager.config.models:
                logger.error(f"Model '{model_usage.name}' not found in ModelManager")
                raise KeyError(f"Model '{model_usage.name}' not found in ModelManager")
            model_info = manager.config.models[model_usage.name]

            if model_info.api_provider not in self.client_map:
                # 缓存API客户端
                self.client_map[model_info.api_provider] = manager.api_client_map[
                    model_info.api_provider
                ]

            self.configs.append((model_info, model_usage))  # 添加模型与使用配置

    async def get_response(
        self,
        messages: list[Message],
        tool_options: list[ToolOption] = None,
        response_format: dict | None = None,  # 暂不启用
        stream_response_handler: callable = None,
        async_response_parser: callable = None,
    ) -> APIResponse | None:
        """
        获取对话响应
        :return:
        """
        # 遍历可用模型，若获取响应失败，则使用下一个模型继续请求
        for config_item in self.configs:
            model_info: ModelInfo = config_item[0]
            model_usage_config: ModelUsageConfigItem = config_item[1]
            client = self.client_map[model_info.api_provider]

            remain_try = (
                model_usage_config.max_retry or self.req_conf.max_retry
            ) + 1  # 初始化：剩余尝试次数 = 最大重试次数 + 1

            compressed_messages = None
            while remain_try > 0:
                record_id: str | None = None
                try:
                    # 创建响应记录
                    record_id = self.usage_statistic.create_usage(
                        model_name=model_info.name,
                        task_name=self.task_name,
                    )
                    # 获取响应
                    response = await client.get_response(
                        model_info,
                        message_list=(
                            compressed_messages if compressed_messages else messages
                        ),
                        tool_options=tool_options,
                        max_tokens=model_usage_config.max_tokens
                        if model_usage_config.max_tokens
                        else self.req_conf.default_max_tokens,
                        temperature=model_usage_config.temperature
                        if model_usage_config.temperature
                        else self.req_conf.default_temperature,
                        response_format=response_format,
                        stream_response_handler=stream_response_handler,
                        async_response_parser=async_response_parser,
                    )
                    # 统计usage
                    if response.usage is not None:
                        # 记录模型使用情况
                        self.usage_statistic.update_usage(
                            record_id=record_id,
                            model_info=model_info,
                            usage_data=response.usage,
                        )
                    return response
                except Exception as e:
                    logger.trace(e)
                    remain_try -= 1  # 剩余尝试次数减1

                    # 若有RecordID，更新模型使用情况
                    if record_id:
                        self.usage_statistic.update_usage(
                            record_id=record_id,
                            model_info=model_info,
                            stat=UsageCallStatus.FAILURE,
                            ext_msg=str(e),
                        )

                    # 处理异常
                    handle_res = default_exception_handler(
                        e,
                        self.task_name,
                        model_info.name,
                        remain_try,
                        retry_interval=self.req_conf.retry_interval,
                        messages=(messages, compressed_messages is not None),
                    )

                    if handle_res[0] == -1:
                        # 等待间隔为-1，表示不再请求该模型
                        remain_try = 0
                    elif handle_res[0] != 0:
                        # 等待间隔不为0，表示需要等待
                        await asyncio.sleep(handle_res[0])
                    if handle_res[1] is not None:
                        # 压缩消息
                        compressed_messages = handle_res[1]

        logger.error(f"任务-'{self.task_name}' 请求执行失败，所有模型均不可用")
        return None  # 所有请求尝试均失败

    async def get_embedding(
        self,
        embedding_input: str,
    ) -> APIResponse | None:
        """
        获取嵌入向量
        :return:
        """
        for config in self.configs:
            model_info: ModelInfo = config[0]
            model_usage_config: ModelUsageConfigItem = config[1]
            client = self.client_map[model_info.api_provider]
            remain_try = (
                model_usage_config.max_retry or self.req_conf.max_retry
            ) + 1  # 初始化：剩余尝试次数 = 最大重试次数 + 1

            while remain_try:
                record_id: str | None = None
                try:
                    # 创建响应记录
                    record_id = self.usage_statistic.create_usage(
                        model_name=model_info.name,
                        task_name=self.task_name,
                    )
                    # 获取嵌入向量
                    embedding = await client.get_embedding(
                        model_info=model_info,
                        embedding_input=embedding_input,
                    )
                    return embedding
                except Exception as e:
                    logger.trace(e)
                    remain_try -= 1  # 剩余尝试次数减1

                    # 若有RecordID，更新模型使用情况
                    if record_id:
                        self.usage_statistic.update_usage(
                            record_id=record_id,
                            model_info=model_info,
                            stat=UsageCallStatus.FAILURE,
                            ext_msg=str(e),
                        )

                    # 处理异常
                    handle_res = default_exception_handler(
                        e,
                        self.task_name,
                        model_info.name,
                        remain_try,
                        retry_interval=self.req_conf.retry_interval,
                    )

                    if handle_res[0] == -1:
                        # 等待间隔为-1，表示不再请求该模型
                        remain_try = 0
                    elif handle_res[0] != 0:
                        # 等待间隔不为0，表示需要等待
                        await asyncio.sleep(handle_res[0])

        logger.error(f"任务-'{self.task_name}' 请求执行失败，所有模型均不可用")
        return None  # 所有请求尝试均失败
