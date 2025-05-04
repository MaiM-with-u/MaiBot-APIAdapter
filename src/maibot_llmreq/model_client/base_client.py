import asyncio
from typing import Callable

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from ..config.config import ModelInfo, APIProvider
from ..payload_content.message import Message
from ..payload_content.tool_option import ToolOption, ToolCall


class APIResponse:
    """
    API响应类
    """

    content: str | None  # 响应内容
    reasoning_content: str | None  # 推理内容
    tool_calls: list[ToolCall] | None  # 工具调用 [(工具名称, 工具参数), ...]
    embedding: list[float] | None  # 嵌入向量
    usage: (
        tuple[int, int, int] | None
    )  # 使用情况 (prompt_tokens, completion_tokens, total_tokens)
    raw_data: any  # 原始数据

    def __init__(self):
        self.content = None
        self.reasoning_content = None
        self.tool_calls = None
        self.embedding = None
        self.usage = None
        self.raw_data = None


class BaseClient:
    """
    基础客户端
    """

    api_provider: APIProvider

    def __init__(self, api_provider: APIProvider):
        self.api_provider = api_provider

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict | None = None,
        stream_response_handler: Callable[
            [AsyncStream[ChatCompletionChunk], asyncio.Event | None], APIResponse
        ]
        | None = None,
        async_response_parser: Callable[[ChatCompletion], APIResponse] | None = None,
        interrupt_flag: asyncio.Event | None = None,
    ) -> APIResponse:
        """
        获取对话响应
        :param model_info: 模型信息
        :param message_list: 对话体
        :param tool_options: 工具选项（可选，默认为None）
        :param max_tokens: 最大token数（可选，默认为1024）
        :param temperature: 温度（可选，默认为0.7）
        :param response_format: 响应格式（可选，默认为 NotGiven ）
        :param stream_response_handler: 流式响应处理函数（可选）
        :param async_response_parser: 响应解析函数（可选）
        :param interrupt_flag: 中断信号量（可选，默认为None）
        :return: (响应文本, 推理文本, 工具调用, 其他数据)
        """
        raise RuntimeError("This method should be overridden in subclasses")

    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str,
    ) -> APIResponse:
        """
        获取文本嵌入
        :param model_info: 模型信息
        :param embedding_input: 嵌入输入文本
        :return: 嵌入响应
        """
        raise RuntimeError("This method should be overridden in subclasses")
