import json
import re

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    NOT_GIVEN,
    AsyncStream,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base_client import APIResponse
from ..config.config import ModelInfo, APIProvider
from . import BaseClient

from ..exceptions import (
    RespParseException,
    NetworkConnectionError,
    RespNotOkException,
)
from ..payload_content.message import Message, RoleType
from ..payload_content.tool_option import ToolOption, ToolParam, ToolCall


def _convert_messages(messages: list[Message]) -> list[dict]:
    """
    转换消息格式 - 将消息转换为OpenAI API所需的格式
    :param messages: 消息列表
    :return: 转换后的消息列表
    """

    def _convert_message_item(message: Message) -> dict:
        """
        转换单个消息格式
        :param message: 消息对象
        :return: 转换后的消息字典
        """
        ret = {
            "role": message.role.value,
        }

        # 添加Content
        content: str | list[dict]
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = []
            for item in message.content:
                if isinstance(item, tuple):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{item[0].lower()};base64,{item[1]}"
                            },
                        }
                    )
                elif isinstance(item, str):
                    content.append({"type": "text", "text": item})
        else:
            raise RuntimeError("无法触及的代码：请使用MessageBuilder类构建消息对象")

        ret["content"] = content

        # 添加工具调用ID
        if message.role == RoleType.Tool:
            if not message.tool_call_id:
                raise ValueError("无法触及的代码：请使用MessageBuilder类构建消息对象")
            ret["tool_call_id"] = message.tool_call_id

        return ret

    return [_convert_message_item(message) for message in messages]


def _convert_tool_options(tool_options: list[ToolOption]) -> list[dict]:
    """
    转换工具选项格式 - 将工具选项转换为OpenAI API所需的格式
    :param tool_options: 工具选项列表
    :return: 转换后的工具选项列表
    """

    def _convert_tool_param(tool_option_param: ToolParam) -> dict:
        """
        转换单个工具参数格式
        :param tool_option_param: 工具参数对象
        :return: 转换后的工具参数字典
        """
        return {
            "type": tool_option_param.param_type.value,
            "description": tool_option_param.description,
        }

    def _convert_tool_option_item(tool_option: ToolOption) -> dict:
        """
        转换单个工具项格式
        :param tool_option: 工具选项对象
        :return: 转换后的工具选项字典
        """
        ret = {
            "name": tool_option.name,
            "description": tool_option.description,
        }
        if tool_option.params:
            ret["parameters"] = {
                "type": "object",
                "properties": {
                    param.name: _convert_tool_param(param)
                    for param in tool_option.params
                },
                "required": [
                    param.name for param in tool_option.params if param.required
                ],
            }
        return ret

    return [_convert_tool_option_item(tool_option) for tool_option in tool_options]


pattern = re.compile(
    r"<think>(?P<think>.*?)</think>(?P<content>.*)|<think>(?P<think_unclosed>.*)|(?P<content_only>.+)",
    re.DOTALL,
)
"""用于解析推理内容的正则表达式"""


def _default_stream_response_handler(
    resp: AsyncStream[ChatCompletionChunk],
) -> APIResponse:
    """
    流式响应处理函数 - 处理OpenAI API的流式响应
    :param resp: 流式响应对象
    :return: APIResponse对象
    """

    # TODO: 实现流式输出模式

    def _default_stream_event_handler():
        pass

    raise RuntimeError("流式输出模式尚未实现")


def _default_async_response_parser(resp: ChatCompletion) -> APIResponse:
    """
    解析对话补全响应 - 将OpenAI API响应解析为APIResponse对象
    :param resp: 响应对象
    :return: APIResponse对象
    """
    api_response = APIResponse()

    if not hasattr(resp, "choices") or len(resp.choices) == 0:
        raise RespParseException(resp, "响应解析失败，缺失choices字段")
    message_part = resp.choices[0].message

    # 检查是否有单独的推理字段
    if hasattr(message_part, "reasoning_content"):
        api_response.content = message_part.content
        api_response.reasoning_content = message_part.reasoning_content
    else:
        # 提取推理和内容
        match = pattern.match(message_part.content)
        if not match:
            raise RespParseException(resp, "响应解析失败，无法捕获推理内容和输出内容")
        if match.group("think") is not None:
            result = match.group("think").strip(), match.group("content").strip()
        elif match.group("think_unclosed") is not None:
            result = match.group("think_unclosed").strip(), None
        else:
            result = None, match.group("content_only").strip()
        api_response.reasoning_content, api_response.content = result

    # 提取工具调用
    if hasattr(message_part, "tool_calls"):
        api_response.tool_calls = []
        for call in message_part.tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
                api_response.tool_calls.append(
                    ToolCall(call.id, call.function.name, arguments)
                )
            except json.JSONDecodeError:
                raise RespParseException(resp, "响应解析失败，无法解析工具调用参数")

    # 提取Usage信息
    if hasattr(resp, "usage"):
        api_response.usage = (
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            resp.usage.total_tokens,
        )

    # 将原始响应存储在原始数据中
    api_response.raw_data = resp

    return api_response


class OpenaiClient(BaseClient):
    client: AsyncOpenAI

    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)
        self.client = AsyncOpenAI(
            base_url=api_provider.base_url,
            api_key=api_provider.api_key,
            max_retries=0,
        )

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict | None = None,
        stream_response_handler: callable = None,
        async_response_parser: callable = None,
    ) -> APIResponse:
        """
        获取对话响应
        :param model_info: 模型信息
        :param message_list: 对话体
        :param tool_options: 工具选项（可选，默认为None）
        :param max_tokens: 最大token数（可选，默认为1024）
        :param temperature: 温度（可选，默认为0.7）
        :param response_format: 响应格式（可选，默认为 NotGiven ）
        :param stream_response_handler: 流式响应处理函数（可选，默认为default_stream_response_handler）
        :param async_response_parser: 响应解析函数（可选，默认为default_response_parser）
        :return: (响应文本, 推理文本, 工具调用, 其他数据)
        """
        if async_response_parser is None:
            async_response_parser = _default_async_response_parser

        if stream_response_handler is None:
            stream_response_handler = _default_stream_response_handler

        # 将messages构造为OpenAI API所需的格式
        messages = _convert_messages(message_list)
        # 将tool_options转换为OpenAI API所需的格式
        tools = _convert_tool_options(tool_options) if tool_options else None

        if model_info.force_stream_mode:
            try:
                response = stream_response_handler(
                    await self.client.chat.completions.create(
                        model=model_info.model_identifier,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        response_format=response_format
                        if response_format
                        else NOT_GIVEN,
                    )
                )
                return response
            except Exception:
                raise  # TODO: 完善流式输出模式的异常处理
        else:
            try:
                response = async_response_parser(
                    await self.client.chat.completions.create(
                        model=model_info.model_identifier,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                        response_format=response_format
                        if response_format
                        else NOT_GIVEN,
                    )
                )
                return response
            except APIConnectionError:
                raise NetworkConnectionError()
            except APIStatusError as e:
                # 重封装APIError为RespNotOkException
                raise RespNotOkException(e.status_code, e.message)

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
        try:
            raw_response = await self.client.embeddings.create(
                model=model_info.model_identifier,
                input=embedding_input,
            )
        except APIConnectionError:
            raise NetworkConnectionError()
        except APIStatusError as e:
            # 重封装APIError为RespNotOkException
            raise RespNotOkException(e.status_code)

        response = APIResponse()

        # 解析嵌入响应
        if hasattr(raw_response, "data"):
            response.embedding = raw_response.data[0].embedding
        else:
            raise RespParseException(raw_response, "响应解析失败，缺失data字段")

        # 解析使用情况
        if hasattr(raw_response, "usage"):
            response.usage = (
                raw_response.usage.prompt_tokens,
                raw_response.usage.completion_tokens,
                raw_response.usage.total_tokens,
            )

        return response
