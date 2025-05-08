import asyncio
import io
from typing import Callable, Iterator, TypeVar, AsyncIterator, Iterable

from .. import _logger as logger

from google import genai
from google.genai import types
from google.genai.types import FunctionDeclaration, GenerateContentResponse
from google.genai.errors import (
    ClientError,
    ServerError,
    UnknownFunctionCallArgumentError,
    UnsupportedFunctionError,
    FunctionInvocationError,
)

from .base_client import APIResponse
from ..config.config import ModelInfo, APIProvider
from . import BaseClient

from ..exceptions import (
    RespParseException,
    NetworkConnectionError,
    RespNotOkException,
    ReqAbortException,
)
from ..payload_content.message import Message, RoleType
from ..payload_content.tool_option import ToolOption, ToolParam, ToolCall

MODELS_NEED_TRANSFORMATION = ["2.5"]
"""gemini2.5专用转换变量"""

T = TypeVar("T")
"""类型转换用的"""


def _convert_messages(messages: list[Message]) -> tuple[list[dict], str | None]:
    """
    转换消息格式 - 将消息转换为Gemini API所需的格式
    但是我懒得处理如果有一堆system消息的情况了，所以我决定直接把它们合在一起，如果真的有人在引用这个类的时候发了好几条相互矛盾的system消息，那我只能说他的脑子可能有一些问题
    :param messages: 消息列表
    :return: 转换后的消息列表(和可能存在的system消息)
    """

    def _convert_message_item(message: Message) -> dict:
        """
        转换单个消息格式，除了system和tool类型的消息
        :param message: 消息对象
        :return: 转换后的消息字典
        """

        # 将openai格式的角色重命名为gemini格式的角色
        ret = {
            "role": message.role.value,
        }
        if message.role == RoleType.Assistant:
            ret["role"] = "model"

        # 添加Content
        content: str | list
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = []
            for item in message.content:
                if isinstance(item, tuple):
                    content.append(
                        types.Part.from_bytes(
                            data=item[1], mime_type='"image/' + item[0] + '"'
                        )
                    )
                elif isinstance(item, str):
                    content.append(item)
        else:
            raise RuntimeError("无法触及的代码：请使用MessageBuilder类构建消息对象")

        ret["contents"] = content
        return ret

    # 然后我需要在这里确保最多最多只有一个system，如果多了我就合并，tmd
    ret1: list[dict] = []
    system_instructions: str = ""
    for message in messages:
        if message.role == RoleType.System:
            if isinstance(message.content, str):
                system_instructions += message.content
            else:
                raise RuntimeError("你tm怎么往system里面塞图片base64？")
        elif message.role == RoleType.Tool:
            if not message.tool_call_id:
                raise ValueError("无法触及的代码：请使用MessageBuilder类构建消息对象")
            pass
        else:
            ret1.append(_convert_message_item(message))
    if system_instructions != "":
        # 如果有system消息，就把它加上去
        ret2: tuple = (ret1, system_instructions)
    else:
        # 如果没有system消息，就直接返回
        ret2: tuple = (ret1, None)

    return ret2


def _convert_tool_options(tool_options: list[ToolOption]) -> list[FunctionDeclaration]:
    """
    转换工具选项格式 - 将工具选项转换为Gemini API所需的格式
    :param tool_options: 工具选项列表
    :return: 转换后的工具对象列表
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

    def _convert_tool_option_item(tool_option: ToolOption) -> FunctionDeclaration:
        """
        转换单个工具项格式
        :param tool_option: 工具选项对象
        :return: 转换后的Gemini工具选项对象
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
        ret1 = types.FunctionDeclaration(**ret)
        return ret1

    return [_convert_tool_option_item(tool_option) for tool_option in tool_options]


def _process_delta(
    delta: GenerateContentResponse,
    fc_delta_buffer: io.StringIO,
    tool_calls_buffer: list[tuple[str, str, dict]],
):
    try:
        # 接收content,大部分其实和非流式相同
        if delta.candidates:
            candidate = delta.candidates[-1]
            # 鉴定一下怎么结束的,gemini特色了属于是
            finish_reason_enum = candidate.finish_reason
            finish_reason_str = finish_reason_enum.name
            if finish_reason_enum not in [
                types.FinishReason.STOP,
                types.FinishReason.MAX_TOKENS,
                None,
            ]:
                logger.info(f"Gemini生成停止，原因：{finish_reason_str}")
                if finish_reason_enum == types.FinishReason.SAFETY:
                    safety_ratings_str = "，".join(
                        [
                            f"{rating.category.name}: {rating.probability.name}"
                            for rating in candidate.safety_ratings
                        ]
                    )
                    logger.error(
                        f"Gemini安全设置重新思考了回答，安全评分: {safety_ratings_str}"
                    )

            if delta.text:
                fc_delta_buffer.write(delta.text)

            if (
                delta.function_calls
            ):  # 为什么不用hasattr呢，是因为这个属性一定有，即使是个空的
                for call in delta.function_calls:
                    try:
                        if not isinstance(
                            call.args, dict
                        ):  # gemini返回的function call参数就是dict格式的了
                            raise RespParseException(
                                delta, "响应解析失败，工具调用参数无法解析为字典类型"
                            )
                        tool_calls_buffer.append(
                            (
                                call.id,
                                call.name,
                                call.args,
                            )
                        )
                    except Exception:
                        raise RespParseException(
                            delta, "响应解析失败，无法解析工具调用参数"
                        )

        else:  # 基本是被block了
            block_reason = (
                delta.prompt_feedback.block_reason.name
                if delta.prompt_feedback
                else "未知"
            )
            block_reason_msg = (
                delta.prompt_feedback.block_reason_message
                if delta.prompt_feedback
                else "无具体信息"
            )
            logger.warning(
                f"Gemini回复中没有candidate。可能原因: {block_reason}. 信息: {block_reason_msg}"
            )
            if (
                delta.prompt_feedback
                and delta.prompt_feedback.block_reason == types.BlockReason.SAFETY
            ):
                safety_ratings_str = ", ".join(
                    [
                        f"{rating.category.name}: {rating.probability.name}"
                        for rating in delta.prompt_feedback.safety_ratings
                    ]
                )
                logger.error(
                    f"Gemini因Prompt安全问题阻止了请求。安全评分: {safety_ratings_str}"
                )
                raise RespParseException(
                    delta, f"请求因安全原因被阻止 ({safety_ratings_str})"
                )
            raise RespParseException(
                delta, f"未从Gemini收到有效候选回复 (原因: {block_reason})"
            )
    except Exception as e:
        response_str = str(delta) if "delta" in locals() else "响应对象不可用"
        raise RespParseException(
            delta,
            "响应解析失败，response字段异常"
            + str(e)
            + "\n响应对象内容："
            + response_str,
        )


def _build_stream_api_resp(
    _fc_delta_buffer: io.StringIO,
    _tool_calls_buffer: list[tuple[str, str, dict]],
    _usage_record: tuple[int, int, int] | None,
) -> APIResponse:
    resp = APIResponse()

    if _fc_delta_buffer.tell() > 0:
        # 如果正式内容缓冲区不为空，则将其写入APIResponse对象
        resp.content = _fc_delta_buffer.getvalue()
    _fc_delta_buffer.close()
    if len(_tool_calls_buffer) > 0:
        # 如果工具调用缓冲区不为空，则将其解析为ToolCall对象列表
        resp.tool_calls = []
        for call_id, function_name, arguments_buffer in _tool_calls_buffer:
            if arguments_buffer.tell() > 0:
                arguments = arguments_buffer
                if not isinstance(arguments, dict):
                    raise RespParseException(
                        None,
                        "响应解析失败，工具调用参数无法解析为字典类型。工具调用参数原始响应：\n"
                        f"{arguments_buffer}",
                    )
            else:
                arguments = None

            resp.tool_calls.append(ToolCall(call_id, function_name, arguments))
    if _usage_record is not None:
        # 如果使用情况记录不为空，则将其存储在APIResponse对象中
        resp.usage = _usage_record

    return resp


async def _to_async_iterable(iterable: Iterable[T]) -> AsyncIterator[T]:
    """
    将迭代器转换为异步迭代器
    :param iterable: 迭代器对象
    :return: 异步迭代器对象
    """
    for item in iterable:
        await asyncio.sleep(0)
        yield item


async def _default_stream_response_handler(
    resp_stream: Iterator[GenerateContentResponse],
    interrupt_flag: asyncio.Event | None,
) -> APIResponse:
    """
    流式响应处理函数 - 处理Gemini API的流式响应
    :param resp_stream: 流式响应对象,是一个神秘的iterator，我完全不知道这个玩意能不能跑，不过遍历一遍之后它就空了，如果跑不了一点的话可以考虑改成别的东西
    :return: APIResponse对象
    """
    _fc_delta_buffer = io.StringIO()  # 正式内容缓冲区，用于存储接收到的正式内容
    _tool_calls_buffer = []  # 工具调用缓冲区，用于存储接收到的工具调用
    _usage_record = None  # 使用情况记录

    try:
        async for chunk in _to_async_iterable(resp_stream):
            # 检查是否有中断量
            if interrupt_flag and interrupt_flag.is_set():
                # 如果中断量被设置，则抛出ReqAbortException
                raise ReqAbortException("请求被外部信号中断")

            _process_delta(
                chunk,
                _fc_delta_buffer,
                _tool_calls_buffer,
            )

            if chunk.usage_metadata:
                # 如果有使用情况，则将其存储在APIResponse对象中
                _usage_record = (
                    chunk.usage_metadata.prompt_token_count,
                    chunk.usage_metadata.candidates_token_count
                    + chunk.usage_metadata.thoughts_token_count,
                    chunk.usage_metadata.total_token_count,
                )

        return _build_stream_api_resp(
            _fc_delta_buffer,
            _tool_calls_buffer,
            _usage_record,
        )
    finally:
        # 确保缓冲区被关闭
        if _fc_delta_buffer:
            _fc_delta_buffer.close()


def _default_async_response_parser(resp: GenerateContentResponse) -> APIResponse:
    """
    解析对话补全响应 - 将Gemini API响应解析为APIResponse对象
    :param resp: 响应对象
    :return: APIResponse对象
    """
    api_response = APIResponse()

    try:
        if resp.candidates:  # 为什么不用hasattr呢，是因为这个属性一定有，即使是个空的
            candidate = resp.candidates[-1]

            # 鉴定一下怎么结束的,gemini特色了属于是
            finish_reason_enum = candidate.finish_reason
            finish_reason_str = finish_reason_enum.name
            if finish_reason_enum not in [
                types.FinishReason.STOP,
                types.FinishReason.MAX_TOKENS,
            ]:
                logger.info(f"Gemini生成停止，原因：{finish_reason_str}")
                if finish_reason_enum == types.FinishReason.SAFETY:
                    safety_ratings_str = "，".join(
                        [
                            f"{rating.category.name}: {rating.probability.name}"
                            for rating in candidate.safety_ratings
                        ]
                    )
                    logger.error(
                        f"Gemini安全设置重新思考了回答，安全评分: {safety_ratings_str}"
                    )

            # 解析回复文字（如有）
            try:
                api_response.content = (
                    resp.text
                )  # 这里偷个懒直接用genai自带的text属性了
            except ValueError:
                logger.info("响应中没有文本内容，可能只有函数调用。")

            # 解析工具调用（如有）
            if (
                resp.function_calls
            ):  # 为什么不用hasattr呢，是因为这个属性一定有，即使是个空的
                api_response.tool_calls = []
                for call in resp.function_calls:
                    try:
                        if not isinstance(call.args, dict):
                            raise RespParseException(
                                resp, "响应解析失败，工具调用参数无法解析为字典类型"
                            )
                        api_response.tool_calls.append(
                            ToolCall(call.id, call.name, call.args)
                        )
                    except Exception:
                        raise RespParseException(
                            resp, "响应解析失败，无法解析工具调用参数"
                        )

            # 把用量信息加进去
            if resp.usage_metadata:
                api_response.usage = (
                    resp.usage_metadata.prompt_token_count,
                    resp.usage_metadata.candidates_token_count
                    + resp.usage_metadata.thoughts_token_count,
                    # 谷歌这个sb玩意，不仅仅把token分成了candidate和thoughts，还不返回thoughts，更可气的是不仅不返回，还tmd不知道这部分收不收费，如果按照2.5pro来看是得收费了，我的建议是除非你家有钱烧的，否则别用谷歌思维链，钱都不知道那里扣的
                    resp.usage_metadata.total_token_count,
                )
            else:
                logger.warning("Gemini回复中没有用量信息")

            api_response.raw_data = resp

        else:  # 基本是被block了
            block_reason = (
                resp.prompt_feedback.block_reason.name
                if resp.prompt_feedback
                else "未知"
            )
            block_reason_msg = (
                resp.prompt_feedback.block_reason_message
                if resp.prompt_feedback
                else "无具体信息"
            )
            logger.warning(
                f"Gemini回复中没有candidate。可能原因: {block_reason}. 信息: {block_reason_msg}"
            )
            if (
                resp.prompt_feedback
                and resp.prompt_feedback.block_reason == types.BlockReason.SAFETY
            ):
                safety_ratings_str = ", ".join(
                    [
                        f"{rating.category.name}: {rating.probability.name}"
                        for rating in resp.prompt_feedback.safety_ratings
                    ]
                )
                logger.error(
                    f"Gemini因Prompt安全问题阻止了请求。安全评分: {safety_ratings_str}"
                )
                raise RespParseException(
                    resp, f"请求因安全原因被阻止 ({safety_ratings_str})"
                )
            raise RespParseException(
                resp, f"未从Gemini收到有效候选回复 (原因: {block_reason})"
            )

    except Exception as e:
        response_str = str(resp) if "resp" in locals() else "响应对象不可用"
        raise RespParseException(
            resp,
            "响应解析失败，response字段异常"
            + str(e)
            + "\n响应对象内容："
            + response_str,
        )

    return api_response


class GeminiClient(BaseClient):
    client: genai.Client

    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)
        self.client = genai.Client(
            api_key=api_provider.api_key,
        )  # 这里和openai不一样，gemini会自己决定自己是否需要retry

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        thinking_budget: int = 0,
        response_format: dict | None = None,
        stream_response_handler: Callable[
            [Iterator[GenerateContentResponse], asyncio.Event | None], APIResponse
        ]
        | None = None,
        async_response_parser: Callable[[GenerateContentResponse], APIResponse]
        | None = None,
        interrupt_flag: asyncio.Event | None = None,
    ) -> APIResponse:
        """
        获取对话响应
        :param model_info: 模型信息
        :param message_list: 对话体
        :param tool_options: 工具选项（可选，默认为None）
        :param max_tokens: 最大token数（可选，默认为1024）
        :param temperature: 温度（可选，默认为0.7）
        :param thinking_budget: 思考预算（可选，默认为0）
        :param response_format: 响应格式（默认为text/plain,如果是输入的JSON Schema则必须遵守OpenAPI3.0格式,理论上和openai是一样的，暂不支持其它相应格式输入）
        :param stream_response_handler: 流式响应处理函数（可选，默认为default_stream_response_handler）
        :param async_response_parser: 响应解析函数（可选，默认为default_response_parser）
        :param interrupt_flag: 中断信号量（可选，默认为None）
        :return: (响应文本, 推理文本, 工具调用, 其他数据)
        """
        if stream_response_handler is None:
            stream_response_handler = await _default_stream_response_handler

        if async_response_parser is None:
            async_response_parser = _default_async_response_parser

        # 将messages构造为Gemini API所需的格式
        messages = _convert_messages(message_list)
        # 将tool_options转换为Gemini API所需的格式
        tools = _convert_tool_options(tool_options) if tool_options else None
        # 将response_format转换为Gemini API所需的格式
        generation_config_dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_modalities": ["TEXT"],  # 暂时只支持文本输出
        }
        if MODELS_NEED_TRANSFORMATION in model_info.model_identifier.lower():
            # 我偷个懒，在这里识别一下2.5然后开摆，反正现在只有2.5支持思维链，然后我测试之后发现它不返回思考内容，反正我也怕他有朝一日返回了，我决定干掉任何有关的思维内容
            generation_config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=False
            )
        if tools:
            generation_config_dict["tools"] = types.Tool(tools)
        if messages[1]:
            # 如果有system消息，则将其添加到配置中
            generation_config_dict["system_instructions"] = messages[1]
        if response_format:
            generation_config_dict["response_mime_type"] = "application/json"
            generation_config_dict["response_schema"] = response_format

        generation_config = types.GenerateContentConfig(**generation_config_dict)

        if model_info.force_stream_mode:
            try:
                req_task = asyncio.create_task(
                    self.client.models.generate_content_stream(
                        model=model_info.model_identifier,
                        contents=messages[0],
                        config=generation_config,
                    )
                )
                while not req_task.done():
                    if interrupt_flag and interrupt_flag.is_set():
                        # 如果中断量存在且被设置，则取消任务并抛出异常
                        req_task.cancel()
                        raise ReqAbortException("请求被外部信号中断")
                    await asyncio.sleep(0.1)  # 等待0.1秒后再次检查任务&中断信号量状态
                # 和openai一样，不可能流式实时处理的，肯定是等全都处理完了再处理
                return stream_response_handler(req_task.result(), interrupt_flag)
            except (ClientError, ServerError) as e:
                # 重封装ClientError和ServerError为RespNotOkException
                raise RespNotOkException(e.status_code, e.message)
            except (
                UnknownFunctionCallArgumentError,
                UnsupportedFunctionError,
                FunctionInvocationError,
            ) as e:
                # 我翻了翻源代码，我感觉这个异常似乎是在generate之后才会抛出来的，很神秘
                raise ValueError("工具类型错误：请检查工具选项和参数：" + str(e))
            except Exception:
                # 由于genai库没有断网之类的，只能放在这里了
                raise NetworkConnectionError()
        else:
            try:
                # 发送请求并获取响应
                req_task = asyncio.create_task(
                    self.client.models.generate_content(
                        model=model_info.model_identifier,
                        contents=messages[0],
                        config=generation_config,
                    )
                )
                while not req_task.done():
                    if interrupt_flag and interrupt_flag.is_set():
                        # 如果中断量存在且被设置，则取消任务并抛出异常
                        req_task.cancel()
                        raise ReqAbortException("请求被外部信号中断")
                    await asyncio.sleep(0.5)  # 等待0.5秒后再次检查任务&中断信号量状态

                return async_response_parser(req_task.result())
            except (ClientError, ServerError) as e:
                # 重封装ClientError和ServerError为RespNotOkException
                raise RespNotOkException(e.status_code, e.message)
            except (
                UnknownFunctionCallArgumentError,
                UnsupportedFunctionError,
                FunctionInvocationError,
            ) as e:
                raise ValueError("工具类型错误：请检查工具选项和参数：" + str(e))
            except Exception:
                # 由于genai库没有断网之类的，只能放在这里了
                raise NetworkConnectionError()

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
            raw_response: types.EmbedContentResponse = await self.client.aio.models.embed_content(
                model=model_info.model_identifier,
                contents=embedding_input,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY"
                ),  # 我读了一下文档，gemini的嵌入任务类型里面这个理论上最适合LPMM和麦麦https://ai.google.dev/api/embeddings#v1beta.TaskType
            )
        except (ClientError, ServerError) as e:
            # 重封装ClientError和ServerError为RespNotOkException
            raise RespNotOkException(e.status_code)
        except Exception:
            # 由于genai库没有断网之类的，只能放在这里了
            raise NetworkConnectionError()

        response = APIResponse()

        # 解析嵌入响应和使用情况
        if hasattr(raw_response, "embeddings"):
            response.embedding = raw_response.embeddings[0].values
            # google的api不包含使用情况，加上所有的嵌入模型都是免费的，所以我决定用长度来替代算了，本来想用tiktoken但是我突然觉得不用加了,这个地方有点子奇异搞笑的
            response.usage = (
                len(embedding_input),
                0,
                len(embedding_input),
            )
        else:
            raise RespParseException(raw_response, "响应解析失败，缺失embeddings字段")

        return response
