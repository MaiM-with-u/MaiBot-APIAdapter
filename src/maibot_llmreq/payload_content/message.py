from enum import Enum
from typing import Tuple, List


# 设计这系列类的目的是为未来可能的扩展做准备


class RoleType(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "gif"]


class Message:
    # 角色：system、user、assistant、tool
    role: RoleType
    # 内容：提示词、图片与提示词的列表
    content: str | List[Tuple[str, str] | str]
    # 工具调用指令的id
    tool_call_id: str | None

    def __init__(
        self,
        role: RoleType,
        content: str | List[Tuple[str, str] | str],
        tool_call_id: str | None = None,
    ):
        """
        初始化消息对象
        （不应直接修改Message类，而应使用MessageBuilder类来构建对象）
        """
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id


class MessageBuilder:
    __role: RoleType
    __content: List[Tuple[str, str] | str]
    __tool_call_id: str | None

    def __init__(self):
        self.__role = RoleType.User
        self.__content = []
        self.__tool_call_id = None

    def set_role(self, role: RoleType = RoleType.User) -> "MessageBuilder":
        """
        设置角色（默认为User）
        :param role: 角色
        :return: MessageBuilder对象
        """
        self.__role = role
        return self

    def add_text_content(self, text: str) -> "MessageBuilder":
        """
        添加文本内容
        :param text: 文本内容
        :return: MessageBuilder对象
        """
        self.__content.append(text)
        return self

    def add_image_content(
        self, image_format: str, image_base64: str
    ) -> "MessageBuilder":
        """
        添加图片内容
        :param image_format: 图片格式
        :param image_base64: 图片的base64编码
        :return: MessageBuilder对象
        """
        if image_format.lower() not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError("不受支持的图片格式")
        if image_base64 == "":
            raise ValueError("图片的base64编码不能为空")
        self.__content.append((image_format, image_base64))
        return self

    def add_tool_call(self, tool_call_id: str) -> "MessageBuilder":
        """
        添加工具调用指令（调用时请确保已设置为Tool角色）
        :param tool_call_id: 工具调用指令的id
        :return: MessageBuilder对象
        """
        if self.__role != RoleType.Tool:
            raise ValueError("仅当角色为Tool时才能添加工具调用ID")
        if tool_call_id == "":
            raise ValueError("工具调用ID不能为空")
        self.__tool_call_id = tool_call_id
        return self

    def build(self) -> Message:
        """
        构建消息对象
        :return: Message对象
        """
        if len(self.__content) == 0:
            raise ValueError("内容不能为空")
        if self.__role == RoleType.Tool and self.__tool_call_id is None:
            raise ValueError("Tool角色的工具调用ID不能为空")

        message = Message(
            role=self.__role,
            content=(
                self.__content[0]
                if (len(self.__content) == 1 and isinstance(self.__content[0], str))
                else self.__content
            ),
            tool_call_id=self.__tool_call_id,
        )

        return message
