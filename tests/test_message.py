from src.maibot_llmreq.payload_content.message import MessageBuilder, RoleType


class TestMessageBuilder:
    def test_set_role_updates_role_correctly(self):
        builder = (
            MessageBuilder().set_role(RoleType.Assistant).add_text_content("Hello")
        )
        message = builder.build()
        assert message.role == RoleType.Assistant

    def test_add_text_content_appends_text(self):
        builder = MessageBuilder()
        builder.add_text_content("Hello")
        message = builder.build()
        assert message.content == "Hello"

    def test_add_image_content_appends_image(self):
        builder = MessageBuilder()
        builder.add_image_content("png", "base64data")
        message = builder.build()
        assert message.content == [("png", "base64data")]

    def test_add_image_content_raises_error_for_empty_base64(self):
        builder = MessageBuilder()
        try:
            builder.add_image_content("png", "")
            assert False
        except ValueError as e:
            assert str(e) == "图片的base64编码不能为空"

    def test_add_image_content_raises_error_for_unsupported_format(self):
        builder = MessageBuilder()
        try:
            builder.add_image_content("bmp", "base64data")
            assert False
        except ValueError as e:
            assert str(e) == "不受支持的图片格式"

    def test_add_tool_call_requires_tool_role(self):
        builder = MessageBuilder()
        try:
            builder.add_tool_call("call_123")
            assert False
        except ValueError as e:
            assert str(e) == "仅当角色为Tool时才能添加工具调用ID"

    def test_add_tool_call_raises_error_when_tool_call_id_is_empty(self):
        builder = MessageBuilder().set_role(RoleType.Tool)
        builder.add_text_content("Tool message")
        try:
            builder.add_tool_call("")
            assert False
        except ValueError as e:
            assert str(e) == "工具调用ID不能为空"

        try:
            builder.build()
            assert False
        except ValueError as e:
            assert str(e) == "Tool角色的工具调用ID不能为空"

    def test_build_raises_error_when_content_empty(self):
        builder = MessageBuilder()
        try:
            builder.build()
            assert False
        except ValueError as e:
            assert str(e) == "内容不能为空"

    def test_build_includes_tool_call_id_when_role_is_tool(self):
        builder = MessageBuilder()
        builder.set_role(RoleType.Tool).add_tool_call("call_123").add_text_content(
            "Tool message"
        )
        message = builder.build()
        assert message.tool_call_id == "call_123"
        assert message.role == RoleType.Tool
        assert message.content == "Tool message"

    def test_build_raises_error_when_tool_call_id_missing_for_tool_role(self):
        builder = MessageBuilder()
        builder.set_role(RoleType.Tool).add_text_content("Tool message")
        try:
            builder.build()
            assert False
        except ValueError as e:
            assert str(e) == "Tool角色的工具调用ID不能为空"

    def test_build_handles_mixed_content_correctly(self):
        builder = MessageBuilder()
        builder.add_text_content("Text").add_image_content("jpg", "base64data")
        message = builder.build()
        assert message.content == ["Text", ("jpg", "base64data")]
