from src.maibot_llmreq.payload_content.tool_option import (
    ToolOptionBuilder,
    ToolParamType,
)


class TestToolOptionBuilder:
    def test_set_name_raises_error_for_empty_name(self):
        builder = ToolOptionBuilder()
        try:
            builder.set_name("")
            assert False
        except ValueError as e:
            assert str(e) == "工具名称不能为空"

    def test_set_description_raises_error_for_empty_description(self):
        builder = ToolOptionBuilder()
        try:
            builder.set_description("")
            assert False
        except ValueError as e:
            assert str(e) == "工具描述不能为空"

    def test_add_param_raises_error_for_empty_name_or_description(self):
        builder = ToolOptionBuilder()
        try:
            builder.add_param("", ToolParamType.String, "Valid description", True)
            assert False
        except ValueError as e:
            assert str(e) == "参数名称/描述不能为空"

        try:
            builder.add_param("ValidName", ToolParamType.String, "", True)
            assert False
        except ValueError as e:
            assert str(e) == "参数名称/描述不能为空"

    def test_build_raises_error_when_name_or_description_is_empty(self):
        builder = ToolOptionBuilder()
        try:
            builder.build()
            assert False
        except ValueError as e:
            assert str(e) == "工具名称/描述不能为空"

    def test_build_creates_tool_option_with_valid_data(self):
        builder = ToolOptionBuilder()
        tool = (
            builder.set_name("ToolName")
            .set_description("ToolDescription")
            .add_param("param1", ToolParamType.String, "A string parameter", True)
            .add_param("param2", ToolParamType.Int, "An integer parameter", False)
            .build()
        )
        assert tool.name == "ToolName"
        assert tool.description == "ToolDescription"
        assert len(tool.params) == 2
        assert tool.params[0].name == "param1"
        assert tool.params[0].param_type == ToolParamType.String
        assert tool.params[0].required is True
        assert tool.params[1].name == "param2"
        assert tool.params[1].param_type == ToolParamType.Int
        assert tool.params[1].required is False

    def test_build_creates_tool_option_without_params(self):
        builder = ToolOptionBuilder()
        tool = builder.set_name("ToolName").set_description("ToolDescription").build()
        assert tool.name == "ToolName"
        assert tool.description == "ToolDescription"
        assert tool.params is None
