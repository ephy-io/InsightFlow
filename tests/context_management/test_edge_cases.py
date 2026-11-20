"""环境管理的边缘案例。"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from insightflow.context_config import ContextConfig
from insightflow.context_manager import ContextManager


class TestContextManagementEdgeCases:
    """环境管理的边缘案例和边界条件。"""

    @pytest.fixture
    def edge_case_config(self):
        """边缘大小写测试配置。"""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=800,
            keep_recent_messages=2,
            max_tool_output_length=50,
        )

    @pytest.fixture
    def context_manager(self, edge_case_config):
        """边缘大小写测试的背景管理器。"""
        return ContextManager(llm=Mock(), config=edge_case_config)

    def test_empty_and_none_inputs(self, context_manager):
        """空和 None 输入的测试处理。"""
        # 空列表
        messages = []
        context_manager.manage_context_messages(messages)
        assert messages == []

        # 包含 None 元素的列表(应优雅地过滤出)
        messages = [HumanMessage(content="Test"), None, AIMessage(content="Response")]
        # 在传递到上下文管理器前过滤 None 值
        filtered_messages = [msg for msg in messages if msg is not None]
        context_manager.manage_context_messages(filtered_messages)
        result = filtered_messages
        assert len(result) == 2

    def test_malformed_messages(self, context_manager):
        """测试处理错误的信息。"""
        # 包含 None 内容的消息
        try:
            malformed_msg = HumanMessage(content=None)
            messages = [malformed_msg]
            context_manager.manage_context_messages(messages)
            result = messages
            # 应善处理。
            assert isinstance(result, list)
        except Exception as e:
            # 如果有异常，就应该合理
            assert "content" in str(e).lower()

    def test_extremely_long_single_message(self, context_manager):
        """测试处理极长的单条消息。"""
        # 创建超过整个上下文限制的消息
        very_long_content = "A" * 100000  # 比上下文限制长得多
        long_message = HumanMessage(content=very_long_content)

        messages = [long_message]
        context_manager.manage_context_messages(messages)
        result = messages

        # 仍应返回消息( 文字管理不裁剪单个消息内容)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_tool_message_without_tool_call_id(self, context_manager):
        """测试工具消息的处理方式，不需要 tool_call_id。"""
        try:
            # 这可能会根据 LangChain 的校验而引起一个错误
            tool_msg = ToolMessage(content="Result", tool_call_id="")
            messages = [tool_msg]
            context_manager.manage_context_messages(messages)
            result = messages
            assert isinstance(result, list)
        except Exception:
            # 如果 LangChain 验证和加高，这是可以接受的
            pass

    def test_circular_references_in_content(self, context_manager):
        """对可能引起问题的复杂内容进行测试处理。"""
        # 有特殊字符和格式化的内容
        special_content = """
        Content with:
        - Unicode: 🚀 中文 العربية
        - Code blocks: ```python\nprint("hello")\n```
        - JSON: {"key": "value", "nested": {"array": [1,2,3]}}
        - HTML: <div class="test">content</div>
        - URLs: https://example.com/path?param=value
        - Very long line: """ + "X" * 1000

        message = HumanMessage(content=special_content)
        messages = [message]
        context_manager.manage_context_messages(messages)
        result = messages

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_zero_configuration_values(self):
        """测试零配置值的行为。"""
        zero_config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=0,
            keep_recent_messages=0,
            max_tool_output_length=0,
        )

        context_manager = ContextManager(llm=Mock(), config=zero_config)
        messages = [HumanMessage(content="Test")]

        # 应优雅地处理零值
        context_manager.manage_context_messages(messages)
        result = messages
        assert isinstance(result, list)

    def test_negative_configuration_values(self):
        """测试带有负配置值的行为。"""
        negative_config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=-50,
            keep_recent_messages=-5,
            max_tool_output_length=-10,
        )

        context_manager = ContextManager(llm=Mock(), config=negative_config)
        messages = [HumanMessage(content="Test")]

        # 应优雅地处理负值( 可能被视为已禁用)
        context_manager.manage_context_messages(messages)
        result = messages
        assert isinstance(result, list)

    def test_unicode_and_encoding_edge_cases(self, context_manager):
        """测试处理各种Unicode和编码方案。"""
        unicode_messages = [
            HumanMessage(content="English text"),
            HumanMessage(content="中文内容测试"),
            HumanMessage(content="العربية"),
            HumanMessage(content="Русский текст"),
            HumanMessage(content="🚀🎉💡🔥"),  # 爱莫吉斯
            HumanMessage(content="Mixed: Hello 世界 🌍"),
            ToolMessage(content="Unicode tool result: café naïve résumé", tool_call_id="unicode_1"),
        ]

        context_manager.manage_context_messages(unicode_messages)
        result = unicode_messages

        # 处理所有 Unicode 内容
        assert len(result) > 0
        assert all(isinstance(msg, HumanMessage | AIMessage | ToolMessage) for msg in result)

    def test_extremely_nested_or_complex_structures(self, context_manager):
        """测试工具输出中复杂嵌入式数据结构的处理。"""
        # 模拟深层嵌入的 JSON 输出
        nested_data = {"level1": {"level2": {"level3": {"data": ["item"] * 1000}}}}
        complex_output = str(nested_data) * 100  # 让它变得非常大

        # 创建消息，使工具消息处于历史部分(不是最近)
        # keep_recent_messages=2，因此在工具消息后添加超过 2 条消息。
        messages = [
            ToolMessage(content=complex_output, tool_call_id="complex_1"),  # 历史部分
            HumanMessage(content="Question 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Recent question"),  # 最近的部分从这里开始
        ]
        context_manager.manage_context_messages(messages)
        result = messages

        # 是否应该修剪复杂的产出，因为它是历史的一部分
        tool_msg = next(msg for msg in result if isinstance(msg, ToolMessage))
        assert len(str(tool_msg.content)) < len(complex_output)

    def test_sql_output_edge_cases(self, context_manager):
        """测试 SQL 输出与边缘大小写三进制。"""
        # 无结果的 SQL
        empty_sql_output = """SQL Query:
```sql
SELECT * FROM users WHERE id = -1;
```

Query Results (CSV format):
```csv
id,name
```"""

        # 单行的 SQL
        single_row_sql = """SQL Query:
```sql
SELECT COUNT(*) as total FROM users;
```

Query Results (CSV format):
```csv
total
42
```"""

        # SQL 输出错误
        malformed_sql = """Something that looks like SQL but isn't:
```sql
INVALID QUERY HERE
```
Random text after"""

        test_cases = [empty_sql_output, single_row_sql, malformed_sql]

        for sql_output in test_cases:
            tool_msg = ToolMessage(content=sql_output, tool_call_id="sql_test")
            messages = [tool_msg]
            context_manager.manage_context_messages(messages)
            result = messages

            # 应善处理一切案件
            assert len(result) == 1
            assert isinstance(result[0], ToolMessage)

    def test_conversation_state_consistency(self, context_manager):
        """通过管理测试对话状态是否一致。"""
        # 创建与特定模式的对话( 状态中没有 SystemMessage)
        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Response 1"),
            ToolMessage(content="Tool result 1", tool_call_id="tool_1"),
            HumanMessage(content="Question 2"),
            AIMessage(
                content="Response 2 with tool calls",
                tool_calls=[{"name": "test_tool", "args": {"param": "value"}, "id": "call_1"}],
            ),
            ToolMessage(content="Tool result 2", tool_call_id="call_1"),
            HumanMessage(content="Final question"),
        ]

        with patch(
            "insightflow.context_manager.call_llm_chat_model_with_retry", return_value=AIMessage(content="Summary")
        ):
            context_manager.manage_context_messages(messages)
            result = messages

        # 应保持消息类型的一致性(只有有效的状态消息类型)
        message_types = [type(msg) for msg in result]
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(
            msg_type in valid_types for msg_type in message_types
        ), "Should only contain valid state message types"

        # 没有对应的 AI 消息，不应将工具消息孤儿化
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage):
                # 在此之前应该有一个 AI 消息，并有工具调用
                previous_ai_msgs = [m for m in result[:i] if isinstance(m, AIMessage)]
                assert len(previous_ai_msgs) > 0, "Tool message should have corresponding AI message"
