"""基于消息的上下文管理操作测试。"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from insightflow.context_config import ContextConfig
from insightflow.context_manager import ContextManager


class TestMessageBasedContextManagement:
    """测试基于消息的上下文管理并进行直接修改。"""

    @pytest.fixture
    def test_config(self):
        """用于测试消息操作的配置。"""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=300,  # 启动管理的门槛较低
            keep_recent_messages=3,
            max_tool_output_length=200,
            preserve_tool_errors=True,
            preserve_recent_sql=True,
        )

    @pytest.fixture
    def context_manager(self, test_config):
        """测试的背景管理器。"""
        mock_llm = Mock()
        return ContextManager(llm=mock_llm, config=test_config)

    def test_no_operations_when_disabled(self, context_manager):
        """测试在上下文管理被禁用时不执行操作。"""
        context_manager.config.enabled = False

        messages = [HumanMessage(content="Test", id="test_1")]
        original_messages = messages.copy()

        context_manager.manage_context_messages(messages)
        assert messages == original_messages  # 应该不变

    def test_no_operations_when_under_limit(self, context_manager):
        """测试上下文处于符号限制下时不执行操作。"""
        # 无法触发上下文管理的短信息
        messages = [HumanMessage(content="Hi", id="human_1"), AIMessage(content="Hello", id="ai_1")]
        original_messages = messages.copy()

        context_manager.manage_context_messages(messages)
        assert messages == original_messages  # 应该不变

    def test_historical_tool_compression(self, context_manager):
        """测试历史工具消息的压缩。"""
        # 禁用对话摘要来测试只使用工具压缩
        context_manager.config.enable_conversation_summary = False
        context_manager.config.enable_summarization = False

        # 创建带有大型历史工具输出的消息
        messages = [
            HumanMessage(content="Query data", id="human_1"),
            AIMessage(content="Running query", id="ai_1"),
            # 大型历史工具消息(应当压缩)
            ToolMessage(content="A" * 1000, tool_call_id="query_1", id="tool_1_historical"),  # 内容大
            HumanMessage(content="More analysis", id="human_2"),
            AIMessage(content="Analyzing", id="ai_2"),
            # 另一个大型历史工具消息
            ToolMessage(content="B" * 800, tool_call_id="query_2", id="tool_2_historical"),  # 内容大
            # 最近的消息(应保存)
            HumanMessage(content="Recent question", id="human_recent"),
            AIMessage(content="Recent response", id="ai_recent"),
            ToolMessage(content="Recent result", tool_call_id="recent_1", id="tool_recent"),
        ]

        original_count = len(messages)
        context_manager.manage_context_messages(messages)

        # 消息数量应该相同，但有些内容应该压缩
        assert len(messages) == original_count

        # 检查历史工具消息是否被压缩
        historical_tool_msgs = [
            msg
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.id in ["tool_1_historical", "tool_2_historical"]
        ]
        for msg in historical_tool_msgs:
            assert len(str(msg.content)) < 1000, "Historical tool messages should be compressed"

    def test_error_message_preservation(self, context_manager):
        """测试错误信息即使具有历史意义，也会被保存。"""
        error_content = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    raise ValueError("Test error")
ValueError: Test error"""

        messages = [
            HumanMessage(content="Run code", id="human_1"),
            AIMessage(content="Executing", id="ai_1"),
            # 历史错误消息(应当保存)
            ToolMessage(content=error_content, tool_call_id="code_1", id="error_tool_historical"),
            # 最近的消息
            HumanMessage(content="What happened?", id="human_recent"),
            AIMessage(content="There was an error", id="ai_recent"),
        ]

        original_error_content = messages[2].content
        context_manager.manage_context_messages(messages)

        # 错误消息应当保存
        error_msg = next(msg for msg in messages if msg.id == "error_tool_historical")
        assert error_msg.content == original_error_content, "Error messages should be preserved"

    def test_sql_content_preservation(self, context_manager):
        """测试配置时 SQL 内容被保存。"""
        sql_content = """SQL Query:
```sql
SELECT * FROM users WHERE active = 1;
```

Query Results (CSV format):
```csv
id,name,email
1,John,john@example.com
2,Jane,jane@example.com
```"""

        messages = [
            HumanMessage(content="Get user data", id="human_1"),
            AIMessage(content="Querying users", id="ai_1"),
            # 历史的 SQL 结果(如果 preserve_recent_sql = True).
            ToolMessage(content=sql_content, tool_call_id="sql_1", id="sql_tool_historical"),
            # 最近的消息
            HumanMessage(content="Analyze results", id="human_recent"),
            AIMessage(content="Analyzing", id="ai_recent"),
        ]

        # 启用使用 SQL 保存的测试
        context_manager.config.preserve_recent_sql = True
        original_sql_content = messages[2].content
        context_manager.manage_context_messages(messages)

        # SQL 当 preserve_recent_sql = True 时应当保存。
        sql_msg = next(msg for msg in messages if msg.id == "sql_tool_historical")
        assert sql_msg.content == original_sql_content, "SQL content should be preserved when configured"

    @patch("insightflow.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summarization(self, mock_llm_call, context_manager):
        """测试对话的总结，并修改消息。"""
        # Mock LLM 对总和的反应
        mock_llm_call.return_value = AIMessage(content="Summary of the conversation")

        # 创建一个会触发归纳的长对话
        messages = []

        # 添加许多历史消息
        for i in range(20):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}" * 10, id=f"human_{i}"),
                    AIMessage(content=f"Response {i}" * 10, id=f"ai_{i}"),
                ]
            )

        # 添加最近的消息
        messages.extend(
            [
                HumanMessage(content="Recent question", id="human_recent"),
                AIMessage(content="Recent response", id="ai_recent"),
                ToolMessage(content="Recent result", tool_call_id="recent_1", id="tool_recent"),
            ]
        )

        original_count = len(messages)
        context_manager.manage_context_messages(messages)

        # 由于汇总，消息应减少
        assert len(messages) < original_count

        # 应该有一个简讯
        summary_msgs = [msg for msg in messages if isinstance(msg, AIMessage) and "Summary" in str(msg.content)]
        assert len(summary_msgs) > 0, "Should create a summary message"

    def test_content_type_detection(self, context_manager):
        """测试内容类型检测方法。"""
        # 测试错误内容检测
        error_contents = [
            "Error: Something went wrong",
            "Traceback (most recent call last):\n  File test.py",
            "ValueError: Invalid input",
            "Connection failed with status 500",
        ]

        for content in error_contents:
            assert context_manager._is_error_content(content), f"Should detect error in: {content[:50]}"

        # 测试 SQL 内容检测
        sql_contents = [
            "```sql\nSELECT * FROM users;\n```",
            "Query results: 100 rows returned",
            "SQL Query:\nSELECT id FROM table",
        ]

        for content in sql_contents:
            assert context_manager._is_sql_content(content), f"Should detect SQL in: {content[:50]}"

        # 测试数据查询结果检测
        data_contents = [
            "```csv\nid,name\n1,test\n```",
            "Query Results (CSV format):",
            "Found 500 records in the database",
        ]

        for content in data_contents:
            assert context_manager._is_data_query_result(content), f"Should detect data result in: {content[:50]}"

    def test_should_compress_logic(self, context_manager):
        """测试决定是否压缩历史工具消息的逻辑。"""
        # 短内容不应压缩
        short_msg = ToolMessage(content="Short", tool_call_id="test", id="short")
        assert not context_manager._should_compress_historical_tool_message(short_msg, "Short")

        # 长期的非敏感内容应当压缩
        long_content = "A" * 1000
        long_msg = ToolMessage(content=long_content, tool_call_id="test", id="long")
        assert context_manager._should_compress_historical_tool_message(long_msg, long_content)

        # 长错误内容不应压缩(如果 preserve_tool_errors = True).
        error_content = "Error: " + "A" * 1000
        error_msg = ToolMessage(content=error_content, tool_call_id="test", id="error")
        context_manager.config.preserve_tool_errors = True
        assert not context_manager._should_compress_historical_tool_message(error_msg, error_content)

        # 但如果 preserve_tool_errors = False ,则应当压缩。
        context_manager.config.preserve_tool_errors = False
        assert context_manager._should_compress_historical_tool_message(error_msg, error_content)

    def test_recent_messages_always_preserved(self, context_manager):
        """测试最近的消息无论内容如何都会被保存。"""
        # 创建最近大但仍应保存的邮件
        messages = []

        # 历史消息
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Historical {i}", id=f"hist_human_{i}"),
                    ToolMessage(content="A" * 500, tool_call_id=f"hist_{i}", id=f"hist_tool_{i}"),
                ]
            )

        # 最近的消息(包括大型工具输出)
        messages.extend(
            [
                HumanMessage(content="Recent question", id="recent_human"),
                AIMessage(content="Recent response", id="recent_ai"),
                ToolMessage(content="B" * 1000, tool_call_id="recent", id="recent_tool"),  # 大型但近期
            ]
        )

        len(messages)
        context_manager.manage_context_messages(messages)

        # 近期的信息应当保存(即使由于汇总而压缩内容)。
        recent_ids = ["recent_human", "recent_ai", "recent_tool"]
        remaining_recent = [msg for msg in messages if hasattr(msg, "id") and msg.id in recent_ids]

        # 所有最近的消息 ID仍应存在(即使进行了汇总)
        assert len(remaining_recent) >= 2, "Most recent messages should be preserved"

    def test_message_order_preservation(self, context_manager):
        """测试在上下文管理中保存消息命令。"""
        # 禁用对话摘要来测试只使用工具压缩
        context_manager.config.enable_conversation_summary = False
        context_manager.config.enable_summarization = False

        # 创建有特定顺序的消息
        messages = [
            HumanMessage(content="Question 1", id="human_1"),
            AIMessage(content="Response 1", id="ai_1"),
            ToolMessage(content="A" * 1000, tool_call_id="tool_1", id="tool_1"),  # 将被压缩
            HumanMessage(content="Question 2", id="human_2"),
            AIMessage(content="Response 2", id="ai_2"),
            ToolMessage(content="B" * 1000, tool_call_id="tool_2", id="tool_2"),  # 将被压缩
            HumanMessage(content="Recent question", id="human_recent"),  # 最近，不应压缩
            AIMessage(content="Recent response", id="ai_recent"),  # 近期
            ToolMessage(
                content="C" * 1000, tool_call_id="tool_recent", id="tool_recent"
            ),  # 最近，不应压缩
        ]

        original_order = [msg.id for msg in messages if hasattr(msg, "id")]
        context_manager.manage_context_messages(messages)

        # 在新顺序中提取 ID
        result_order = [msg.id for msg in messages if hasattr(msg, "id")]

        # 顺序应予保留
        assert result_order == original_order, "Message order should be preserved"

        # 校验历史工具消息是否被压缩
        historical_tools = [msg for msg in messages if isinstance(msg, ToolMessage) and msg.id in ["tool_1", "tool_2"]]
        for msg in historical_tools:
            assert len(str(msg.content)) < 1000, "Historical tool messages should be compressed"
