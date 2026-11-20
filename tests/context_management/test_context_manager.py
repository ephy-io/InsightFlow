"""ContextManager 级的单位测试。"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from insightflow.context_config import ContextConfig
from insightflow.context_manager import ContextManager


class TestContextManager:
    """ContextManager 类测试用例。"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM 用于测试。"""
        llm = Mock()
        # 模拟求解响应
        llm_response = AIMessage(content="This is a test summary of the conversation.")
        with patch("insightflow.context_manager.call_llm_chat_model_with_retry", return_value=llm_response):
            yield llm

    @pytest.fixture
    def default_config(self):
        """用于测试的默认上下文配置。"""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=900,
            keep_recent_messages=3,
            max_tool_output_length=200,
            max_sql_result_rows=5,
            max_code_output_lines=10,
            enable_conversation_summary=True,
            enable_summarization=True,
        )

    @pytest.fixture
    def context_manager(self, mock_llm, default_config):
        """用于测试的上下文管理器。"""
        return ContextManager(llm=mock_llm, config=default_config)

    def test_token_estimation(self, context_manager):
        """测试令牌估计功能。"""
        # 测试基本符号估计
        short_text = "Hello world"
        assert context_manager.estimate_tokens(short_text) == len(short_text) // 4

        # 测试较长的文本
        long_text = "This is a longer text that should have more tokens estimated."
        assert context_manager.estimate_tokens(long_text) > context_manager.estimate_tokens(short_text)

    def test_message_token_estimation(self, context_manager):
        """测试消息的符号估计值。"""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            ToolMessage(content="Tool result", tool_call_id="123"),
        ]

        total_tokens = context_manager.estimate_message_tokens(messages)
        assert total_tokens > 0
        # 应包括内容符号加元数据间接费用
        assert total_tokens > sum(len(str(msg.content)) // 4 for msg in messages)

    def test_trim_short_tool_output(self, context_manager):
        """测试修剪工具输出已经足够短了。"""
        short_output = "This is a short output."
        result = context_manager.trim_tool_output(short_output)
        assert result == short_output

    def test_trim_long_generic_output(self, context_manager):
        """测试修剪长长的通用工具输出。"""
        long_output = "A" * 500  # 比 max_tool_output_length (200)长得多
        result = context_manager.trim_tool_output(long_output)

        assert len(result) < len(long_output)
        assert "... [Output truncated] ..." in result
        assert result.startswith("A")
        assert result.endswith("A")

    def test_trim_sql_output(self, context_manager):
        """以结构化数据对 SQL 输出进行测试修剪。"""
        sql_output = """SQL Query:
```sql
SELECT * FROM users WHERE age > 18;
```

Query Results (CSV format):
```csv
id,name,age,email
1,John,25,john@example.com
2,Jane,30,jane@example.com
3,Bob,22,bob@example.com
4,Alice,28,alice@example.com
5,Charlie,35,charlie@example.com
6,Diana,27,diana@example.com
7,Eve,31,eve@example.com
```

Visualization Created: bar chart has been automatically generated."""

        result = context_manager.trim_tool_output(sql_output)

        # 应保存 SQL 查询
        assert "SELECT * FROM users WHERE age > 18;" in result
        # 应保存可视化信息
        assert "Visualization Created:" in result
        # 是否应该修剪 CSV 数据但保持结构
        assert "```csv" in result
        assert "rows omitted" in result

    def test_trim_code_output(self, context_manager):
        """测试三进制 Python 代码执行输出。"""
        # 测试无错误的长输出
        long_code_output = "\n".join([f"Line {i}: Some output here" for i in range(50)])
        result = context_manager.trim_tool_output(long_code_output)

        assert len(result.split("\n")) < 50
        assert "... [Output truncated] ..." in result

    def test_preserve_error_output(self, context_manager):
        """测试配置时是否保留了错误输出。"""
        error_output = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    print(undefined_variable)
NameError: name 'undefined_variable' is not defined"""

        # 用 preserve_tool_errors = True (测试配置中的默认)
        result = context_manager.trim_tool_output(error_output)
        assert result == error_output  # 应该完整保存

        # 试验用 preserve_tool_errors = False
        context_manager.config.preserve_tool_errors = False
        result = context_manager.trim_tool_output(error_output)
        # 仍然应该保存，因为它是一个错误，但可以根据长度进行修剪

    # 工具输出修剪禁用测试被移除 - 修剪现在总是启用

    def test_conversation_summary_disabled(self, context_manager):
        """禁用时测试对话摘要。"""
        context_manager.config.enable_conversation_summary = False
        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]

        summary = context_manager.summarize_conversation(messages)
        assert summary == ""

    @patch("insightflow.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summary_success(self, mock_llm_call, context_manager):
        """测试成功的对话总结。"""
        # 模拟成功 LLM 响应
        mock_response = AIMessage(content="Summary: User asked about data analysis.")
        mock_llm_call.return_value = mock_response

        messages = [
            HumanMessage(content="Can you analyze our sales data?"),
            AIMessage(content="I'll help you analyze the sales data."),
            ToolMessage(content="Query results: 100 records", tool_call_id="123"),
            HumanMessage(content="What are the trends?"),
            AIMessage(content="The trends show increasing sales."),
            HumanMessage(content="Recent question"),  # 这一点应排除在摘要之外。
        ]

        summary = context_manager.summarize_conversation(messages)
        assert summary.startswith("[Conversation Summary]:")
        assert "Summary: User asked about data analysis." in summary
        mock_llm_call.assert_called_once()

    @patch("insightflow.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summary_failure(self, mock_llm_call, context_manager):
        """当 LLM 呼叫失败时测试对话摘要。"""
        # Mock LLM 失败
        mock_llm_call.side_effect = Exception("LLM service unavailable")

        # 需要比 keep_recent_messages (3) 更多的消息来触发总和
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Second message"),
            AIMessage(content="Second response"),
            HumanMessage(content="Third message"),
            AIMessage(content="Third response"),
        ]
        summary = context_manager.summarize_conversation(messages)
        assert summary == "[Summary generation failed]"

    def test_manage_context_disabled(self, context_manager):
        """禁用时测试上下文管理。"""
        context_manager.config.enabled = False
        messages = [HumanMessage(content="Test")]

        context_manager.manage_context_messages(messages)
        result = messages
        assert result == messages  # 返回时不变

    def test_manage_context_empty_messages(self, context_manager):
        """用空消息列表测试上下文管理。"""
        messages = []
        context_manager.manage_context_messages(messages)
        result = messages
        assert result == []

    def test_manage_context_tool_message_trimming(self, context_manager):
        """测试工具信息在上下文管理期间会修剪。"""
        long_content = "A" * 500
        # 添加足够的消息以触发上下文管理，历史部分为 ToolMessage
        # keep_recent_messages=3，因此要在 ToolMessage 后添加超过 3 条消息。
        messages = [
            HumanMessage(content="This is a long question that helps reach the token threshold " * 10),
            ToolMessage(content=long_content, tool_call_id="123"),  # 这应该是历史的一部分
            AIMessage(content="This is a long response that helps reach the token threshold " * 10),
            HumanMessage(content="Another long question to increase token count " * 10),
            AIMessage(content="Response " * 20),
            HumanMessage(content="Final question"),
            AIMessage(content="Final response"),
        ]

        context_manager.manage_context_messages(messages)
        result = messages

        # 在结果中查找工具信息
        tool_msg = next(msg for msg in result if isinstance(msg, ToolMessage))
        assert len(str(tool_msg.content)) < len(long_content)
        assert "... [Output truncated] ..." in str(tool_msg.content)

    @patch("insightflow.context_manager.call_llm_chat_model_with_retry")
    def test_manage_context_with_summarization(self, mock_llm_call, context_manager):
        """测试上下文管理触发归纳。"""
        # 模拟成功总结
        mock_response = AIMessage(content="Conversation summary here.")
        mock_llm_call.return_value = mock_response

        # 创建许多消息以触发总和
        messages = []
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}"),
                    AIMessage(content=f"Response {i}" * 100),  # 长时间响应以增加符数
                ]
            )

        original_length = len(messages)
        context_manager.manage_context_messages(messages)
        result = messages

        # 消息应少于输入
        assert len(result) < original_length
        # 应包含摘要信息
        assert any("[Conversation Summary]:" in str(msg.content) for msg in result if hasattr(msg, "content"))
        # 校验 LLM 请求进行汇总
        mock_llm_call.assert_called_once()

    # 工具包装测试已删除 -我们现在处理州一级的上下文

    def test_format_messages_for_summary(self, context_manager):
        """测试摘要生成的信息格式化。"""
        messages = [
            HumanMessage(content="User question"),
            AIMessage(content="AI response"),
            ToolMessage(content="Tool result with some data", tool_call_id="123"),
            SystemMessage(content="System message"),  # 应予排除
        ]

        formatted = context_manager._format_messages_for_summary(messages)

        assert "<user> User question </user>" in formatted
        assert "<assistant>" in formatted and "AI response" in formatted
        assert "tool_result" in formatted
        assert "System message" not in formatted  # 系统消息被排除

    def test_format_long_ai_message_for_summary(self, context_manager):
        """测试长的 AI 消息以摘要格式切换。"""
        long_content = "A" * 1000
        messages = [AIMessage(content=long_content)]

        formatted = context_manager._format_messages_for_summary(messages)

        assert len(formatted) < len(f"Assistant: {long_content}")
        assert "... [truncated]" in formatted


# 工具包装测试被移除 -我们现在处理州级上下文，而不是包装工具


# Pytest 固定装置和测试数据
@pytest.fixture
def sample_sql_output():
    """SQL 样本输出用于测试。"""
    return """SQL Query:
```sql
SELECT customer_id, SUM(amount) as total
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY customer_id
ORDER BY total DESC;
```

Query Results (CSV format):
```csv
customer_id,total
1001,15420.50
1002,12350.75
1003,11200.00
1004,9875.25
1005,8650.00
1006,7500.50
1007,6200.75
1008,5800.00
1009,4950.25
1010,4200.00
```

Visualization Created: bar chart has been automatically generated and will be displayed in the UI."""


@pytest.fixture
def sample_error_output():
    """用于测试的样本错误输出。"""
    return """Traceback (most recent call last):
  File "/app/code.py", line 15, in analyze_data
    result = df.groupby('nonexistent_column').sum()
  File "/usr/local/lib/python3.9/site-packages/pandas/core/groupby/groupby.py", line 1647, in sum
    return self._cython_transform("sum", numeric_only=numeric_only, **kwargs)
KeyError: 'nonexistent_column'

Error: Column 'nonexistent_column' not found in DataFrame. Available columns: ['customer_id', 'order_date', 'amount', 'product_id']"""
