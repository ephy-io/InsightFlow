"""对代理图与上下文管理进行整合测试。"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from insightflow.agent_graph import _build_graph_core, agent_llm_call, build_agent_graph_async, build_agent_graph_sync
from insightflow.context_config import ContextConfig
from insightflow.context_manager import ContextManager
from insightflow.graph_state import AgentState


class TestAgentGraphIntegration:
    """对代理图与上下文管理进行整合测试。"""

    @pytest.fixture
    def mock_catalog(self):
        """用于测试的Mock目录存储。"""
        catalog = Mock()
        catalog.get_schema = Mock(return_value={"tables": []})
        return catalog

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM 用于测试。"""
        llm = Mock()
        llm.bind_tools = Mock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_tools(self):
        """模拟工具测试。"""

        def mock_tool_func(query: str) -> str:
            return "Mock tool result"

        tool = StructuredTool.from_function(func=mock_tool_func, name="mock_tool", description="Mock tool for testing")
        return [tool]

    @pytest.fixture
    def test_config(self):
        """用于上下文管理的测试配置。"""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=800,
            keep_recent_messages=3,
            max_tool_output_length=100,
        )

    def test_agent_llm_node_with_context_manager(self, mock_llm, mock_tools, test_config):
        """测试代理 llm_node 具有上下文管理器集成功能。"""
        context_manager = ContextManager(llm=mock_llm, config=test_config)

        # Mock LLM 响应
        mock_response = AIMessage(content="Test response", tool_calls=[])
        with patch("insightflow.agent_graph.call_llm_chat_model_with_retry", return_value=mock_response):
            llm_node_func = agent_llm_call(mock_llm, mock_tools, context_manager)

            # 用长消息创建测试状态以触发上下文管理
            long_messages = [
                HumanMessage(content="A" * 500),  # 长消息
                AIMessage(content="B" * 500),  # 长消息
                ToolMessage(content="C" * 200, tool_call_id="123"),  # 较长的工具输出
                HumanMessage(content="Recent question"),
            ]

            state = AgentState(messages=long_messages)
            result = llm_node_func(state)

            # 应该已经处理好了状态
            assert "messages" in result
            assert isinstance(result["messages"][0], AIMessage)

    def test_agent_llm_node_without_context_manager(self, mock_llm, mock_tools):
        """测试代理 llm_node 没有上下文管理器。"""
        mock_response = AIMessage(content="Test response", tool_calls=[])
        with patch("insightflow.agent_graph.call_llm_chat_model_with_retry", return_value=mock_response):
            llm_node_func = agent_llm_call(mock_llm, mock_tools, context_manager=None)

            state = AgentState(messages=[HumanMessage(content="Test")])
            result = llm_node_func(state)

            assert "messages" in result
            assert isinstance(result["messages"][0], AIMessage)

    def test_build_graph_core_with_context_management(self, mock_catalog, mock_llm):
        """启用了上下文管理的测试核心图构建。"""

        def create_mock_tool(name):
            def mock_func(input_str: str) -> str:
                return f"Mock {name} result"

            return StructuredTool.from_function(func=mock_func, name=name, description=f"Mock {name} tool")

        # 直接输入全部工具
        with (
            patch("insightflow.agent_graph.search_knowledge", create_mock_tool("search_knowledge")),
            patch("insightflow.agent_graph.search_schema", create_mock_tool("search_schema")),
            patch("insightflow.agent_graph.show_schema", create_mock_tool("show_schema")),
            patch("insightflow.agent_graph.run_python_code", create_mock_tool("run_python_code")),
            patch("insightflow.agent_graph.save_report", create_mock_tool("save_report")),
            patch("insightflow.agent_graph.get_sql_tools") as mock_get_sql_tools,
            patch("insightflow.agent_graph.build_sql_graph") as mock_sql_graph,
            patch("insightflow.agent_graph.get_memory_tools") as mock_memory_tools,
            patch("insightflow.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("insightflow.agent_graph.get_llm", return_value=mock_llm),
            patch("insightflow.analysis.agent.get_analysis_llm", return_value=mock_llm),
            patch("insightflow.analysis.agent.create_deep_agent", return_value=Mock()),
        ):

            # 设置基于函数的模拟
            mock_get_sql_tools.return_value = create_mock_tool("call_sql_graph_tool")
            mock_sql_graph.return_value = Mock()
            mock_memory_tools.return_value = (
                create_mock_tool("manage_memory_tool"),
                create_mock_tool("search_memory_tool"),
            )
            mock_mcp_tools.return_value = []

            graph = _build_graph_core(
                catalog=mock_catalog,
                sync_mode=True,
                checkpointer=None,
                memory_store=None,
                memory_tools=None,
                mcp_tools=[],
                enable_context_management=True,
            )

            # 创建已编译的图表
            assert graph is not None
            # 验证 SQL 图表已初始化
            mock_sql_graph.assert_called_once()

    def test_build_graph_core_without_context_management(self, mock_catalog, mock_llm):
        """测试核心图构建，上下文管理已禁用。"""

        def create_mock_tool(name):
            def mock_func(input_str: str) -> str:
                return f"Mock {name} result"

            return StructuredTool.from_function(func=mock_func, name=name, description=f"Mock {name} tool")

        # 直接锁定全部工具导入 - 与上下文管理相同的模式
        with (
            patch("insightflow.agent_graph.search_knowledge", create_mock_tool("search_knowledge")),
            patch("insightflow.agent_graph.search_schema", create_mock_tool("search_schema")),
            patch("insightflow.agent_graph.show_schema", create_mock_tool("show_schema")),
            patch("insightflow.agent_graph.run_python_code", create_mock_tool("run_python_code")),
            patch("insightflow.agent_graph.save_report", create_mock_tool("save_report")),
            patch("insightflow.agent_graph.get_sql_tools") as mock_get_sql_tools,
            patch("insightflow.agent_graph.build_sql_graph") as mock_sql_graph,
            patch("insightflow.agent_graph.get_memory_tools") as mock_memory_tools,
            patch("insightflow.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("insightflow.agent_graph.get_llm", return_value=mock_llm),
            patch("insightflow.analysis.agent.get_analysis_llm", return_value=mock_llm),
            patch("insightflow.analysis.agent.create_deep_agent", return_value=Mock()),
        ):

            # 设置基于函数的模拟
            mock_get_sql_tools.return_value = create_mock_tool("call_sql_graph_tool")
            mock_sql_graph.return_value = Mock()
            mock_memory_tools.return_value = (
                create_mock_tool("manage_memory_tool"),
                create_mock_tool("search_memory_tool"),
            )
            mock_mcp_tools.return_value = []

            graph = _build_graph_core(
                catalog=mock_catalog,
                sync_mode=True,
                checkpointer=None,
                memory_store=None,
                memory_tools=None,
                mcp_tools=[],
                enable_context_management=False,
            )

            # 仍应创建已编译的图表
            assert graph is not None

    def test_build_agent_graph_sync_with_context_management(self, mock_catalog):
        """测试同步图构建与上下文管理。"""
        with (
            patch("insightflow.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("insightflow.agent_graph._build_graph_core") as mock_build_core,
        ):

            mock_build_core.return_value = Mock()
            mock_mcp_tools.return_value = []

            graph = build_agent_graph_sync(catalog=mock_catalog, enable_context_management=True)

            # 校验_build_graph_core 使用正确的参数调用
            mock_build_core.assert_called_once()
            call_args = mock_build_core.call_args
            assert call_args[1]["enable_context_management"] is True

            # 应返回图表
            assert graph is not None

    @pytest.mark.asyncio
    async def test_build_agent_graph_async_with_context_management(self, mock_catalog):
        """与上下文管理一起测试ASync图建构。"""
        with (
            patch("insightflow.agent_graph.get_mcp_tools_async") as mock_mcp_tools,
            patch("insightflow.agent_graph._build_graph_core") as mock_build_core,
        ):

            mock_build_core.return_value = Mock()
            # 混音函数
            mock_mcp_tools.return_value = []

            graph = await build_agent_graph_async(catalog=mock_catalog, enable_context_management=True)

            # 校验_build_graph_core 使用正确的参数调用
            mock_build_core.assert_called_once()
            call_args = mock_build_core.call_args
            assert call_args[1]["enable_context_management"] is True

            # 应返回图表
            assert graph is not None

    @patch("insightflow.agent_graph.call_llm_chat_model_with_retry")
    def test_full_context_management_flow(self, mock_llm_call, mock_catalog):
        """在代理图中测试完整上下文管理流程。"""
        # Mock LLM 响应
        mock_llm_call.side_effect = [
            AIMessage(content="Response 1"),
            AIMessage(content="Summary of conversation"),  # 用于总结
            AIMessage(content="Final response"),
        ]

        context_manager = ContextManager(
            llm=Mock(),
            config=ContextConfig(
                enabled=True,
                summary_trigger_tokens=80,
                keep_recent_messages=2,
            ),
        )

        # 创建许多消息以触发上下文管理
        messages = []
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}" * 10),  # 使消息更长
                    AIMessage(content=f"Response {i}" * 10),
                    ToolMessage(content=f"Tool result {i}" * 20, tool_call_id=f"tool_{i}"),
                ]
            )

        # 测试上下文管理
        original_count = len(messages)
        context_manager.manage_context_messages(messages)
        managed_messages = messages

        # 消息应少于输入
        assert len(managed_messages) < original_count

        # 应保存最近的消息
        assert any("Question 9" in str(msg.content) for msg in managed_messages if hasattr(msg, "content"))


class TestContextManagementEdgeCases:
    """在代理图中测试上下文管理的边缘大小写。"""

    def test_empty_message_handling(self):
        """测试空消息的处理。"""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        messages = []
        context_manager.manage_context_messages(messages)
        result = messages
        assert result == []

    def test_state_message_type_validation(self):
        """测试在上下文管理期间只保留有效的状态消息类型。"""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        # 国家只应包含有效消息类型(无 SystemMessage)
        messages = [
            HumanMessage(content="A" * 100),  # 长消息
            AIMessage(content="B" * 100),  # 长消息
            HumanMessage(content="Recent question"),
        ]

        with patch(
            "insightflow.context_manager.call_llm_chat_model_with_retry", return_value=AIMessage(content="Summary")
        ):
            context_manager.manage_context_messages(messages)
            result = messages

        # 只应当包含有效的状态信息类型
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(type(msg) in valid_types for msg in result), "Should only contain valid state message types"

    def test_context_management_with_tool_calls(self):
        """当 AI 消息有工具调用时，测试上下文管理。"""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        ai_message_with_tools = AIMessage(
            content="I'll help you with that.",
            tool_calls=[{"name": "search_tool", "args": {"query": "test"}, "id": "call_123"}],
        )

        messages = [ai_message_with_tools, HumanMessage(content="Follow up")]

        context_manager.manage_context_messages(messages)
        result = messages

        # 使用工具调用 AI 消息应当保存
        ai_msgs = [msg for msg in result if isinstance(msg, AIMessage)]
        assert len(ai_msgs) > 0
        assert any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in ai_msgs)

    @patch("insightflow.context_manager.call_llm_chat_model_with_retry")
    def test_summarization_failure_fallback(self, mock_llm_call):
        """求和失败时测试后退行为。"""
        # Mock LLM 失败
        mock_llm_call.side_effect = Exception("LLM unavailable")

        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        # 创建会触发总和的信息( 状态中没有 SystemMessage)
        messages = [
            HumanMessage(content="A" * 100),  # 要触发的长消息
            AIMessage(content="B" * 100),
            HumanMessage(content="C" * 100),
            AIMessage(content="D" * 100),
            HumanMessage(content="Recent"),
        ]

        context_manager.manage_context_messages(messages)
        result = messages

        # 应回退到滑动窗口
        assert len(result) <= len(messages)
        # 保存最近的消息，只包含有效的状态消息类型
        assert any("Recent" in str(msg.content) for msg in result if hasattr(msg, "content"))
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(type(msg) in valid_types for msg in result), "Should only contain valid state message types"
