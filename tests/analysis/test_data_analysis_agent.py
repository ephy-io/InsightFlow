from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from insightflow.analysis import agent as analysis_agent
from insightflow.analysis.agent import (
    DataAnalysisInput,
    _build_sub_agent_config,
    _extract_final_content,
    _load_data_analysis_prompt,
    build_data_analysis_agent,
    get_data_analysis_tool,
)


@pytest.fixture
def mock_sql_graph():
    return MagicMock(spec=CompiledStateGraph)


def test_load_data_analysis_prompt_injects_min_input_length(monkeypatch):
    # [ min_input_length ]占位符被解析的模型最小值所取代。
    monkeypatch.setattr(analysis_agent, "_resolve_min_input_length", lambda: 48)
    prompt = _load_data_analysis_prompt()
    assert "[min_input_length]" not in prompt
    assert "48 个输入点" in prompt


@patch("insightflow.analysis.agent.create_deep_agent")
@patch("insightflow.analysis.agent.get_analysis_llm")
@patch("insightflow.agent_graph.get_sql_tools")
@patch("insightflow.analysis.agent.check_forecast_service_health")
def test_build_data_analysis_agent(
    mock_check_health, mock_get_sql_tools, mock_get_llm, mock_create_deep_agent, mock_sql_graph
):
    # 设置模拟器
    mock_check_health.return_value = True
    mock_text2sql = MagicMock(spec=StructuredTool)
    mock_text2sql.name = "text2sql"
    mock_get_sql_tools.return_value = mock_text2sql

    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    mock_agent = MagicMock(spec=CompiledStateGraph)
    mock_create_deep_agent.return_value = mock_agent

    # 调用
    agent = build_data_analysis_agent(sql_graph=mock_sql_graph)

    # 显示( A)
    assert agent == mock_agent
    mock_get_sql_tools.assert_called_once_with(sql_graph=mock_sql_graph, sync_mode=False)
    mock_get_llm.assert_called_once()

    # 请检查date=中的日期值 (帮助) create_deep_agent 调用正确参数
    call_args = mock_create_deep_agent.call_args[1]
    assert call_args["model"] == mock_llm
    assert "system_prompt" in call_args

    # 检查工具列表
    tools = call_args["tools"]
    tool_names = [t.name if hasattr(t, "name") else t.__name__ for t in tools]
    assert "text2sql" in tool_names
    assert "run_python_code" in tool_names
    assert "timeseries_forecast" in tool_names
    assert "anomaly_detection" in tool_names
    assert "adtributor_drilldown" in tool_names


@patch("insightflow.analysis.agent.build_data_analysis_agent")
def test_get_data_analysis_tool_sync(mock_build_agent, mock_sql_graph):
    # 设置模拟代理
    mock_agent = MagicMock()
    # 代理返回包含 AIMessage 的 "messages" 密钥的命令
    mock_message = MagicMock()
    mock_message.content = "Analysis result"
    mock_agent.invoke.return_value = {"messages": [mock_message]}
    mock_build_agent.return_value = mock_agent

    # 调用
    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)

    # 显示( A)
    assert isinstance(tool, StructuredTool)
    assert tool.name == "data_analysis"
    assert tool.args_schema == DataAnalysisInput

    # 测试引用
    result = tool.invoke(
        {"reasoning": "test", "task": "analyze this"},
        config={"configurable": {"thread_id": "parent-123"}},
    )
    assert result == "Analysis result"
    mock_agent.invoke.assert_called_once()

    # 副剂必须接收一个孤立的，衍生出来的 thread_id (不是母体).
    _, call_kwargs = mock_agent.invoke.call_args
    sub_config = call_kwargs["config"]
    assert sub_config["configurable"]["thread_id"] == "parent-123:data_analysis"


@pytest.mark.asyncio
@patch("insightflow.analysis.agent.build_data_analysis_agent")
async def test_get_data_analysis_tool_async(mock_build_agent, mock_sql_graph):
    # 设置模拟代理
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Async analysis result"

    # 创建返回命令的曲轴
    async def mock_ainvoke(*args, **kwargs):
        return {"messages": [mock_message]}

    mock_agent.ainvoke = mock_ainvoke
    mock_build_agent.return_value = mock_agent

    # 调用
    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=False)

    # 显示( A)
    assert isinstance(tool, StructuredTool)
    assert tool.name == "data_analysis"

    # 测试引用
    result = await tool.ainvoke({"reasoning": "test", "task": "analyze this"})
    assert result == "Async analysis result"


@patch("insightflow.analysis.agent.build_data_analysis_agent")
def test_sync_tool_propagates_graph_interrupt(mock_build_agent, mock_sql_graph):
    """副剂的 GraphInterrupt 必须泡上( HITL),不能吞噬。"""
    from langgraph.errors import GraphInterrupt

    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = GraphInterrupt(())
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)
    with pytest.raises(GraphInterrupt):
        tool.invoke({"reasoning": "test", "task": "analyze this"})


@pytest.mark.asyncio
@patch("insightflow.analysis.agent.build_data_analysis_agent")
async def test_async_tool_propagates_graph_interrupt(mock_build_agent, mock_sql_graph):
    """从Async子剂得到的 GraphInterrupt 必须泡上( HITL),不能被吞噬。"""
    from langgraph.errors import GraphInterrupt

    mock_agent = MagicMock()

    async def mock_ainvoke(*args, **kwargs):
        raise GraphInterrupt(())

    mock_agent.ainvoke = mock_ainvoke
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=False)
    with pytest.raises(GraphInterrupt):
        await tool.ainvoke({"reasoning": "test", "task": "analyze this"})


@patch("insightflow.analysis.agent.build_data_analysis_agent")
def test_sync_tool_returns_error_string_on_generic_exception(mock_build_agent, mock_sql_graph):
    """非中断异常被捕获并返回为错误字符串。"""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("boom")
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)
    result = tool.invoke({"reasoning": "test", "task": "analyze this"})
    assert "Error occurred during data analysis" in result
    assert "boom" in result


def test_build_sub_agent_config_derives_isolated_thread_id():
    """使用 thread_id 生成的孩子; checkpoint_id 被丢弃并
    检查站命名空间被固定为单级值。
    """
    parent = {
        "configurable": {
            "thread_id": "abc",
            "checkpoint_ns": "use_tool:1",
            "checkpoint_id": "ckpt-1",
        },
        "tags": ["x"],
    }
    sub = _build_sub_agent_config(parent)
    assert sub["configurable"]["thread_id"] == "abc:data_analysis"
    # checkpoint_ns 被钉住(没有弹出),所以子剂流作为巢
    # 子图; 父任务范围命名空间不得漏出。
    assert sub["configurable"]["checkpoint_ns"] == "data_analysis"
    # checkpoint_id 重置是真正的隔离保障。
    assert "checkpoint_id" not in sub["configurable"]
    # 非配置密钥被传播。
    assert sub["tags"] == ["x"]


def test_build_sub_agent_config_handles_none():
    """任何父配置都不应该产生有效的(空的)可配置。"""
    sub = _build_sub_agent_config(None)
    assert sub["configurable"] == {}


def test_extract_final_content_string_and_blocks():
    str_msg = MagicMock()
    str_msg.content = "hello"
    assert _extract_final_content({"messages": [str_msg]}) == "hello"

    block_msg = MagicMock()
    block_msg.content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _extract_final_content({"messages": [block_msg]}) == "a\nb"

    assert _extract_final_content({"messages": []}) == "{'messages': []}"
