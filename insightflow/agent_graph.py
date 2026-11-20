"""主 Agent 图的构建与执行逻辑。"""

import datetime
import json
import logging
import re
import traceback
from collections.abc import Callable
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send, interrupt
from pydantic import BaseModel, Field, ValidationError

from insightflow import config
from insightflow.catalog import CatalogStore
from insightflow.constants import datetime_format
from insightflow.context_config import get_context_config
from insightflow.context_manager import ContextManager
from insightflow.graph_state import AgentState, InputState, OutputState
from insightflow.final_json import check_final_json, check_sql_claims, requests_json
from insightflow.attribution_answer import render
from insightflow.llm.llm import bind_tools_compatibly, call_llm_chat_model_with_retry, get_llm
from insightflow.prompts.system_prompt import get_agent_prompt_template
from insightflow.text2sql.sql_graph import build_sql_graph
from insightflow.tool.ask_human import AskHuman
from insightflow.tool.mcp_tools import create_mcp_tools_sync, get_mcp_tools_async
from insightflow.tool.memory import get_memory_tools
from insightflow.tool.run_python_code import run_python_code
from insightflow.tool.save_report import save_report
from insightflow.tool.search_knowledge import search_knowledge, search_schema, show_schema
from insightflow.utils import get_text_from_content, log, recover_incomplete_tool_calls

logger = logging.getLogger(__name__)

def _get_mcp_servers() -> list[Any]:
    """从配置中获取 MCP 服务列表；配置不可用时返回空列表，便于测试。"""
    try:
        return config.get().mcp_servers
    except ValueError:
        return []


def ask_human(state: AgentState) -> dict[str, Any]:
    """向用户询问补充信息或澄清问题的节点函数。
    参数：
        state (AgentState)：当前图状态，包含消息和上下文。
    返回值：
        dict：更新后的状态，包含用户反馈对应的工具消息和用户输入。
    """
    last_message = cast(AIMessage, state["messages"][-1])
    tool_call = last_message.tool_calls[0]
    tool_call_id = tool_call["id"]
    args = tool_call["args"]
    user_feedback = interrupt({"text": args["question"], "buttons": args.get("options", None)})
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "name": "AskHuman", "content": user_feedback}]
    return {
        "messages": tool_message,
        "history_messages": [AIMessage(args["question"]), HumanMessage(user_feedback)],
        "user_input": user_feedback,
    }


class CallSQLGraphInput(BaseModel):
    reasoning: str = Field(
        description="说明为什么需要调用 Text2SQL 工具",
    )
    context: str | dict[str, Any] | list[dict[str, Any]] | list[str] = Field(
        description="""传递给 Text2SQL 的上下文内容。
        建议采用结构化格式，既保留必要信息，又明确区分各个任务的范围：
        - Shared Context（共享上下文）：稳定的业务背景、业务实体、指标和关键筛选条件
        - Current Subtask（当前子任务，即用户最新的问题）：本次唯一需要立即执行的具体查询。
        - Carry-over From Previous Step（承接上一步）：上一步的 SQL 或查询结果，以及本次需要做出的具体改动（可选）
        - Deferred Tasks（待处理任务）：本次暂不执行的其他阶段任务（可选）""",
    )
    visualize: bool = Field(
        default=True,
        description=(
            "本次 Text2SQL 调用是否生成用于界面展示的可视化 DSL。"
            "如果只是为后续分析工具获取中间数据，请设为 false。"
        ),
    )


# Text2SQL 工具的功能说明
_TEXT2SQL_TOOL_DESCRIPTION = """根据用户的问题和上下文，生成并执行 SQL 查询，同时生成用于界面展示的可视化 DSL。

    返回值：
        str：格式化后的结果，包含 SQL、查询数据和可视化状态。

    重要说明：
    - 不要用本工具探索数据库 Schema、列出数据表，或查找哪些表或字段包含数据。
    调用 Text2SQL 前，先使用 search_schema 搜索相关表结构，再使用 show_schema 查看已确定的数据表详情。
    - 如果用户想修改图表类型或样式，请将该要求写入问题中。
    - 如果本次调用仅用于获取后续分析所需的数据，或用户明确不需要图表，请设置 visualize=false。
    - 问题和上下文使用中文表达，表名、字段名及 SQL 等技术标识保持原样。
    - 在多阶段工作流中，本次调用只传递当前子任务范围内的上下文，其他阶段放入 Deferred Tasks（待处理任务）。
    - 如果本次调用依赖前一步的 SQL 意图，请通过 Carry-over（承接上一步）明确说明需要调整的部分，不要重新发送整个多阶段任务。
"""


def _normalize_text2sql_context(context: str | dict[str, Any] | list[dict[str, Any]] | list[str]) -> str:
    """将传入的工具上下文转为 SQL Graph 所需的字符串。"""
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, ensure_ascii=False)  # 转换成json格式
    except (TypeError, ValueError):
        return str(context)


def _format_sql_response(sql_graph_response: dict[str, Any]) -> str:
    """将 SQL Graph 的返回结果整理为统一的字符串格式。
    参数：
        sql_graph_response：SQL Graph 返回的结果字典。
    返回值：
        str：整理后的结果字符串。
    """
    sql = sql_graph_response.get("sql", "")
    data = sql_graph_response.get("data", "")
    visualization_dsl = sql_graph_response.get("visualization_dsl", {})

    response_parts: list[str] = []
    if sql:
        response_parts.append(f"SQL Query:\n```sql\n{sql}\n```")
    if data:
        response_parts.append(f"\nQuery Results (CSV format):\n```csv\n{data}\n```")

    # 在结果中加入可视化状态
    if visualization_dsl and "error" not in visualization_dsl:
        chart_type = visualization_dsl.get("chart_type", "unknown")
        response_parts.append(
            f"\n可视化已创建：已自动生成 {chart_type} 类型的图表，并将在用户界面中显示。"
        )
    elif visualization_dsl and "error" in visualization_dsl:
        response_parts.append(f"\nVisualization Error: {visualization_dsl['error']}")

    return "\n\n".join(response_parts) if response_parts else "No results returned."


def get_sql_tools(sql_graph: CompiledStateGraph[Any, None, Any, Any], sync_mode: bool = False) -> StructuredTool:
    """将编译完成的 SQL Graph 封装为可供 Agent 调用的工具。

    参数：
        sql_graph (CompiledStateGraph)：已编译的 SQL 生成子图。
        sync_mode (bool)：是否创建同步工具；否则创建异步工具。

    返回值：
        function：封装后的 SQL 工具。
    """

    def call_sql_graph_sync(
        reasoning: str,
        context: str | dict[str, Any] | list[dict[str, Any]] | list[str],
        config: RunnableConfig,
        visualize: bool = True,
    ) -> str:
        """同步执行 Text2SQL 工具的函数。"""
        normalized_context = _normalize_text2sql_context(context)
        log(f"调用 SQL Graph（同步模式），调用原因：{reasoning}，是否生成可视化：{visualize}，上下文：{normalized_context}")
        try:
            sql_graph_response = sql_graph.invoke(
                {"messages": normalized_context, "visualize": visualize}, config=config )
            return _format_sql_response(sql_graph_response)
        # GraphInterrupt 表示工作流中断，需重新抛出交给上层处理；普通异常则记录错误并返回提示信息，避免异常继续向上抛出。
        except GraphInterrupt as e:
            log(f"SQL Graph 执行被中断：\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"执行 SQL Graph 时发生错误：\n{repr(e)}")
            traceback.print_exc()
        return "调用 Text2SQL 工具时发生错误。"

    async def call_sql_graph_async(
        reasoning: str,
        context: str | dict[str, Any] | list[dict[str, Any]] | list[str],
        config: RunnableConfig,
        visualize: bool = True,
    ) -> str:
        """异步执行 Text2SQL 工具的函数。"""
        normalized_context = _normalize_text2sql_context(context)
        log(
            f"调用 SQL Graph（异步模式），调用原因：{reasoning}，是否生成可视化：{visualize}，上下文：{normalized_context}"
        )
        try:
            sql_graph_response = await sql_graph.ainvoke(
                {"messages": normalized_context, "visualize": visualize}, config=config
            )
            return _format_sql_response(sql_graph_response)
        except GraphInterrupt as e:
            log(f"Sql graph interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"Run sql graph error:\n{repr(e)}")
            traceback.print_exc()
        return "Error occurred when calling Text2SQL tool."

    if sync_mode:
        return StructuredTool.from_function(
            func=call_sql_graph_sync,
            name="text2sql",
            description=_TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            return_direct=False,
        )
    else:
        return StructuredTool.from_function(
            coroutine=call_sql_graph_async,
            name="text2sql",
            description=_TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            return_direct=False,
        )


class TaskIntent(BaseModel):
    """识别本轮交付目标并确定取数归属。"""

    route: Literal["attribution", "general"]
    task: str = Field(description="归因时结合上下文写明指标、对照期、实际期、拆分维度和筛选条件；其他任务填空字符串。")


def agent_llm_call(
    llm: BaseChatModel,
    tools: list[Any],
    context_manager: ContextManager | None = None,
) -> Callable[..., Any]:
    """功能： 创建一个调用大模型的函数，让 Agent 根据模型的回复，判断下一步执行哪个节点（如调用工具或结束任务）。

    参数（Args）：
    - llm：负责分析和决策的大模型。
    - tools：Agent 可使用的工具列表。
    - context_manager：可选，用于管理长对话的上下文。
    返回值（Returns）：
    - function：返回一个函数，用于处理 Agent 状态并决定下一步执行哪个节点。
    """

    llm_with_tools = bind_tools_compatibly(llm, tools)
    has_analysis = any(getattr(tool, "name", None) == "data_analysis" for tool in tools)

    def _call_model(state: AgentState) -> dict[str, Any]:
        # 首先，检查是否存在未完成的工具调用，并尝试恢复执行
        recovery_ops = recover_incomplete_tool_calls(state)
        if recovery_ops:
            return {"messages": recovery_ops, "agent_next_node": "llm_node"}

        messages = state["messages"]
        final_messages: list[HumanMessage | AIMessage] = []
        if isinstance(messages[-1], HumanMessage):
            final_messages.append(messages[-1])

        # 如果启用了上下文管理，则在处理任务前先进行上下文管理。
        if context_manager:
            original_count = len(messages)
            # 调用上下文管理器，按需压缩和整理历史消息，避免上下文过长
            context_manager.manage_context_messages(messages)  # type: ignore[arg-type]
            if len(messages) != original_count:
                logger.info(f"上下文管理：消息数量从 {original_count} 条变为 {len(messages)} 条")
        # 加载 Agent 系统提示词，并将时间占位符替换为当前日期时间
        system_prompt = get_agent_prompt_template().replace(
            "[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format)
        )

        user_request = next((get_text_from_content(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        json_requested = requests_json(user_request)
        evidence = None
        sql_output = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage) and msg.name == "text2sql" and sql_output is None:
                sql_output = str(msg.content)
            if isinstance(msg, ToolMessage) and msg.name == "data_analysis":
                evidence = msg.artifact
                break
        response = None
        last = messages[-1]
        if has_analysis and (isinstance(last, HumanMessage) or isinstance(last, ToolMessage) and last.name == "AskHuman"):
            # 每轮先判断用户要交付的业务结论；已经取得的工具结果不会再次触发分类。
            instruction = (
                "你负责主 Agent 的任务分派。结合对话判断当前目标，不执行查询。判定依据是用户希望得到什么结论。"
                "需要解释经营指标的变化由哪些分组推动、拖累或抵消，选 attribution，即使交付形式只是排名或单个数字。"
                "只需列出数值、比较差额、排名、一般统计、预测、知识查询或复述已有分析，选 general。"
                "归因范围不足且需要用户澄清时也选 general。把相对日期按当前时间换算，忠实保留条件。当前时间："
                + datetime.datetime.now().strftime(datetime_format)
            )
            # 这里的结构化工具只生成路由决策，业务工具由原有 ToolNode 执行。
            planner = llm.bind_tools([TaskIntent], tool_choice="TaskIntent")
            decision = planner.invoke([SystemMessage(instruction), *messages], config={"tags": ["intent"]})
            calls = decision.tool_calls
            try:
                intent = TaskIntent.model_validate(calls[0]["args"] if len(calls) == 1 and calls[0]["name"] == "TaskIntent" else {})
            except ValidationError:
                response = AIMessage(content="本轮任务分类未能完成，尚未执行查询。")
            else:
                if intent.route == "attribution":
                    response = AIMessage(content="", tool_calls=[{
                        "name": "data_analysis", "id": "call_" + uuid4().hex,
                        "args": {"reasoning": "本轮需要解释各分组对指标变动的贡献。",
                                 "task": intent.task + "\n本轮用户要求：\n" + user_request,
                                 "attribution_only": True},
                    }])
        if response is None:
            response = call_llm_chat_model_with_retry(
                cast(BaseChatModel, llm_with_tools),
                ([SystemMessage(system_prompt)] + messages),
                streaming_tokens=not (json_requested or evidence or sql_output),
                bound_tools=tools,
                parallel_tool_call=True,
            )
        if isinstance(response, AIMessage):
            tool_calls = response.tool_calls
            print("Tool Call:", ", ".join(tool["name"] for tool in tool_calls))
            if tool_calls:
                # 按调用类型拆分工具调用，便于分别路由
                ask_human_calls = [call for call in tool_calls if call["name"] == "AskHuman"]
                normal_tool_calls = [call for call in tool_calls if call["name"] != "AskHuman"]

                # 创建 Send 任务，用于将调用分派给对应节点
                sends: list[Send] = []
                if ask_human_calls:
                    # 只保留 AskHuman 调用，构造发送给人工交互节点的消息
                    ask_human_msg = AIMessage(content=response.content, tool_calls=ask_human_calls)
                    sends.append(Send("ask_human", {"messages": [ask_human_msg]}))

                if normal_tool_calls:
                    # 只保留普通工具调用，构造发送给工具执行节点的消息
                    tool_msg = AIMessage(content=response.content, tool_calls=normal_tool_calls)
                    sends.append(Send("use_tool", {"messages": [tool_msg]}))

                return {"messages": [response], "history_messages": final_messages, "sends": sends}
            else:
                final_answer = get_text_from_content(response.content)
                if sql_output and not evidence:
                    final_answer = check_sql_claims(final_answer, sql_output)
                if evidence:
                    verified, payload = render(evidence, user_request)
                    # 留存报告等后续工具的返回仍需交付给用户。
                    links = re.findall(r"/api/download/report/[\w.-]+", final_answer)
                    if not json_requested or payload:
                        final_answer = verified + ("\n```json\n"+payload+"\n```" if payload else "")
                        if links:
                            final_answer += "\n" + "\n".join(dict.fromkeys(links))
                    else:
                        final_answer = verified + "\n无法从已核对的分组数据唯一确定所要求的 JSON 结构。"
                if json_requested:
                    final_answer = check_final_json(final_answer, user_request)
                if json_requested or evidence or sql_output:
                    response = response.model_copy(update={
                        "content": final_answer,
                        "additional_kwargs": {**response.additional_kwargs, "validated_json_delivery": True},
                    })
                final_messages.append(AIMessage(response.content))
                return {
                    "messages": [response],
                    "final_answer": final_answer,
                    "history_messages": final_messages,
                    "agent_next_node": END,
                }
        elif response is None:
            return {
                "messages": [AIMessage("Sorry, the LLM service is currently unavailable.")],
                "history_messages": final_messages,
                "agent_next_node": END,
            }
        else:
            return {"messages": [response], "history_messages": final_messages, "agent_next_node": END}

    return _call_model


def _build_graph_core(
    catalog: CatalogStore,
    sync_mode: bool,
    checkpointer: Checkpointer | None,
    memory_store: BaseStore | None,
    memory_tools: list[Any] | None,
    mcp_tools: list[Any],
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[AgentState, None, InputState, OutputState]:
    """同步和异步版本共用的核心图构建逻辑。

    参数（Args）：
    - catalog：存储数据库 Schema（表结构等）信息的目录存储。
    - sync_mode：是否使用同步模式执行工具和操作。
    - checkpointer：检查点保存器，用于持久化保存 Agent 的运行状态。
    - memory_store：长期记忆存储器，用于保存 Agent 的长期记忆。
    - memory_tools：记忆管理工具列表，包括记忆管理工具和记忆搜索工具。
    - mcp_tools：已经初始化好的 MCP 工具。
    - enable_context_management：是否启用上下文管理功能。
    返回值（Returns）：

    - CompiledStateGraph：编译完成的 Agent 状态图，可以直接执行
    """
    sql_graph = build_sql_graph(catalog, checkpointer, cast(BaseStore, memory_store), llm_provider=llm_provider)
    call_sql_graph_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    # 创建数据分析子 Agent 工具
    from insightflow.analysis.agent import get_data_analysis_tool

    data_analysis_tool = get_data_analysis_tool(
        sql_graph=sql_graph,
        sync_mode=sync_mode,
        llm_provider=llm_provider,
        checkpointer=cast(BaseCheckpointSaver[Any] | None, checkpointer if checkpointer is not False else None),
        memory_store=memory_store,
    )

    # 优先使用传入的记忆工具；没有则自行创建
    memory_tools_for_graph: list[Any] | None = list(memory_tools) if memory_tools else None
    if not memory_tools_for_graph:
        memory_tools_for_graph = get_memory_tools(get_llm(llm_provider), sync_mode=sync_mode, store=memory_store)

    log(str(mcp_tools))
    normal_tools: list[Any] = [
        search_knowledge,
        search_schema,
        show_schema,
        call_sql_graph_tool,
        data_analysis_tool,
        run_python_code,
        save_report,
    ]
    if memory_tools_for_graph:
        normal_tools.extend(memory_tools_for_graph)
    # 预测、异常检测和归因分析统一交给 data_analysis 子 Agent 处理；
    # timeseries_forecast 等工具及其健康检查由子 Agent 管理，
    # 因此这里不直接向主 Agent 暴露这些工具。
    normal_tools.extend(mcp_tools)

    # 如果启用上下文管理，就创建上下文管理器
    context_manager = None
    if enable_context_management:
        context_manager = ContextManager(llm=get_llm(llm_provider), config=get_context_config())

    tool_node = ToolNode(normal_tools)

    # 创建主 Agent 的状态图
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    # 向状态图注册各个执行节点
    agent_tools: list[Any] = [*normal_tools, AskHuman]
    graph.add_node("llm_node", agent_llm_call(get_llm(llm_provider), agent_tools, context_manager))
    graph.add_node("ask_human", ask_human)
    graph.add_node("use_tool", tool_node)

    # 定义节点之间的固定执行路径
    graph.add_edge(START, "llm_node")
    graph.add_edge("ask_human", "llm_node")
    graph.add_edge("use_tool", "llm_node")

    # 为 llm_node 设置条件路由，根据状态决定下一步执行哪个节点
    def route_tools(state: AgentState) -> str | list[Send]:
        # 只有最新消息是带工具调用的 AI 消息时，才使用 sends 分派任务
        last_message = state["messages"][-1] if state["messages"] else None
        if (
            last_message
            and isinstance(last_message, AIMessage)
            and last_message.tool_calls
            and "sends" in state
            and state["sends"]
        ):
            return state["sends"]  # 返回 Send 任务列表，由 LangGraph 分派执行
        next_node = state.get("agent_next_node")
        if next_node:
            return next_node  # 返回单个目标节点名称
        return END
    # llm_node 执行完 → 调用 route_tools() → 根据返回值找到目标节点
    graph.add_conditional_edges(
        "llm_node",      # ① 从哪个节点出发
        route_tools,      # ② 使用哪个函数判断下一步
        {                   # ③ 路由映射：将判断结果对应到实际节点
            "llm_node": "llm_node",
            "ask_human": "ask_human",
            "use_tool": "use_tool",
            END: END,
        },
    )

    compiled_graph = graph.compile(name="agent_graph", checkpointer=checkpointer, store=memory_store)
    return compiled_graph


def build_agent_graph_sync(
    catalog: CatalogStore,
    checkpointer: Checkpointer | None = None,
    memory_store: BaseStore | None = None,
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """构建包含所有节点和连线的主 Agent 图（同步版本）。

    参数：
        catalog：保存数据库 Schema 信息的目录存储。
        checkpointer：保存执行状态的检查点组件（短期记忆）；为 None 时不启用。
        memory_store：长期记忆存储；为 None 时根据同步模式自动分配。
        enable_context_management：是否启用长对话的上下文管理。

    返回值：
        CompiledStateGraph：编译完成、可直接执行的 Agent 图。
    """
    # 初始化同步模式使用的 MCP 工具
    mcp_tools = create_mcp_tools_sync(_get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=True,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=None,  # 同步版本传入 None，由内部自行创建记忆工具
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
        llm_provider=llm_provider,
    )

async def build_agent_graph_async(
    catalog: CatalogStore,
    checkpointer: Checkpointer | None = None,
    memory_store: BaseStore | None = None,
    memory_tools: list[Any] | None = None,
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """构建包含所有节点和连线的主 Agent 图（异步版本）。

    与同步版本的构建逻辑相同，但会以异步方式初始化 MCP 工具，
    适用于异步调用环境。

    参数：
        catalog：保存数据库 Schema 信息的目录存储。
        checkpointer：保存执行状态的检查点组件（短期记忆）；为 None 时不启用。
        memory_store：长期记忆存储；为 None 时根据同步模式自动分配。
        memory_tools：记忆管理和搜索工具列表；为 None 时自行创建异步工具。
        enable_context_management：是否启用长对话的上下文管理。

    返回值：
        CompiledStateGraph：编译完成、可直接执行的 Agent 图。
    """
    # 异步初始化 MCP 工具
    mcp_tools = await get_mcp_tools_async(_get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=False,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=memory_tools,
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
        llm_provider=llm_provider,
    )
