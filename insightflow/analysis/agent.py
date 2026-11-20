"""Data Analysis Agent 的构建与工具封装。"""

import csv
import io
import logging
from typing import cast

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from insightflow.llm.llm import get_analysis_llm
from insightflow.attribution_answer import extract_evidence, render
from insightflow.tool.adtributor_tool import adtributor_drilldown
from insightflow.tool.anomaly_detection import anomaly_detection
from insightflow.tool.run_python_code import run_python_code
from insightflow.tool.search_knowledge import search_schema, show_schema
from insightflow.tool.timeseries_forecast import check_forecast_service_health, timeseries_forecast

logger = logging.getLogger(__name__)


class AttributionEvidence(AgentMiddleware):
    """只有本轮子 Agent 已取得查询明细，才允许归因工具消费数据。"""

    attribution_tools = frozenset({"search_schema", "show_schema", "text2sql", "adtributor_drilldown"})

    @staticmethod
    def is_attribution(messages):
        task = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        return task is not None and task.name == "attribution_request"

    def select_tools(self, request):
        if not self.is_attribution(request.state["messages"]):
            return request
        return request.override(tools=[t for t in request.tools if getattr(t, "name", None) in self.attribution_tools])

    def wrap_model_call(self, request, handler):
        return handler(self.select_tools(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.select_tools(request))

    @staticmethod
    def missing_evidence(request):
        if (AttributionEvidence.is_attribution(request.state["messages"])
                and request.tool_call["name"] not in AttributionEvidence.attribution_tools):
            return ToolMessage(content="本次委派仅处理指标变动贡献。请用 Text2SQL 整理两期数据，再交给 Adtributor。",
                               tool_call_id=request.tool_call["id"], name=request.tool_call["name"], status="error")
        if request.tool_call["name"] != "adtributor_drilldown":
            return None
        for message in reversed(request.state["messages"]):
            if isinstance(message, HumanMessage):
                break
            if not isinstance(message, ToolMessage) or message.name != "text2sql" or message.status == "error":
                continue
            # SQL 工具以带表头的 CSV 返回结果；失败提示和仅含表头的空集都不能作为归因输入。
            content = message.content
            if isinstance(content, str) and "**SQL_ERROR**" not in content:
                _, marker, tail = content.partition("Query Results (CSV format):\n```csv\n")
                body, end, _ = tail.partition("\n```")
                if marker and end:
                    rows = csv.reader(io.StringIO(body))
                    if next(rows, None) and next(rows, None):
                        return None
        return ToolMessage(content="本次归因还没有可用的查询明细。请先用 text2sql 取得各分组的基线与实际值；没有数据时说明原因。",
                           tool_call_id=request.tool_call["id"], name="adtributor_drilldown", status="error")

    def wrap_tool_call(self, request, handler):
        return self.missing_evidence(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self.missing_evidence(request) or await handler(request)


def _build_sub_agent_config(config: RunnableConfig | None) -> RunnableConfig:
    """
    为 Data Analysis Agent 创建独立配置，避免与主 Agent 的状态和流式事件混淆。

    1. thread_id：父子 Agent 可能共享同一个 Checkpointer，
    因此根据主 Agent 的 thread_id 派生独立的子 Agent thread_id，
    避免双方的检查点状态相互干扰。

    2. checkpoint_ns：固定为 "data_analysis"，用于保留子 Agent 的
    流式事件层级，避免事件被合并到主 Agent 的第 0 层，
    使前端能够区分主 Agent 和子 Agent 的执行过程。

    3. checkpoint_id：重置父 Agent 传入的检查点 ID，
    避免子 Agent 错误继承父 Agent 的状态快照。
    检查点隔离主要由独立的 thread_id 和 checkpoint_id 重置实现。

    其他配置（callbacks、tags、metadata）保持不变，直接传递给子 Agent。
    """
    sub_config: RunnableConfig = cast(RunnableConfig, {})
    configurable: dict = {}
    if config:
        sub_config = cast(RunnableConfig, {k: v for k, v in config.items() if k != "configurable"})
        ## 单独复制 configurable 以便修改子 Agent 的线程和检查点配置，同时保留其他配置不变，避免影响主 Agent。
        configurable = dict(config.get("configurable") or {})

    parent_thread_id = configurable.get("thread_id")
    if parent_thread_id is not None:
        # 根据主 Agent 的线程 ID，生成子 Agent 独立的线程 ID，避免检查点状态混淆
        configurable["thread_id"] = f"{parent_thread_id}:data_analysis"
        # 设置子 Agent 的命名空间，使其流式事件以嵌套子图的形式展示
        configurable["checkpoint_ns"] = "data_analysis"

    # 清除父Agent传入的checkpoint_id，确保子Agent使用自身线程的检查点，避免沿用父Agent的检查点。
    configurable.pop("checkpoint_id", None)
    # 将修改后的 configurable 放回子 Agent 的运行配置中
    sub_config["configurable"] = configurable
    return sub_config


def _extract_final_content(response: dict) -> str:
    """提取 Agent 响应的最后一条消息，并将其内容转换为字符串。"""
    if isinstance(response, dict) and response.get("messages"):
        content = response["messages"][-1].content
        if isinstance(content, str):
            return content
        # # 若消息内容包含多个内容块（如多模态数据），则提取并拼接其中的文本。
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", "") or "")
            return "\n".join(p for p in parts if p)
        return str(content)
    return str(response)


def _resolve_min_input_length() -> int:
    """将最少历史数据条数提供给 Agent，读取失败时采用默认值。"""
    from insightflow import config
    from insightflow.tool.timeseries_forecast import DEFAULT_MIN_FORECAST_INPUT_LENGTH, get_min_forecast_input_length

    try:
        service_url = config.get().timeseries_forecasting_service_url
        return get_min_forecast_input_length(service_url)
    except Exception:
        return DEFAULT_MIN_FORECAST_INPUT_LENGTH


def _load_data_analysis_prompt() -> str:
    """加载数据分析 Agent 的 Prompt 模板，并填入最少历史数据条数。"""
    import os

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "data_analysis_prompt.md")
    try:
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logger.warning(f"Data analysis prompt file not found at {prompt_path}")
        return "你是数据分析 Agent。请依据用户提供的数据和任务要求开展分析，并如实说明结论。"

    return template.replace("[min_input_length]", str(_resolve_min_input_length()))


def build_data_analysis_agent(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> CompiledStateGraph:
    """构建数据分析 Agent。

    Args:
        sql_graph: `text2sql` 工具使用的已编译 SQL 生成图。
        sync_mode: 是否使用同步模式。
        llm_provider: 要使用的模型提供方。
        checkpointer: 用于持久化状态的 Checkpointer。
        memory_store: 长期记忆存储。

    Returns:
        CompiledStateGraph: 已编译的数据分析 Agent。
    """
    # 在此导入以避免循环进口
    from insightflow.agent_graph import get_sql_tools

    # 1. 准备工具
    text2sql_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    tools = [search_schema, show_schema, text2sql_tool, run_python_code]

    # 如果服务健康，增加预报和异常检测
    if check_forecast_service_health():
        tools.append(timeseries_forecast)
        tools.append(anomaly_detection)
    else:
        logger.warning("Time series forecasting service is not healthy. Skipping forecast and anomaly detection tools.")

    tools.append(adtributor_drilldown)

    # 2. 获取 LLM (如果配置了 analysis_llm ,否则默认)
    llm = get_analysis_llm(llm_provider)

    # 3. 快点
    system_prompt = _load_data_analysis_prompt()

    # 4. 创建代理(在深层代理框架上构建)
    # 注:深剂. create_deep_agent 期望一个模型字符串或 BaseChatModel
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[AttributionEvidence()],
        checkpointer=checkpointer,
        store=memory_store,
    )

    return agent


class DataAnalysisInput(BaseModel):
    """数据分析工具的输入结构。"""

    reasoning: str = Field(description="说明为什么需要委派给数据分析 Agent")
    attribution_only: bool = Field(default=False, description="专门拆解指标变化贡献时设为 true，其他专业分析保留通用工具。")
    task: str = Field(
        description=(
            "请提供完整、独立且明确的分析任务说明。数据分析 Agent 不知道当前时间和对话上下文，"
            "因此必须把“上周”“今天”“昨天”等相对时间换算成具体日期，并明确指标、维度、粒度、时间范围和分析类型。"
            "异常检测还要注明关注的检测区间，以及数据分析窗口的结束日期。"
        )
    )


def get_data_analysis_tool(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> StructuredTool:
    """将数据分析 Agent 封装为可委派的工具。

    Args:
        sql_graph: 已编译的 SQL 生成图。
        sync_mode: 是否使用同步模式。
        llm_provider: 要使用的模型提供方。
        checkpointer: 用于持久化状态的 Checkpointer。
        memory_store: 长期记忆存储。

    Returns:
        StructuredTool: 可由主 Agent 调用的数据分析工具。
    """
    agent = build_data_analysis_agent(
        sql_graph=sql_graph,
        sync_mode=sync_mode,
        llm_provider=llm_provider,
        checkpointer=checkpointer,
        memory_store=memory_store,
    )

    def tool_result(response, task, attribution_only):
        evidence = extract_evidence(response["messages"]) if attribution_only else None
        if evidence:
            prose, structured = render(evidence, task)
            return prose + ("\n"+structured if structured else ""), evidence
        if attribution_only and any(isinstance(m, AIMessage) and any(
            c["name"] == "adtributor_drilldown" and not c["args"].get("derived")
            for c in m.tool_calls) for m in response["messages"]):
            return "归因分组未能与同一份查询明细完整核对，暂无法确认汇总数值。", None
        return _extract_final_content(response), None

    def call_data_analysis_sync(reasoning: str, task: str, config: RunnableConfig, attribution_only: bool = False) -> tuple:
        """以同步模式调用数据分析 Agent。"""
        logger.info(f"Delegating to data analysis agent (sync). Reasoning: {reasoning}, Task: {task}")
        import traceback

        from langgraph.errors import GraphInterrupt

        sub_config = _build_sub_agent_config(config)
        try:
            # 代理接受 "messages" 下的字符串，该字符串被强制使用 HumanMessage .
            message = HumanMessage(content=task, name="attribution_request") if attribution_only else task
            response = agent.invoke({"messages": message}, config=sub_config)
            return tool_result(response, task, attribution_only)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return f"Error occurred during data analysis: {str(e)}", None

    async def call_data_analysis_async(reasoning: str, task: str, config: RunnableConfig, attribution_only: bool = False) -> tuple:
        """以异步模式调用数据分析 Agent。"""
        logger.info(f"Delegating to data analysis agent (async). Reasoning: {reasoning}, Task: {task}")
        import traceback

        from langgraph.errors import GraphInterrupt

        sub_config = _build_sub_agent_config(config)
        try:
            message = HumanMessage(content=task, name="attribution_request") if attribution_only else task
            response = await agent.ainvoke({"messages": message}, config=sub_config)
            return tool_result(response, task, attribution_only)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return f"Error occurred during data analysis: {str(e)}", None

    description = """委派经营指标归因、异常检测、趋势预测、多指标相关性或组合分析。
分析哪些业务分组驱动指标变化时，将完整问题交给本工具。子 Agent 自行查询业务库，
再用 adtributor_drilldown 分解贡献。普通明细、简单聚合和排名由主 Agent 查询。
请写明指标口径、分析维度、基线与观察区间、用户要求的交付结果，并把相对日期换成明确日期。"""

    if sync_mode:
        return StructuredTool.from_function(
            func=call_data_analysis_sync,
            response_format="content_and_artifact",
            name="data_analysis",
            description=description,
            args_schema=DataAnalysisInput,
            return_direct=False,
        )
    else:
        return StructuredTool.from_function(
            coroutine=call_data_analysis_async,
            response_format="content_and_artifact",
            name="data_analysis",
            description=description,
            args_schema=DataAnalysisInput,
            return_direct=False,
        )
