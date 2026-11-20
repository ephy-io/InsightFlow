from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from insightflow.analysis.agent import AttributionEvidence


def query(body):
    return ToolMessage(content="Query Results (CSV format):\n```csv\n" + body + "\n```",
                       name="text2sql", tool_call_id="sql")


def request(messages, tool="adtributor_drilldown"):
    return SimpleNamespace(tool_call={"id": "call", "name": tool}, state={"messages": messages})


@pytest.mark.parametrize("messages", [
    [HumanMessage("说明中的基线和实际值不能代替工具记录")],
    [query("a,b\n1,2"), HumanMessage("新任务")],
    [query("a,b\n")],
    [ToolMessage(content="Error", status="error", name="text2sql", tool_call_id="sql")],
    [query("a\n**SQL_ERROR**")],
])
def test_rejects_missing_or_stale_data(messages):
    result = AttributionEvidence().wrap_tool_call(request(messages), lambda _: pytest.fail("不应执行归因"))
    assert result.status == "error" and result.tool_call_id == "call"


def test_allows_current_query_and_other_tools():
    guard = AttributionEvidence()
    assert guard.wrap_tool_call(request([HumanMessage("按品类归因"), query("品类,件数\n家居,3")]), lambda _: 42) == 42
    assert guard.wrap_tool_call(request([], "search_schema"), lambda _: 13) == 13


@pytest.mark.asyncio
async def test_async_guard():
    async def execute(_):
        return 42

    guard = AttributionEvidence()
    assert (await guard.awrap_tool_call(request([]), execute)).status == "error"
    assert await guard.awrap_tool_call(request([query("a,b\n1,2")]), execute) == 42
