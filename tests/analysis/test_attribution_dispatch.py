from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from insightflow.agent_graph import agent_llm_call
from insightflow.analysis.agent import AttributionEvidence
from insightflow.final_json import check_final_json
from insightflow.streaming import AgentStreamProcessor, StreamToken


QUESTION = "附 JSON 对象，键为 payload。按 [名称, 金额] 返回。"


@pytest.mark.parametrize("route", ["general", "attribution"])
def test_semantic_decision_controls_dispatch(route):
    llm = Mock()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="", tool_calls=[
        {"name": "TaskIntent", "args": {"route": route, "task": "完整业务范围"}, "id": "route"}
    ])
    with patch("insightflow.agent_graph.call_llm_chat_model_with_retry", return_value=AIMessage(content="答复")) as normal:
        result = agent_llm_call(llm, [SimpleNamespace(name="data_analysis")])({"messages": [HumanMessage("需求")]})
    if route == "attribution":
        call = result["messages"][0].tool_calls[0]
        assert call["name"] == "data_analysis" and call["args"]["attribution_only"]
        assert not normal.called
    else:
        assert normal.called and result["final_answer"] == "答复"


def test_attribution_offer_and_execution_are_limited_to_current_delegation():
    guard = AttributionEvidence()
    req = Mock()
    req.tools = [SimpleNamespace(name=n) for n in ("run_python_code", "adtributor_drilldown", "text2sql")]
    req.state = {"messages": [HumanMessage("任务", name="attribution_request")]}
    req.tool_call = {"name": "run_python_code", "id": "python"}
    guard.select_tools(req)
    assert [t.name for t in req.override.call_args.kwargs["tools"]] == ["adtributor_drilldown","text2sql"]
    assert guard.missing_evidence(req).status == "error"
    req.state["messages"].append(HumanMessage("其他分析"))
    assert guard.select_tools(req) is req and guard.missing_evidence(req) is None


def test_fenced_json_preserves_values_and_prose():
    raw = '核对金额。\n```json\n{"payload":[["名称}]",1234.56000000000001]]\n```'
    assert check_final_json(raw, QUESTION) == '核对金额。\n```json\n{"payload":["名称}]",1234.56000000000001]}\n```'


@pytest.mark.parametrize("raw", ['{"payload": ["A",12', '{"payload":[null,2],"payload":[1,2]}',
                                 '{"other":[1,2]}', '{"payload":[[1,2,3]]}', '{"payload":[1,NaN]}'])
def test_ambiguous_or_truncated_value_is_not_rewritten(raw):
    assert check_final_json(raw, QUESTION).startswith(raw + "\n\n格式校验未通过")


def test_nested_and_multirow_data_preserved():
    raw = '{"payload":[["A",2]]}'
    assert check_final_json(raw, "附 JSON 对象，键为 payload。每行按 [名称, 金额] 返回。") == raw


def test_stream_delivers_validated_answer():
    processor = AgentStreamProcessor()
    msg = AIMessage(content='{"a":1}', additional_kwargs={"validated_json_delivery":True})
    events = list(processor.process((), "updates", {"llm_node":{"messages":[msg],"final_answer":msg.content}}))
    assert [e.text for e in events if isinstance(e, StreamToken)] == [msg.content] == [processor.final_response]
