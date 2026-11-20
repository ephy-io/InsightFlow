import json
from pathlib import Path

import pytest

from insightflow.attribution_answer import match_groups, render
from insightflow.final_json import check_final_json, check_sql_claims


@pytest.mark.parametrize("case",["route_01","route_02"])
def test_saved_attribution(case):
    fixture = Path(__file__).parent / "fixtures" / "verified_attribution.json"
    ref = json.loads(fixture.read_text(encoding="utf8"))[case]
    args = ref["args"]
    import csv
    sql = ref["sql_output"]
    rows=list(csv.reader(sql.split("```csv\n")[1].split("```")[0].strip().splitlines()))[1:]
    evidence=match_groups([[r for r in rows if r]],args["data"],args["issue_type"])
    prose,payload=render(evidence,ref["question"])
    assert json.loads(payload)["review"]==ref["expected"]
    if case=="route_02":
        assert all(x in prose for x in ["243358.34","28637.33","31.77%"])
    assert match_groups([rows[:-2]],args["data"],args["issue_type"]) is None


def test_zero_net_and_explicit_json_only():
    prose,_=render({"groups":{"甲":["1","2"],"乙":["2","1"]},"direction":"rise"},"按组查看")
    assert "无定义" in prose
    assert check_final_json('正文\n```json\n{ "键": "A B" }\n```',"只返回JSON")=='{"键":"A B"}'


def test_sql_conflict_uses_original():
    raw="Query Results (CSV format):\n```csv\n净差额\n12.35\n```"
    assert "12.35" in check_sql_claims("净差额为13.35",raw)
    assert check_sql_claims("净差额为12.35",raw)=="净差额为12.35"
