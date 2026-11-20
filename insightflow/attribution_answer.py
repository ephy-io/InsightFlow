"""从本次归因的 SQL 明细和实际工具调用中提取可核对的数值。"""

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import io
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def extract_evidence(messages):
    queries = []
    calls = {}
    evidence = None
    for message in messages:
        if isinstance(message, HumanMessage):
            queries, calls, evidence = [], {}, None
        elif isinstance(message, AIMessage):
            calls.update({c["id"]: c for c in message.tool_calls})
        elif isinstance(message, ToolMessage) and message.status != "error":
            content = str(message.content)
            if message.name == "text2sql":
                block = re.search(r"Query Results \(CSV format\):\s*```csv\s*\n(.*?)\n```", content, re.S)
                if block:
                    queries.append([r for r in csv.reader(io.StringIO(block[1].replace("\r",""))) if r][1:])
            elif message.name == "adtributor_drilldown" and message.tool_call_id in calls:
                args = calls[message.tool_call_id]["args"]
                result = json.loads(content)
                if result.get("status") and not args.get("derived"):
                    evidence = match_groups(queries, args["data"], args.get("issue_type", "drop"))
    return evidence


def match_groups(queries, data, direction):
    if not data or len({r["dimension_name"] for r in data}) != 1:
        return None
    groups = {str(r["element_value"]): [str(r["predict"]), str(r["real"])] for r in data}
    if len(groups) != len(data):
        return None
    # 宽表要求每组各有一行，长表要求各组共享同一对有序期间，且两个值逐项一致。
    for rows in reversed(queries):
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            continue
        width = len(rows[0])
        for group_col in range(width):
            if {r[group_col] for r in rows} != groups.keys():
                continue
            for a in range(width):
                for b in range(width):
                    if len({a,b,group_col}) != 3:
                        continue
                    try:
                        if len(rows) == len(groups):
                            pairs = {r[group_col]: (Decimal(r[a]), Decimal(r[b])) for r in rows}
                        else:
                            periods = sorted({r[a] for r in rows})
                            if len(periods)!=2 or len(rows)!=2*len(groups):
                                continue
                            values = {(r[group_col],r[a]):Decimal(r[b]) for r in rows}
                            pairs = {g:tuple(values[g,t] for t in periods) for g in groups}
                        if all(tuple(map(Decimal,groups[g])) == pair for g,pair in pairs.items()):
                            return {"groups":groups,"direction":direction}
                    except (InvalidOperation, KeyError):
                        continue
    return None


def render(evidence, question):
    groups = [(g,Decimal(a),Decimal(b)) for g,(a,b) in evidence["groups"].items()]
    base = sum(a for _,a,_ in groups)
    actual = sum(b for _,_,b in groups)
    delta = actual-base
    sign = -1 if evidence["direction"] == "drop" else 1
    groups.sort(key=lambda row:sign*(row[2]-row[1]), reverse=True)

    def fmt(x):
        return format(x.quantize(Decimal(".01"), rounding=ROUND_HALF_UP),"f")

    lines = [f"基期合计 {fmt(base)}，当期合计 {fmt(actual)}，净变化 {fmt(delta)}。",
             "| 分组 | 基期 | 当期 | 差额 | 占净变化 |","|---|---:|---:|---:|---:|"]
    for g,a,b in groups:
        share = fmt((b-a)*100/delta)+"%" if delta else "无定义"
        lines.append(f"| {g} | {fmt(a)} | {fmt(b)} | {fmt(b-a)} | {share} |")
    g,a,b = groups[0]
    key = re.search(r"键(?:名)?为\s*[`“\"]?([\w-]+)",question)
    fields = re.search(r"(?:按|格式为|值为)\s*\[([^\[\]]+)\]",question)
    unique = len(groups)==1 or b-a != groups[1][2]-groups[1][1]
    if unique and sign*(b-a)>0:
        lines.append(f"沿该变化方向贡献最大的分组为{g}，绝对变化 {fmt(abs(b-a))}。")
        if key and fields and re.search(r"最大|最多|最少|最强",question) and not re.search(r"每行|每条|多行|各返回",question):
            values = []
            for field in re.split(r"[,，、]",fields[1]):
                if re.search(r"名称|分组|渠道|状态|地区|品类",field):
                    values.append(json.dumps(g,ensure_ascii=False))
                elif re.search(r"基[期线]|对照",field):
                    values.append(fmt(a))
                elif re.search(r"当期|实际|观察期",field):
                    values.append(fmt(b))
                elif re.search(r"贡献率|占比",field) and delta and "%" in field:
                    values.append(fmt((b-a)*100/delta))
                elif re.search(r"绝对|减少量|增额",field):
                    values.append(fmt(abs(b-a)))
                elif re.search(r"差额|净变化",field):
                    values.append(fmt(b-a))
                else:
                    return "\n".join(lines), None
            return "\n".join(lines), "{"+json.dumps(key[1],ensure_ascii=False)+":["+",".join(values)+"]}"
    return "\n".join(lines), None
