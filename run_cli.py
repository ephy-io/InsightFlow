#!/usr/bin/env python
"""在命令行中运行 InsightFlow Agent，可用于端到端测试。

直接在进程内运行 Agent 图，通过 :mod:`insightflow.streaming`
输出与 Streamlit 一致的中间步骤，包括选表、SQL 生成与执行、可视化、子 Agent 进度和工具调用。

示例
----
使用同步图提问并查看流式步骤::

    CONFIG_FILE=example/config.yaml python run_cli.py "How many users signed up last week?"

联调 Text2SQL 和异常检测工具::

    CONFIG_FILE=example/config.yaml python run_cli.py "Are there anomaly drop of daily orders in last week？"

交互模式沿用同一线程和会话，支持恢复人工澄清::

    CONFIG_FILE=example/config.yaml python run_cli.py

输出 NDJSON 事件供自动化检查，每行一个 JSON 对象::

    python run_cli.py --json "show me revenue by region" > events.ndjson

使用异步图并仅输出最终答案::

    python run_cli.py --async --no-stream "..."
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from insightflow.streaming import (
    AgentStreamProcessor,
    StreamInterrupt,
    StreamStep,
    StreamToken,
    StreamUsage,
    extract_final_answer,
)


# --------------------------------------------------------------------------- #
# 图表构造
# --------------------------------------------------------------------------- #
def build_sync_graph(provider: str | None):
    """使用内存 Checkpointer 构建同步 Agent 图。"""
    from langgraph.checkpoint.memory import MemorySaver

    from insightflow import config
    from insightflow.agent_graph import build_agent_graph_sync
    from insightflow.tool.memory import get_sync_memory_store

    return build_agent_graph_sync(
        config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=get_sync_memory_store(),
        llm_provider=provider,
    )


async def build_async_graph(provider: str | None):
    """使用内存 Checkpointer 构建异步 Agent 图。"""
    from langgraph.checkpoint.memory import MemorySaver

    from insightflow import config
    from insightflow.agent_graph import build_agent_graph_async
    from insightflow.tool.memory import get_async_memory_store

    return await build_agent_graph_async(
        config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=await get_async_memory_store(),
        llm_provider=provider,
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
class _Color:
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


class CliRenderer:
    """按可读文本或 NDJSON 将流式事件输出到终端。"""

    def __init__(self, as_json: bool = False, color: bool = True) -> None:
        self.as_json = as_json
        self.color = color and not as_json
        self._step_no = 0
        # 音轨，级别，标签，is_final) 级别为当前符号运行的音轨
        # 属于，所以连续的符号共用一个信头。
        self._token_layer: tuple | None = None
        self._on_token_line = False

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{_Color.RESET}" if self.color else text

    def _emit_json(self, event) -> None:
        if isinstance(event, StreamStep):
            payload = {
                "type": "step",
                "kind": event.kind,
                "level": event.level,
                "label": event.label,
                "text": event.text,
            }
        elif isinstance(event, StreamToken):
            payload = {
                "type": "token",
                "level": event.level,
                "label": event.label,
                "is_final": event.is_final,
                "text": event.text,
            }
        elif isinstance(event, StreamUsage):
            payload = {
                "type": "usage",
                "turn_tokens": event.turn_tokens,
                "turn_cost_usd": event.turn_cost_usd,
                "by_model": event.by_model,
            }
        else:  # StreamInterrupt 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基百科中的相关条目: 维基月球在线解说- 维基月球在线解说- 维基月球
            payload = {"type": "interrupt", "text": event.text, "buttons": event.buttons}
        print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)

    def render(self, event) -> None:
        if self.as_json:
            self._emit_json(event)
            return

        if isinstance(event, StreamToken):
            layer = (event.level, event.label, event.is_final)
            if self._token_layer != layer:
                self._end_token_line()
                if event.is_final:
                    header = self._c("\n🤖 AI Response: ", _Color.BOLD + _Color.GREEN)
                else:
                    indent = "  " * event.level
                    header = self._c(f"\n{indent}💭 {event.label} 思考: ", _Color.DIM)
                sys.stdout.write(header)
                self._token_layer = layer
            sys.stdout.write(event.text)
            sys.stdout.flush()
            self._on_token_line = True

        elif isinstance(event, StreamStep):
            self._end_token_line()
            self._token_layer = None
            if event.level == 0:
                self._step_no += 1
                prefix = self._c(f"Step {self._step_no}: ", _Color.BOLD + _Color.CYAN)
                print(f"\n{prefix}{event.text}")
            else:
                indent = "  " * event.level
                tag = self._c(f"↳ [{event.label}] ", _Color.DIM)
                print(f"\n{indent}{tag}{event.text}")

        elif isinstance(event, StreamUsage):
            self._end_token_line()
            self._token_layer = None
            cost_text = f"~${event.turn_cost_usd:.4f}" if event.turn_cost_usd is not None else "费用未知"
            line = self._c(f"Turn: {event.turn_tokens} tokens ({cost_text})", _Color.DIM)
            print(f"\n{line}")

        elif isinstance(event, StreamInterrupt):
            self._end_token_line()
            self._token_layer = None
            box = self._c("⏸  Interrupt (input required):", _Color.BOLD + _Color.YELLOW)
            print(f"\n{box} {event.text}")
            if event.buttons:
                print(self._c(f"   options: {event.buttons}", _Color.DIM))

    def _end_token_line(self) -> None:
        if self._on_token_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._on_token_line = False

    def final_answer(self, text: str) -> None:
        if self.as_json:
            if text:
                print(json.dumps({"type": "final_answer", "text": text}, ensure_ascii=False), flush=True)
            return
        self._end_token_line()
        if text:
            banner = self._c("═" * 60, _Color.DIM)
            label = self._c("FINAL ANSWER", _Color.BOLD + _Color.GREEN)
            print(f"\n{banner}\n{label}\n{banner}\n{text}\n")


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# 转动执行( 同步 + 自动执行)
# --------------------------------------------------------------------------- #
def run_turn_sync(graph, stream_input, config, renderer: CliRenderer, stream: bool) -> bool:
    """同步执行一轮，已中断时返回 True。"""
    processor = AgentStreamProcessor()
    if stream:
        for namespace, event_type, event_value in graph.stream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                renderer.render(event)
    else:
        graph.invoke(stream_input, config=config)

    state = graph.get_state(config)
    return _handle_state(state, processor, renderer)


async def run_turn_async(graph, stream_input, config, renderer: CliRenderer, stream: bool) -> bool:
    """异步执行一轮，已中断时返回 True。"""
    processor = AgentStreamProcessor()
    if stream:
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                renderer.render(event)
    else:
        await graph.ainvoke(stream_input, config=config)

    state = await graph.aget_state(config)
    return _handle_state(state, processor, renderer)


def _handle_state(state, processor: AgentStreamProcessor, renderer: CliRenderer) -> bool:
    """根据工作流结束时的状态显示中断提示或最终回答。"""
    if state.interrupts:
        value = state.interrupts[0].value or {}
        renderer.render(StreamInterrupt(text=value.get("text", ""), buttons=value.get("buttons", []) or []))
        return True

    # 在最后答案之前，每回送信/成本滚动。
    usage = processor.emit_turn_usage()
    if usage is not None:
        renderer.render(usage)

    final = extract_final_answer(processor.final_response)
    if not final:
        # 返回到最后一个消息内容，当没有抓取流回的答案时
        # (如 --no-stream ,或非流式最终节点).
        values = getattr(state, "values", None) or {}
        messages = values.get("messages") or []
        if messages:
            final = str(getattr(messages[-1], "content", "") or "")
    renderer.final_answer(final)
    return False


# --------------------------------------------------------------------------- #
# 入境点
# --------------------------------------------------------------------------- #
def _initial_input(message: str):
    return {"messages": [{"role": "user", "content": message}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless CLI to run the InsightFlow agent end-to-end (streams intermediate steps)."
    )
    parser.add_argument("question", nargs="?", help="Question to ask. Omit to start an interactive REPL.")
    parser.add_argument("--user-id", default="cli", help="User id (default: cli).")
    parser.add_argument("--session-id", default="cli", help="Session id (default: cli).")
    parser.add_argument("--provider", default=None, help="LLM provider name (optional).")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use the async graph.")
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false", help="Disable step streaming; print only the final answer."
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit NDJSON events instead of human-readable output."
    )
    parser.add_argument("--no-color", dest="color", action="store_false", help="Disable ANSI colors.")
    args = parser.parse_args(argv)

    renderer = CliRenderer(as_json=args.as_json, color=args.color)
    from insightflow.observability.tracing import build_run_config

    config = build_run_config(user_id=args.user_id, session_id=args.session_id)

    if args.use_async:
        return asyncio.run(_run_async(args, config, renderer))
    return _run_sync(args, config, renderer)


def _iter_questions(first: str | None):
    """先读取命令行中给定的问题，再按行读取标准输入中的交互问题。"""
    if first is not None:
        yield first
        return
    print("InsightFlow CLI — type a question (Ctrl-D / 'exit' to quit).", file=sys.stderr)
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if line.lower() in {"exit", "quit"}:
            break
        if line:
            yield line


def _run_sync(args, config, renderer: CliRenderer) -> int:
    from langgraph.types import Command

    graph = build_sync_graph(args.provider)
    interrupted = False
    for question in _iter_questions(args.question):
        stream_input = Command(resume=question) if interrupted else _initial_input(question)
        interrupted = run_turn_sync(graph, stream_input, config, renderer, args.stream)
    return 0


async def _run_async(args, config, renderer: CliRenderer) -> int:
    from langgraph.types import Command

    graph = await build_async_graph(args.provider)
    interrupted = False
    for question in _iter_questions(args.question):
        stream_input = Command(resume=question) if interrupted else _initial_input(question)
        interrupted = await run_turn_async(graph, stream_input, config, renderer, args.stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
