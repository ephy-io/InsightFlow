"""从 LangGraph 检查器中恢复聊天历史，用于样本 UI。

该代理在 SQLite 中坚持完整对话状态( ``messages``)
由 ``thread_id = "{user_id}-{session_id}"`` 键入的检查器。 样本 UI
只渲染在浏览器/会话内存中保存的消息，因此一个页面刷新丢失
尽管代理商本身仍有上下文，但可见的历史。

这个模块将空隙连接起来:它读取了持续 ``messages`` 通过
LangGraph ``get_state`` API (储存-不可知) 并将其转换为
UIs 渲染的简化消息。中间工具调用和
UI -只有流体文物(思考步骤，图表)是故意的
已丢弃 - 恢复视图显示用户提示和 AI 文本回复。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from insightflow.observability.tracing import build_run_config
from insightflow.utils import log


def _extract_text(content: Any) -> str:
    """从 ``content`` 消息中提取纯文本( 字符串或内容块)。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain 内容块: { "type" : "text" , "text" : "..." }.
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def messages_to_ui_history(messages: Any) -> list[dict]:
    """将检查点 LangChain 消息转换为 UI 历史消息 dicts.

    绘图规则( 简化视图) :
    - ``HumanMessage`` -> ``{"role": "user", "type": "text", "content": <text>}``
    - 带有非空文本的 ``AIMessage`` ->
      ``{"role": "assistant", "type": "text", "content": <text>}`` 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基月球在线解说- 维基百科中的相关条目: 维基月球在线解说- 维基月球在线解说- 维基月球
    - ``ToolMessage`` / ``SystemMessage`` / 工具调用只使用 ``AIMessage`` /
      空内容消息被跳过。

    原始顺序保存。 空/ None 输入返回 ``[]``。
    """
    if not messages:
        return []

    history: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = _extract_text(message.content)
            if text:
                history.append({"role": "user", "type": "text", "content": text})
        elif isinstance(message, AIMessage):
            text = _extract_text(message.content)
            # 跳过纯工具调用转折(无用户调用文本)。
            if text:
                history.append({"role": "assistant", "type": "text", "content": text})
        # ToolMessage , SystemMessage 和其他任何东西被故意丢弃。
    return history


def _ui_history_to_tuples(history: list[dict]) -> list[tuple[str, str]]:
    """将订购的简化式 UI 历史配为 ``(user, assistant)`` 礼服。

    配对规则(容忍非限制性交替):
    - 一个 ``user`` 消息打开一个新的 ``(user, "")`` 槽。
    - ``assistant`` 消息如果助手的侧面是空的，则会填充最后一个空位，
      否则打开一个新的 ``("", assistant)`` 槽。
    """
    pairs: list[list[str]] = []
    for item in history:
        if item["role"] == "user":
            pairs.append([item["content"], ""])
        else:  # 助理人员
            if pairs and pairs[-1][1] == "":
                pairs[-1][1] = item["content"]
            else:
                pairs.append(["", item["content"]])
    return [(user, assistant) for user, assistant in pairs]


def messages_to_chat_tuples(messages: Any) -> list[tuple[str, str]]:
    """将检查点消息转换为 Gradio ``type="tuples"`` 聊天对。

    重用 : func: `messages_to_ui_history` 作为真理的单一来源( 相同)
    过滤/排序为 Streamlit 视图，然后将命令的简化对齐
    将消息整理为 ``(user, assistant)`` 元组。空输入和 None 均返回 ``[]``。
    """
    return _ui_history_to_tuples(messages_to_ui_history(messages))


async def load_session_history(graph: Any, user_id: str, session_id: str) -> list[dict]:
    """作为 UI 消息dicts 的 ``{user_id}-{session_id}`` 载入持续历史。

    返回 ``[]`` 当线程没有检查点，状态没有消息，
    读取失败( graceful decovery - 永不阻断对话)。
    """
    try:
        config = build_run_config(user_id=user_id, session_id=session_id)
        state = await graph.aget_state(config)
        if state is None:
            return []
        values = getattr(state, "values", None) or {}
        messages = values.get("messages", [])
        return messages_to_ui_history(messages)
    except Exception as exc:  # pragma: no cover - defensive degradation
        log(f"Failed to load session history for {user_id}-{session_id}: {exc!r}")
        return []


async def load_session_history_tuples(graph: Any, user_id: str, session_id: str) -> list[tuple[str, str]]:
    """装入持续历史为 Gradio ``(user, assistant)`` 聊天燕尾服。

    复用:func: `load_session_history` 用于读作+优雅的退化，那么
    将结果对成拖曳。在缺少/空/未读时返回 ``[]``。
    """
    history = await load_session_history(graph, user_id, session_id)
    return _ui_history_to_tuples(history)
