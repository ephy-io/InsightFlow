"""使用 FastAPI 和 Gradio 为 InsightFlow 列出 UI 的内存。"""

import json
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI

from sample_ui.style import custom_css


def get_thread_memory_store() -> Any:
    """建立线程安全的记忆存储连接。"""
    try:
        import pysqlite3 as sqlite3
    except ImportError:
        import sqlite3
    from langgraph.store.sqlite import SqliteStore

    from insightflow import config

    conn = sqlite3.connect("memory.db", check_same_thread=False)
    conn.isolation_level = None  # 使用自动承诺模式避免交易冲突
    store = SqliteStore(conn, index={"dims": 1536, "embed": config.get().embedding_model, "fields": ["text"]})
    try:
        store.setup()
    except Exception:
        pass  # 可能已经设置了仓库
    return store, conn


def list_all_memories() -> list[dict[str, Any]]:
    """从记忆库中获取所有记忆。

    Returns:
        带有元数据的内存项列表
    """
    try:
        memory_store, conn = get_thread_memory_store()
        memories = []

        try:
            # 使用带有部分命名空间的搜索来查找所有内存项
            items = memory_store.search(("memories",), limit=1000)
            for item in items:
                memory_data = {
                    "namespace": item.namespace,
                    "key": item.key,
                    "value": item.value,
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)
        except Exception as e:
            return [{"error": f"Failed to retrieve memories: {str(e)}"}]
        finally:
            conn.close()

        return memories

    except Exception as e:
        return [{"error": f"Failed to access memory store: {str(e)}"}]


def format_memories_for_display(memories: list[dict[str, Any]]) -> str:
    """在 Gradio 界面中格式化显示的记忆。

    Args:
        memories: 内存项目列表

    Returns:
        显示格式字符串
    """
    if not memories:
        return "No memories found."

    if len(memories) == 1 and "error" in memories[0]:
        return f"Error: {memories[0]['error']}"

    formatted = []
    for i, memory in enumerate(memories, 1):
        if "error" in memory:
            formatted.append(f"**Error:** {memory['error']}")
            continue

        formatted.append(f"## Memory {i}")
        formatted.append(f"**Namespace:** {memory['namespace']}")
        formatted.append(f"**Key:** {memory['key']}")

        # 正确格式化该值
        value = memory["value"]
        if isinstance(value, dict):
            try:
                value_str = json.dumps(value, indent=2)
                formatted.append(f"**Content:**\n```json\n{value_str}\n```")
            except Exception:
                formatted.append(f"**Content:** {str(value)}")
        else:
            formatted.append(f"**Content:** {str(value)}")

        formatted.append(f"**Created:** {memory['created_at']}")
        formatted.append(f"**Updated:** {memory['updated_at']}")
        formatted.append("---")

    return "\n".join(formatted)


def refresh_memories() -> list[list[str]]:
    """刷新并返回格式化的记忆。"""
    memories = list_all_memories()
    return format_memories_for_display(memories)


def delete_memory_by_key(namespace_str: str, key: str) -> str:
    """按命名空间和密钥删除内存。

    Args:
        namespace_str: 命名空间的字符串表示(例如"( 'memories' , 'user1')").
        key: 要删除的内存密钥

    Returns:
        状态消息
    """
    try:
        import ast

        memory_store, conn = get_thread_memory_store()

        try:
            # 解析命名空间字符串回到 Tuple
            namespace = ast.literal_eval(namespace_str)

            # 删除项目
            memory_store.delete(namespace, key)
            return f"Successfully deleted memory: {key} from namespace {namespace}"
        finally:
            conn.close()
    except Exception as e:
        return f"Failed to delete memory: {str(e)}"


# ---------- FastAPI 接口 ----------
app = FastAPI()

# ---------- Gradio 界面 ----------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Memory Store Viewer")
    gr.Markdown("View and manage long-term memories stored in the InsightFlow system.")

    with gr.Row():
        with gr.Column(scale=3):
            memories_display = gr.Markdown(value=refresh_memories(), elem_id="memories-display")

        with gr.Column(scale=1):
            gr.Markdown("### Actions")
            refresh_btn = gr.Button("🔄 Refresh Memories", variant="primary")

            gr.Markdown("### Delete Memory")
            namespace_input = gr.Textbox(
                label="Namespace",
                placeholder="('memories', 'user_id')",
                info="Copy the exact namespace from the memory list",
            )
            key_input = gr.Textbox(
                label="Key", placeholder="memory_key", info="Copy the exact key from the memory list"
            )
            delete_btn = gr.Button("🗑️ Delete Memory", variant="stop")
            delete_status = gr.Textbox(label="Status", interactive=False)

    # 事件处理器
    refresh_btn.click(fn=refresh_memories, outputs=[memories_display])

    delete_btn.click(fn=delete_memory_by_key, inputs=[namespace_input, key_input], outputs=[delete_status]).then(
        fn=refresh_memories, outputs=[memories_display]
    )

# - - - - - - - - 应用程序启动 - - - - 应用程序启动
# 挂载 Gradio 应用到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/memory")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
