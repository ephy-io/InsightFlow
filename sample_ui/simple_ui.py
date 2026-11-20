"""使用 FastAPI 和 Gradio 为 InsightFlow 的简单网络 UI ."""

from collections import defaultdict

import gradio as gr
import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from insightflow import config
from insightflow.agent_graph import build_agent_graph_sync
from insightflow.tool.memory import get_sync_memory_store
from insightflow.utils import get_report_download_response, log
from sample_ui.style import custom_css

# 会话状态存储: session_id ->state
session_interrupt = defaultdict(bool)

# 持久性使用 SqliteSaver
sqlite_checkpointer_cm = SqliteSaver.from_conn_string("checkpoints.db")
sqlite_checkpointer = sqlite_checkpointer_cm.__enter__()
graph = build_agent_graph_sync(
    config.get().catalog_store, checkpointer=sqlite_checkpointer, memory_store=get_sync_memory_store()
)

# ---------- FastAPI 接口 ----------
app = FastAPI()


# ---------- Gradio 界面 ----------
def chat_fn(message: str, history: list[tuple[str, str]], user_id: str = "default", session_id: str = "default") -> str:
    """Gradio 接口的聊天功能。"""
    user_session_id = f"{user_id}-{session_id}"
    config = {"configurable": {"thread_id": user_session_id, "user_id": user_id}}

    if session_interrupt[user_session_id]:
        inputs = Command(resume=message)
    else:
        inputs = {"messages": [{"role": "user", "content": message}]}

    # 使用同步调用
    result = graph.invoke(inputs, config=config)
    state = graph.get_state(config)
    if state.interrupts:
        log(f"state.interrupts: {state.interrupts}")
        output_content = state.interrupts[0].value.get("text")
        session_interrupt[user_session_id] = True
    else:
        session_interrupt[user_session_id] = False
        output_content = result["messages"][-1].content

    return output_content


# 创建自定义 CSS 和主题的 Gradio 接口
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 💬 InsightFlow Agent Chatbot")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chatbot", label="", bubble_full_width=False, height=600)
            msg = gr.Textbox(placeholder="Type a message and press Enter", label="Input", elem_id="msg")
        with gr.Column(scale=1):
            user_box = gr.Textbox(value="default", label="User ID", interactive=True)
            session_box = gr.Textbox(value="default", label="Session ID", interactive=True)
            gr.Markdown(
                """
            **Instructions**
            - Type a message and press Enter to send
            - User ID is used for memory isolation between users
            - Session ID can be used to differentiate between conversations
            """,
                elem_id="description",
            )

    def respond(
        message: str, chat_history: list[tuple[str, str]], user_id: str, session_id: str
    ) -> tuple[str, list[tuple[str, str]]]:
        """在 Gradio 聊天界面中处理响应。"""
        response = chat_fn(message, chat_history, user_id, session_id)
        chat_history.append((message, response))
        return "", chat_history

    msg.submit(respond, [msg, chatbot, user_box, session_box], [msg, chatbot])


# ---------- API 接口 ----------
@app.get("/api/download/report/{filename}")
def download_report(filename: str):
    """下载保存的报告文件。"""
    return get_report_download_response(filename)


# - - - - - - - - 应用程序启动 - - - - 应用程序启动
# 挂载 Gradio 应用到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        # 清理检查器
        sqlite_checkpointer_cm.__exit__(None, None, None)
