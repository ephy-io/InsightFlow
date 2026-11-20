"""Gradio -基于 UI 为 InsightFlow 的流线，具有实时聊天界面。"""

# langraph 的 sqlite 模块在导入前必须查看选中的 sqlite 模块。
# ruff: noqa: E402

import asyncio
import sys
from collections import defaultdict
from contextlib import asynccontextmanager

import gradio as gr

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover
    import sqlite3
from fastapi import FastAPI

sys.modules["sqlite3"] = sqlite3

from langgraph.types import Command

from insightflow.streaming import AgentStreamProcessor, StreamStep, StreamToken
from insightflow.utils import get_report_download_response, log
from sample_ui.async_graph_manager import AsyncGraphManager
from sample_ui.history_loader import load_session_history
from sample_ui.plotly_utils import create_inline_chart_markdown, visualization_dsl_to_gradio_plot
from sample_ui.style import custom_css

# 会话状态存储: user_session_id ->state
session_interrupt = defaultdict(bool)

# async 操作的全局事件环路(类似于 Streamlit 方法)
global_event_loop = None


# 全局图管理器( 类似于 Streamlit 方法)
graph_manager = AsyncGraphManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 寿命的 Async 上下文管理器"""
    # 启动
    await graph_manager.initialize()
    yield
    # 关机
    await graph_manager.cleanup()


# ---------- FastAPI 接口 ----------
app = FastAPI(lifespan=lifespan)


# - - - - - - - - - - Gradio UI 函数 - - - - - - -


def get_or_create_event_loop():
    """获取或创建独立事件循环"""
    global global_event_loop

    if global_event_loop is None or global_event_loop.is_closed():
        global_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(global_event_loop)

    return global_event_loop


async def _async_respond_helper(message, chat_history, user_id, session_id):
    """包含实际合成逻辑的帮助器合成函数。
    这将在一个独立的事件循环中运行。
    每次活动一到来，溪流就逐步更新。
    """
    user_session_id = f"{user_id}-{session_id}"
    full_response = ""
    plot_figure = None
    chart_panel_update = gr.update()
    step_number = 0
    current_token_layer = None

    if session_interrupt[user_session_id]:
        stream_input = Command(resume=message)
    else:
        stream_input = {"messages": [{"role": "user", "content": message}]}

    from insightflow.observability.tracing import build_run_config

    config = build_run_config(user_id=user_id, session_id=session_id)

    # 确保图表可用
    if not graph_manager._initialized:
        try:
            await graph_manager.initialize()
        except Exception as e:
            log(f"Failed to initialize graph: {e}")
            chat_history[-1]["content"] = f"Error: Failed to initialize system - {str(e)}"
            yield "", chat_history, plot_figure, chart_panel_update
            return

    processor = AgentStreamProcessor()

    def build_update():
        chat_history[-1]["content"] = full_response
        return ("", chat_history, plot_figure, chart_panel_update)

    # 通过 LangGraph 流同步分布
    graph = await graph_manager.get_graph()
    async for namespace, event_type, event_value in graph.astream(
        stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True, debug=True
    ):
        for event in processor.process(namespace, event_type, event_value):
            if isinstance(event, StreamToken):
                if event.is_final:
                    if current_token_layer != "main":
                        full_response += "\n\n🤖 AI Response: "
                        current_token_layer = "main"
                else:
                    token_layer = ("think", event.level, event.label)
                    if current_token_layer != token_layer:
                        full_response += f"\n\n{'  ' * event.level}💭 {event.label} Thinking: "
                        current_token_layer = token_layer
                full_response += event.text
                yield build_update()

            elif isinstance(event, StreamStep):
                desc = event.text

                if event.kind == "visualization":
                    data_csv = event.data.get("data")
                    visualization_dsl = event.data.get("visualization_dsl")
                    if not data_csv:
                        continue
                    try:
                        plot_figure, plot_description = visualization_dsl_to_gradio_plot(data_csv, visualization_dsl)
                        chart_markdown = create_inline_chart_markdown(data_csv, visualization_dsl)
                        desc = f"📊 Generated visualization: {plot_description}"
                        full_response += f"\n\n{chart_markdown}"
                        chart_panel_update = gr.update(visible=True)
                    except Exception as e:
                        log(f"Visualization generation error: {str(e)}")
                        desc = f"⚠️ Visualization error: {str(e)}"

                if full_response and not full_response.endswith("\n\n"):
                    full_response += "\n\n"

                if event.level == 0:
                    step_number += 1
                    full_response += f"**Step {step_number}:** {desc}\n"
                else:
                    full_response += f"{'  ' * event.level}↳ [{event.label}] {desc}\n"

                current_token_layer = None
                yield build_update()

    # 获取最终状态并检查可视化数据
    state = await graph.aget_state(config)

    if state.interrupts:
        log(f"state.interrupts: {state.interrupts}")
        output_content = state.interrupts[0].value.get("text", "")
        if "buttons" in state.interrupts[0].value:
            output_content += str(state.interrupts[0].value.get("buttons"))
        full_response += output_content
        session_interrupt[user_session_id] = True
        yield build_update()
    else:
        session_interrupt[user_session_id] = False


def respond(message, chat_history, user_id, session_id="default"):
    """同步调用 Gradio Chatbot ,并进行流派更新。

    此函数处理来自 LangGraph 代理的用户输入和流响应。
    返回: message_input , chat_history , plot_figure , chart_panel_visibility .
    """
    # 在聊天历史中添加占位符
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    plot_figure = None
    chart_panel_update = gr.update()
    yield "", chat_history, plot_figure, chart_panel_update  # 流到 UI 的更新

    # 获取或创建独立事件循环
    loop = get_or_create_event_loop()

    # 在独立事件循环中运行 ASync 帮助器
    async_stream = _async_respond_helper(message, chat_history, user_id, session_id)
    try:
        while True:
            try:
                next_update = loop.run_until_complete(async_stream.__anext__())
            except StopAsyncIteration:
                break
            yield next_update
    except Exception as e:
        log(f"Error in respond: {e}")
        import traceback

        traceback.print_exc()
        chat_history[-1]["content"] = f"Error: {str(e)}"
        yield "", chat_history, plot_figure, chart_panel_update
    finally:
        try:
            loop.run_until_complete(async_stream.aclose())
        except Exception:
            pass


async def _async_load_history_helper(user_id, session_id):
    """如果需要，初始化图表并加载历史为聊天信息。"""
    if not graph_manager._initialized:
        await graph_manager.initialize()
    graph = await graph_manager.get_graph()
    return await load_session_history(graph, user_id, session_id)


def load_history_to_chatbot(user_id, session_id="default"):
    """将活动线索的检查点历史装入聊天台。

    以 ``demo.load`` (页开) 和 用户/会议 ID ``change`` 事件为界，
    因此可见的历史跟随活动的 ``{user_id}-{session_id}`` 线程。
    返回完整的聊天器值，该值将替换(从不附加)它。
    """
    loop = get_or_create_event_loop()
    try:
        return loop.run_until_complete(_async_load_history_helper(user_id, session_id))
    except Exception as e:
        log(f"Error loading history for {user_id}-{session_id}: {e}")
        return []


# - - - - - - - 记忆管理功能 - - - - - - - - -
def list_user_memories(user_id: str) -> str:
    """为特定用户列出所有记忆。"""
    try:
        import json

        try:
            import pysqlite3 as sqlite3
        except ImportError:
            import sqlite3
        from langgraph.store.sqlite import SqliteStore

        from insightflow import config

        # 在此线程中创建新连接以避免 SQLite 线程问题
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        conn.isolation_level = None  # 使用自动承诺模式避免交易冲突
        thread_memory_store = SqliteStore(
            conn, index={"dims": 1536, "embed": config.get().embedding_model, "fields": ["text"]}
        )
        try:
            thread_memory_store.setup()
        except Exception:
            pass  # 可能已经设置了仓库

        memories = []
        namespace = ("memories", user_id)

        try:
            # 使用带有命名空间的搜索查找此用户的所有项目
            items = thread_memory_store.search(namespace, limit=1000)
            for item in items:
                memory_data = {
                    "key": item.key,
                    "value": item.value,
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)
        except Exception as e:
            return f"No memories found for user {user_id} or error: {str(e)}"
        finally:
            conn.close()

        if not memories:
            return f"No memories found for user {user_id}"

        formatted = [f"## Memories for User: {user_id}\n"]
        for i, memory in enumerate(memories, 1):
            formatted.append(f"### Memory {i}")
            formatted.append(f"**Key:** {memory['key']}")

            value = memory["value"]
            if isinstance(value, dict):
                try:
                    value_str = json.dumps(value, indent=2)
                    formatted.append(f"**Content:**\n```json\n{value_str}\n```")
                except ValueError:
                    formatted.append(f"**Content:** {str(value)}")
            else:
                formatted.append(f"**Content:** {str(value)}")

            formatted.append(f"**Created:** {memory['created_at']}")
            formatted.append(f"**Updated:** {memory['updated_at']}")
            formatted.append("---")

        return "\n".join(formatted)

    except Exception as e:
        return f"Error accessing memories: {str(e)}"


# ---------- Gradio 界面组件 ----------

# 创建 Gradio 接口。 应用程序挂载时应用主题/ CSS。
with gr.Blocks() as demo:
    gr.Markdown("## 💬 InsightFlow Agent Chatbot with Streaming & On-Demand Visualization")

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="Chat",
                        layout="bubble",
                        height=500,
                        show_label=False,
                        sanitize_html=False,
                        render_markdown=True,
                    )
                    msg = gr.Textbox(placeholder="Type a message and press Enter", label="Input", elem_id="msg")

                with gr.Column(scale=2, visible=False) as chart_panel:
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### 📊 Interactive Chart")
                        with gr.Column(scale=1):
                            hide_chart_btn = gr.Button("✖️ Hide", elem_id="hide-chart-btn", size="sm")
                    plot = gr.Plot(label="", visible=True, show_label=False)

                with gr.Column(scale=1):
                    user_box = gr.Textbox(value="default", label="User ID", interactive=True)
                    session_box = gr.Textbox(value="default", label="Session ID", interactive=True)
                    show_chart_btn = gr.Button("📊 Show Chart Panel", variant="secondary")
                    gr.Markdown(
                        """
                    **Instructions**
                    - Type a data question and press Enter
                    - Supports streaming output (real-time display)
                    - Click chart links in chat to view interactive charts
                    - Use 'Show Chart Panel' to make panel visible
                    - Session ID can be used to differentiate between conversations
                    """,
                        elem_id="description",
                    )

            def show_chart_panel():
                """显示图表面板。"""
                return gr.update(visible=True)

            def hide_chart_panel():
                """隐藏图表面板。"""
                return gr.update(visible=False)

            # 注册 async 提交处理器，用于带有图输出的消息输入
            msg.submit(
                respond,
                [msg, chatbot, user_box, session_box],
                [msg, chatbot, plot, chart_panel],
                api_visibility="undocumented",
            )
            show_chart_btn.click(show_chart_panel, outputs=[chart_panel], api_visibility="undocumented")
            hide_chart_btn.click(hide_chart_panel, outputs=[chart_panel], api_visibility="undocumented")

            # 从检查器中恢复持续聊天历史:在页面加载中
            # ,当用户/会议 ID 改变时(切换线程)。
            user_box.change(load_history_to_chatbot, [user_box, session_box], [chatbot], api_visibility="undocumented")
            session_box.change(
                load_history_to_chatbot, [user_box, session_box], [chatbot], api_visibility="undocumented"
            )

        with gr.TabItem("🧠 Memory Store"):
            gr.Markdown("### Long-term Memory Viewer")
            gr.Markdown("View memories stored for each user in the system.")

            with gr.Row():
                with gr.Column(scale=3):
                    memory_display = gr.Markdown(
                        value="Enter a User ID and click 'Load Memories' to view stored memories.",
                        elem_id="memory-display",
                    )

                with gr.Column(scale=1):
                    memory_user_input = gr.Textbox(label="User ID", placeholder="default", value="default")
                    load_memories_btn = gr.Button("🔍 Load Memories", variant="primary")

            # 装入记忆的事件处理器
            load_memories_btn.click(
                fn=list_user_memories,
                inputs=[memory_user_input],
                outputs=[memory_display],
                api_visibility="undocumented",
            )

    # 当页面首次加载时将持续的历史恢复到聊天台中。
    demo.load(load_history_to_chatbot, [user_box, session_box], [chatbot], api_visibility="undocumented")


# ---------- API 接口 ----------
@app.get("/api/download/report/{filename}")
async def download_report(filename: str):
    """下载保存的报告文件。"""
    return get_report_download_response(filename)


# - - - - - - - - 应用程序启动 - - - - 应用程序启动
# 挂载 Gradio 应用到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui", css=custom_css, theme=gr.themes.Soft())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
