"""Streamlit -基于 UI 对 InsightFlow 的串流，并带有可折叠的思维部分。"""

import asyncio
import atexit
import sys
import uuid
from pathlib import Path

import plotly.graph_objects as go

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover
    import sqlite3
import streamlit as st

sys.modules["sqlite3"] = sqlite3

from langgraph.types import Command  # noqa: E402

from insightflow import config as insightflow_config  # noqa: E402
from insightflow.llm.llm import list_llm_providers  # noqa: E402
from insightflow.streaming import AgentStreamProcessor, StreamStep, StreamToken  # noqa: E402
from insightflow.utils import log  # noqa: E402
from sample_ui.async_graph_manager import AsyncGraphManager  # noqa: E402
from sample_ui.history_loader import load_session_history  # noqa: E402
from sample_ui.plotly_utils import visualization_dsl_to_gradio_plot  # noqa: E402

# 配置
st.set_page_config(page_title="InsightFlow 企业经营分析", page_icon="💬", layout="wide")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = AsyncGraphManager()
if "session_interrupts" not in st.session_state:
    st.session_state.session_interrupts = {}
if "event_loop" not in st.session_state:
    st.session_state.event_loop = None
if "loaded_thread" not in st.session_state:
    st.session_state.loaded_thread = None


async def process_user_message_stream(
    message: str, user_id: str, session_id: str, llm_provider: str | None, thinking_container, response_container
):
    """通过 InsightFlow 图表处理用户消息并实时更新
    处理时更新 thinking_container 和 response_container
    """
    thinking_steps = []
    top_level_step_number = 0
    # 按流深键的分级计数器 :
    # 0级QQ顶级阶梯，1级+QQ巢次阶梯。
    step_counters: dict[int, int] = {0: 0}
    final_response = ""
    plot_figure = None

    # 需要时初始化图表
    if not st.session_state.graph_manager._initialized:
        await st.session_state.graph_manager.initialize()
    graph = await st.session_state.graph_manager.get_graph(llm_provider)

    user_session_id = f"{user_id}-{session_id}"

    # 检查中断
    if st.session_state.session_interrupts.get(user_session_id, False):
        stream_input = Command(resume=message)
    else:
        stream_input = {"messages": [{"role": "user", "content": message}]}

    from insightflow.observability.tracing import build_run_config

    config = build_run_config(user_id=user_id, session_id=session_id)

    # 实时更新时使用空容器
    thinking_placeholder = thinking_container.empty()

    # 按时间顺序构建内容 -所有事件按时间顺序排列
    base_content = "🔄 **Processing...**\n\n"
    chronological_content = ""  # 所有内容按时间顺序排列
    # 正在附加的图层符号的音轨，所以连续
    # 同一来源的符号归在一个标题下. "main" =主
    # 代理，一个命名空间拖曳=一个子代理，None =重置(在一步行之后).
    current_token_layer = None

    def update_display():
        full_content = base_content + chronological_content
        thinking_placeholder.markdown(full_content)

    # 初始显示
    update_display()

    # 使用共享事件解析器在图表中流过，所以 CLI
    # HTTP API 和这个 UI 都表面是相同的中间步骤。
    processor = AgentStreamProcessor()
    async for namespace, event_type, event_value in graph.astream(
        stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True, debug=True
    ):
        for event in processor.process(namespace, event_type, event_value):
            if isinstance(event, StreamToken):
                if event.is_final:
                    # 主要代理源流(中间/最后)答案。
                    final_response += event.text
                    if current_token_layer != "main":
                        chronological_content += "\n\n**🤖 AI Response:** "
                        current_token_layer = "main"
                else:
                    # 子代理(数据分析)思考符。
                    token_layer = ("think", event.level, event.label)
                    if current_token_layer != token_layer:
                        chronological_content += f"\n\n{'　' * event.level}💭 *{event.label} Thinking:* "
                        current_token_layer = token_layer
                chronological_content += event.text
                update_display()

            elif isinstance(event, StreamStep):
                desc = event.text

                # 构建 Plotly 数字是 UI - 特异性，所以它停留在这里。
                # 共享解析器将 DSL 和最新的 CSV 数据交给我们。
                if event.kind == "visualization":
                    data_csv = event.data.get("data")
                    visualization_dsl = event.data.get("visualization_dsl")
                    if not data_csv:
                        # 匹配先前的行为:没有数据就没有图表。
                        continue
                    try:
                        plot_figure, plot_description = visualization_dsl_to_gradio_plot(data_csv, visualization_dsl)
                        desc = f"📊 Generated visualization: {plot_description}"
                    except Exception as e:
                        desc = f"⚠️ Visualization error: {str(e)}"

                thinking_steps.append(desc)
                if chronological_content and not chronological_content.endswith("\n\n"):
                    chronological_content += "\n\n"
                if event.level == 0:
                    top_level_step_number += 1
                    step_counters[0] = top_level_step_number
                    # 进入新的顶级台阶时重置嵌套计数器。
                    step_counters = {k: v for k, v in step_counters.items() if k == 0}
                    chronological_content += f"**Step {top_level_step_number}:** {desc}\n\n"
                else:
                    # 确保我们总是有顶级背景，即使
                    # 巢步由于事件命令而先出现。
                    if step_counters.get(0, 0) == 0:
                        top_level_step_number = 1
                        step_counters[0] = top_level_step_number

                    step_counters[event.level] = step_counters.get(event.level, 0) + 1
                    # 当我们向上移动时降低更深的关卡。
                    step_counters = {k: v for k, v in step_counters.items() if k <= event.level}

                    step_number_parts = [
                        str(step_counters[level]) for level in sorted(step_counters.keys()) if level <= event.level
                    ]
                    hierarchical_step_number = ".".join(step_number_parts)
                    chronological_content += (
                        f"{'　' * event.level}↳ Step {hierarchical_step_number} " f"[{event.label}] {desc}\n\n"
                    )
                # 重置令牌分组后，下一流令牌会得到一个头条。
                current_token_layer = None
                update_display()

    # 检查是否在最后状态中中断
    state = await graph.aget_state(config)
    if state.interrupts:
        log(f"State interrupts: {state.interrupts}")
        output_content = state.interrupts[0].value.get("text", "")
        if "buttons" in state.interrupts[0].value:
            output_content += str(state.interrupts[0].value.get("buttons"))
        final_response += output_content

        # 将中断内容添加到时间顺序内容
        chronological_content += output_content
        update_display()

        st.session_state.session_interrupts[user_session_id] = True
    else:
        st.session_state.session_interrupts[user_session_id] = False

    # 最后更新 - 将补全消息添加到时间顺序内容
    # 如果最后的内容没有以新行结尾，则添加一些间隔
    if not chronological_content.endswith("\n\n"):
        chronological_content += "\n\n"
    chronological_content += "✅ **Analysis complete!**"
    update_display()

    # 提取最终答案(最后一部分没有工具调用) 并显示外部思维
    if final_response:
        # 查找工具使用中最后一次出现的情况以区分最终答案
        lines = final_response.split("\n")
        final_answer_lines = []
        collecting_final = False

        for line in reversed(lines):
            if "Use tool:" in line or "Using tools:" in line or "Using tool:" in line:
                break
            final_answer_lines.append(line)
            collecting_final = True

        if collecting_final and final_answer_lines:
            # 反转到正确的顺序
            final_answer_lines.reverse()
            final_answer_text = "\n".join(final_answer_lines).strip()

            if final_answer_text:
                with response_container:
                    processed_final_answer_text = process_download_links(final_answer_text)
                    render_content_with_downloads(processed_final_answer_text)

    # 对响应容器的最后更新 -如果有，只显示模式(文本响应在思维容器中)
    with response_container:
        if plot_figure:
            st.plotly_chart(plot_figure, width="stretch", key=str(uuid.uuid4()))

    # 提取单独存储的最后答案
    final_answer_text = ""
    if final_response:
        lines = final_response.split("\n")
        final_answer_lines = []
        collecting_final = False

        for line in reversed(lines):
            if "Use tool:" in line or "Using tools:" in line or "Using tool:" in line:
                break
            final_answer_lines.append(line)
            collecting_final = True

        if collecting_final and final_answer_lines:
            final_answer_lines.reverse()
            final_answer_text = "\n".join(final_answer_lines).strip()

    return final_response, plot_figure, thinking_steps, chronological_content, final_answer_text


async def _load_history_async(user_id: str, session_id: str, llm_provider: str | None) -> list[dict]:
    """如果需要，初始化图表，并加载一个线程的持续历史。"""
    if not st.session_state.graph_manager._initialized:
        await st.session_state.graph_manager.initialize()
    graph = await st.session_state.graph_manager.get_graph(llm_provider)
    return await load_session_history(graph, user_id, session_id)


def restore_history_if_needed(user_id: str, session_id: str, llm_provider: str | None) -> None:
    """当活动线程发生变化时，将检查点历史加载到 UI 中。

    每个线程运行一次: 切换 ``user_id`` / ``session_id`` 重装历史，
    在同一线条内重运行时是无操作的，所以从来就没有直播信息
    已复制。
    """
    thread = f"{user_id}-{session_id}"
    if st.session_state.loaded_thread == thread:
        return

    try:
        if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
            st.session_state.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(st.session_state.event_loop)
        loop = st.session_state.event_loop
        st.session_state.messages = loop.run_until_complete(_load_history_async(user_id, session_id, llm_provider))
    except Exception as e:
        log(f"Error restoring history for {thread}: {e}")
        st.session_state.messages = []
    finally:
        st.session_state.loaded_thread = thread


def get_available_reports() -> list[str]:
    """获取可用的报表文件列表供下载。"""
    try:
        # 在此导入配置以避免循环导入
        from insightflow import config

        report_dir = Path(config.get().report_directory)
        if not report_dir.exists():
            return []

        # 在报表目录中获取全部文件
        report_files = []
        for file_path in report_dir.iterdir():
            if file_path.is_file():
                report_files.append(file_path.name)

        return sorted(report_files)
    except Exception as e:
        log(f"读取报告列表失败：{e!r}")
        st.error("读取报告列表失败，请检查服务日志。")
        return []


def get_report_file_content(filename: str) -> tuple[bytes | None, str | None]:
    """获取报告文件内容以下载。

    Returns:
        tuple: ( file_content_bytes , mime_type) 或 ( None , None)如果错误
    """
    try:
        # 在此导入配置以避免循环导入
        from insightflow import config

        report_dir = Path(config.get().report_directory)
        file_path = report_dir / filename

        # 安全检查 -确保文件在报告目录内
        if not file_path.exists() or not file_path.is_file():
            st.error(f"Report file not found: {filename}")
            return None, None

        try:
            file_path.resolve().relative_to(report_dir.resolve())
        except ValueError:
            st.error("Access denied to file")
            return None, None

        # 确定 MIME 类型
        mime_type_map = {
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".html": "text/html",
            ".xml": "application/xml",
        }

        file_extension = file_path.suffix.lower()
        mime_type = mime_type_map.get(file_extension, "application/octet-stream")

        # 读取文件内容
        with open(file_path, "rb") as f:
            content = f.read()

        return content, mime_type

    except Exception as e:
        log(f"读取报告文件失败：{e!r}")
        st.error("读取报告文件失败，请检查服务日志。")
        return None, None


def process_download_links(content: str) -> str:
    """处理内容中的下载链接，并将其替换为 Streamlit - 兼容的链接。

    Args:
        content: 可能包含下载链接的消息内容

    Returns:
        str: 内容已替换下载链接
    """
    import re

    if not content:
        return content

    # 匹配完整 URL 和只路径下载链接的模式
    # 匹配 http://localhost:8501/api/download/report/filename.ext 或 /api/download/report/filename.ext
    download_pattern = r"(?:https?://[^/\s]+)?/api/download/report/([^)\s\]<>]+)"

    def replace_link(match):
        filename = match.group(1)
        # 返回一个占位符，我们用实际的下载按钮替换它
        return f"[DOWNLOAD_LINK:{filename}]"

    processed_content = re.sub(download_pattern, replace_link, content)

    # 调试日志以查看处理是否有效
    if processed_content != content:
        st.write(f"🔍 Debug: Processed download links - found {content.count('/api/download/report/')} links")

    return processed_content


def render_content_with_downloads(content: str) -> None:
    """渲染内容并用实际下载按钮替换下载占位符。"""
    import re

    # 按下载占位符分割内容
    download_pattern = r"\[DOWNLOAD_LINK:([^)]+)\]"
    parts = re.split(download_pattern, content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # 正常内容
            if part.strip():
                st.markdown(part)
        else:
            # 下载链接文件名
            filename = part
            file_content, mime_type = get_report_file_content(filename)

            if file_content is not None:
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=file_content,
                    file_name=filename,
                    mime=mime_type,
                    key=f"inline_download_{filename}_{hash(content)}",
                )
            else:
                st.error(f"❌ Could not load report: {filename}")


def display_message_with_thinking(
    role: str, content: str, thinking_steps: list[str] = None, plot_figure: go.Figure = None
):
    """显示带有可折叠思维部分的消息"""
    with st.chat_message(role):
        if thinking_steps and role == "assistant":
            # 创建包含全部内容的思维部分
            with st.expander("💭 AI Thinking Process", expanded=False):
                for i, step in enumerate(thinking_steps, 1):
                    st.markdown(f"**Step {i}:** {step}")

                if content:
                    st.markdown("**🤖 AI Response:**")
                    render_content_with_downloads(content)

                st.success("✅ Analysis complete")

        # 对于非辅助消息，通常显示内容
        elif content and role != "assistant":
            render_content_with_downloads(content)

        # 如果可用，显示图( 外部思维容器)
        if plot_figure:
            st.plotly_chart(plot_figure, width="stretch", key=str(uuid.uuid4()))


# 主语 UI
st.title("💬 InsightFlow 企业经营分析")
st.markdown("*用自然语言查询和分析经营数据*")

# 配置的侧边栏
with st.sidebar:
    st.header("⚙️ 会话设置")
    user_id = st.text_input("用户标识", value="default", help="用于隔离用户会话和长期记忆")
    session_id = st.text_input("会话标识", value="default", help="用于恢复本次对话上下文")

    # 可选多供应商支持
    llm_provider = None
    provider_options = list_llm_providers()
    if provider_options:
        try:
            default_provider = getattr(insightflow_config.get(), "llm_provider", None)
        except Exception:
            default_provider = None
        default_index = provider_options.index(default_provider) if default_provider in provider_options else 0
        llm_provider = st.selectbox(
            "模型提供方",
            options=provider_options,
            index=default_index,
            help="选择本次会话使用的已配置模型",
        )

    st.markdown("---")
    st.markdown("""
    **💡 使用说明：**
    - 输入经营分析问题
    - 在可折叠区域查看处理步骤
    - 查看查询结果、分析结论和图表
    - 使用不同会话标识隔离对话
    """)

    if st.button("🗑️ 清空会话记录"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 报告下载")

    # 获取可用报告
    available_reports = get_available_reports()

    if available_reports:
        selected_report = st.selectbox(
            "选择要下载的报告：", options=[""] + available_reports, help="选择已保存的报告文件"
        )

        if selected_report and st.button("📥 准备下载"):
            file_content, mime_type = get_report_file_content(selected_report)
            if file_content is not None:
                st.download_button(
                    label=f"💾 保存 {selected_report}",
                    data=file_content,
                    file_name=selected_report,
                    mime=mime_type,
                    key=f"download_{selected_report}",
                )
                st.success(f"✅ {selected_report} 已准备好下载。")
    else:
        st.info("当前没有可下载的报告。")

# 活动线程改变时从检查器恢复持续历史
# (如页面刷新或切换用户/会议 ID). 在同一线条内不开放。
restore_history_if_needed(user_id, session_id, llm_provider)

# 显示聊天历史
for msg in st.session_state.messages:
    if msg["type"] == "chronological_message":
        # 在扩展器中显示时间顺序内容 - 完成后全部崩溃
        with st.chat_message(msg["role"]):
            with st.expander("💭 AI Thinking Process", expanded=False):
                st.markdown(msg["chronological_content"])

            # 提取并显示外部思考的最后答案文本
            if msg.get("final_answer"):
                render_content_with_downloads(msg["final_answer"])

            # 如果可用，显示图( 外部思维容器)
            if msg.get("plot_figure"):
                st.plotly_chart(msg["plot_figure"], width="stretch", key=str(uuid.uuid4()))

    elif msg["type"] == "thinking_message":
        display_message_with_thinking(
            msg["role"], msg["content"], msg.get("thinking_steps", []), msg.get("plot_figure")
        )
    else:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                render_content_with_downloads(msg["content"])
            elif msg["type"] == "plot" and msg.get("plot_figure"):
                st.plotly_chart(msg["plot_figure"], width="stretch", key=str(uuid.uuid4()))

# 聊天输入
if prompt := st.chat_input("请输入经营数据问题……"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 流程助理实时响应
    with st.chat_message("assistant"):
        # 创建思维和反应容器
        thinking_expander = st.expander("💭 AI Thinking Process...", expanded=True)
        thinking_container = thinking_expander.container()
        response_container = st.container()

        # 与实时更新同步处理消息
        try:
            # 重用同一事件循环以避免约束性问题
            if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
                st.session_state.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(st.session_state.event_loop)

            loop = st.session_state.event_loop
            final_response, plot_figure, thinking_steps, full_chronological_content, final_answer = (
                loop.run_until_complete(
                    process_user_message_stream(
                        prompt, user_id, session_id, llm_provider, thinking_container, response_container
                    )
                )
            )

            # 不需要创建另一个扩展器 - 内容已实时显示
            # 在存储前处理内容中的下载链接
            processed_chronological_content = process_download_links(full_chronological_content)
            processed_final_answer = process_download_links(final_answer) if final_answer else final_answer

            # 将完整的消息与已处理的内容一起存储
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "type": "chronological_message",
                    "chronological_content": processed_chronological_content,
                    "final_answer": processed_final_answer,
                    "plot_figure": plot_figure,
                }
            )

            # 触发器重运行以崩溃思维部分
            st.rerun()

        except Exception as e:
            log(f"处理用户请求失败：{e!r}")
            st.error("❌ 处理请求时发生错误，请检查配置后重试。")
            error_content = "❌ 处理请求失败，请检查配置后重试。"
            processed_error_content = process_download_links(error_content)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": processed_error_content})


# 在会话结束时清理
def cleanup_session():
    """当会话结束时清理资源"""
    if "graph_manager" in st.session_state:
        try:
            # 使用同一事件循环进行清理
            if st.session_state.event_loop and not st.session_state.event_loop.is_closed():
                loop = st.session_state.event_loop
                loop.run_until_complete(st.session_state.graph_manager.cleanup())
                loop.close()
                st.session_state.event_loop = None
        except Exception as e:
            log(f"Error during session cleanup: {e}")


# 登记清理(这是一种简化的方法-在生产过程中，你可能需要更严格的清理)

atexit.register(cleanup_session)
