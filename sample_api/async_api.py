"""Async API 用于从 InsightFlow 流传的聊天响应。"""

import asyncio
import dataclasses
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from insightflow import config
from insightflow.agent_graph import build_agent_graph_async
from insightflow.streaming import (
    AgentStreamProcessor,
    StreamInterrupt,
    StreamStep,
    StreamToken,
    StreamUsage,
    extract_final_answer,
)
from insightflow.utils import get_report_download_response

logger = logging.getLogger(__name__)

# 会话状态存储: session_id ->state
sessions = defaultdict(dict)

# 以提供者名称键入的图表
graphs: dict[str, Any] = {}
graphs_lock = asyncio.Lock()


async def get_or_build_graph(provider: str | None):
    """获取指定模型提供方的图，首次调用时构建。"""
    key = provider or "__default__"
    if key in graphs:
        return graphs[key]
    async with graphs_lock:
        if key in graphs:
            return graphs[key]
        graphs[key] = await build_agent_graph_async(config.get().catalog_store, llm_provider=provider)
        return graphs[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序寿命事件。"""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    # 启动: 初始化 async图
    graphs["__default__"] = await build_agent_graph_async(config.get().catalog_store)
    yield
    # 关闭: 必要时进行清理
    graphs.clear()


app = FastAPI(lifespan=lifespan)


class UserRequest(BaseModel):
    """请求流式聊天模式。"""

    input: str
    user_id: str | None = "default"
    session_id: str | None = "default"
    provider: str | None = None
    # "events" → 结构化的 NDJSON (步骤 + 符号 + 最终答案，如)
    # Streamlit UI). "text" – 遗留的纯文本只答流。
    mode: str | None = "events"


def _event_to_dict(event) -> dict[str, Any]:
    """串行串行事件为 JSON - 友好的dict。"""
    if isinstance(event, StreamStep):
        return {
            "type": "step",
            "kind": event.kind,
            "level": event.level,
            "label": event.label,
            "text": event.text,
            "data": _json_safe(event.data),
        }
    if isinstance(event, StreamToken):
        return {
            "type": "token",
            "level": event.level,
            "label": event.label,
            "is_final": event.is_final,
            "text": event.text,
        }
    if isinstance(event, StreamInterrupt):
        return {"type": "interrupt", "text": event.text, "buttons": _json_safe(event.buttons)}
    if isinstance(event, StreamUsage):
        return {
            "type": "usage",
            "turn_tokens": event.turn_tokens,
            "turn_cost_usd": event.turn_cost_usd,
            "by_model": event.by_model,
        }
    return {"type": "unknown"}


def _json_safe(obj: Any) -> Any:
    """任意请求数据最佳转换为 JSON -可串行数据。"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return str(obj)


@app.post("/chat/stream")
async def chat_stream(req: UserRequest):
    """从代理图表中流出聊天响应。

    当 ``mode == "events"`` (默认) 时，响应是新线限定 JSON
    ( ``application/x-ndjson``):每条线上有一个带有 ``type`` 字段的物体。
    ``step`` → ``token`` → ``interrupt`` → ``final_answer``。此镜像。
    中间步骤 Streamlit UI 显示。当 ``mode == "text"`` 显示时
    返回纯文本答案流。
    """
    user_id = req.user_id or "default"
    session_id = req.session_id or "default"
    provider = req.provider

    # 创建用户区段 ID 和 UI 一样

    stream_input = {"messages": [("user", req.input)]}
    from insightflow.observability.tracing import build_run_config

    config = build_run_config(user_id=user_id, session_id=session_id)

    try:
        graph = await get_or_build_graph(provider)
    except ValueError as e:
        logger.warning("模型提供方配置无效：%s", e)
        raise HTTPException(status_code=400, detail="模型服务配置无效，请检查 InsightFlow 配置。") from e

    async def text_generator():
        """遗留的纯文本生成器: 仅生成助手的回答文本。"""
        processor = AgentStreamProcessor()
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                if isinstance(event, StreamToken) and event.is_final and event.text:
                    yield event.text

    async def event_generator():
        """结构化的 NDJSON 生成器:步骤，符号，中断，最终答案。"""
        processor = AgentStreamProcessor()
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                yield json.dumps(_event_to_dict(event), ensure_ascii=False) + "\n"

        # 每回送信/费用上市。
        usage = processor.emit_turn_usage()
        if usage is not None:
            yield json.dumps(_event_to_dict(usage), ensure_ascii=False) + "\n"

        # Emit中断或最终从终端状态回答。
        state = await graph.aget_state(config)
        if state.interrupts:
            value = state.interrupts[0].value or {}
            interrupt = StreamInterrupt(text=value.get("text", ""), buttons=value.get("buttons", []) or [])
            yield json.dumps(_event_to_dict(interrupt), ensure_ascii=False) + "\n"
        else:
            final = extract_final_answer(processor.final_response)
            yield json.dumps({"type": "final_answer", "text": final}, ensure_ascii=False) + "\n"

    if (req.mode or "events") == "text":
        return StreamingResponse(text_generator(), media_type="text/plain")
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.get("/user/{user_id}/memories")
async def get_user_memories(user_id: str):
    """为特定用户获取全部记忆。"""
    try:
        # 导入内存访问所需的模块
        import json

        from insightflow.tool.memory import get_async_memory_store

        # 获取 Aync 记忆存储器
        memory_store = await get_async_memory_store()

        memories = []
        namespace = ("memories", user_id)

        try:
            # 为此用户搜索全部记忆
            search_results = await memory_store.asearch(namespace)

            for item in search_results:
                # 解析内存数据
                try:
                    if isinstance(item.value, bytes):
                        content = json.loads(item.value.decode("utf-8"))
                    elif isinstance(item.value, str):
                        content = json.loads(item.value)
                    else:
                        content = item.value
                except (json.JSONDecodeError, AttributeError):
                    content = str(item.value)

                memory_data = {
                    "key": item.key,
                    "content": content,
                    "namespace": str(namespace),
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)

            return {"user_id": user_id, "total_memories": len(memories), "memories": memories}

        except Exception as e:
            logger.exception("读取记忆失败")
            raise HTTPException(status_code=500, detail="读取记忆失败，请稍后重试。") from e

    except Exception as e:
        logger.exception("访问记忆存储失败")
        raise HTTPException(status_code=500, detail="访问记忆存储失败，请稍后重试。") from e


@app.get("/api/download/report/{filename}")
async def download_report(filename: str):
    """下载保存的报告文件。"""
    return get_report_download_response(filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
