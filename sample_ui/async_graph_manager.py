"""用于UIs的常见 AsyncGraphManager ."""

from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from insightflow import config
from insightflow.agent_graph import build_agent_graph_async
from insightflow.tool.memory import cleanup_async_memory_store, get_async_memory_store, setup_async_memory_store
from insightflow.utils import log


class AsyncGraphManager:
    """管理同步图和检查器寿命周期"""

    def __init__(self):
        self.checkpointer = None
        self.graph = None  # 默认图形( 背向兼容)
        self.graphs: dict[str, Any] = {}
        self._context_manager = None
        self._memory_store = None
        self._initialized = False

    async def initialize(self):
        """初始化图表和检查符"""
        if self._initialized:
            return

        try:
            # 设置 Aync 内存库
            await setup_async_memory_store()

            # 初始化检查器
            self._context_manager = AsyncSqliteSaver.from_conn_string("checkpoints.db")
            self.checkpointer = await self._context_manager.__aenter__()

            # 用于图形构建的缓存存储
            self._memory_store = await get_async_memory_store()

            self._initialized = True

            # 为反向兼容性构建默认图表
            self.graph = await self.get_graph()

            log("Graph initialized successfully")

        except Exception as e:
            self._initialized = False
            log(f"Failed to initialize graph: {e}")
            raise

    async def get_graph(self, llm_provider: str | None = None):
        """为请求的 LLM 提供者获取或构建一个图表。"""
        if not self._initialized:
            await self.initialize()

        key = llm_provider or "__default__"
        if key in self.graphs:
            return self.graphs[key]

        graph = await build_agent_graph_async(
            config.get().catalog_store,
            checkpointer=self.checkpointer,
            memory_store=self._memory_store,
            memory_tools=None,  # 让图构建器创建适合提供者的工具
            llm_provider=llm_provider,
        )
        self.graphs[key] = graph
        return graph

    async def cleanup(self):
        """清理资源"""
        if self.checkpointer is not None and self._context_manager is not None:
            try:
                await self._context_manager.__aexit__(None, None, None)
                await cleanup_async_memory_store()
                log("Graph cleaned up successfully")
            except Exception as e:
                log(f"Error during cleanup: {e}")
            finally:
                self.checkpointer = None
                self.graph = None
                self.graphs = {}
                self._context_manager = None
                self._memory_store = None
                self._initialized = False
