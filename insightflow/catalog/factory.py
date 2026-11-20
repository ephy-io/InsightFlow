import logging
import os
from typing import Any

from insightflow.catalog.catalog_loader import load_catalog_from_data_warehouse
from insightflow.catalog.catalog_store import CatalogStore
from insightflow.catalog.store.database import DatabaseCatalogStore
from insightflow.catalog.store.file_system import FileSystemCatalogStore

logger = logging.getLogger(__name__)


# 创建 CatalogStore 实例的工厂函数
def create_catalog_store(
    store_type: str, auto_load: bool = True, data_warehouse_config: dict[str, Any] | None = None, **kwargs
) -> CatalogStore:
    """创建 CatalogStore 实例

    Args:
        store_type (str): 存储类型，支持 'file_system' 和 'database'
        auto_load (bool): 如果目录数据不存在，是否从数据库自动加载
        data_warehouse_config (dict): 数据仓库配置词典。这是
            仅用于构建数据仓库执行引擎，从未再使用
            作为目录持续数据库连接。
        **kwargs:其他参数。
            为 'file_system' : data_path .
            对于 'database' : connection_string ,引擎，auto_create_schema ,回声。

    Returns:
        CatalogStore : CatalogStore 实例

    Raises:
        ValueError :不支持存储类型
    """
    if store_type == "file_system":
        data_path = kwargs.get("data_path", "data")
        # 将相对路径转换为绝对路径
        if not data_path.startswith("/"):
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), data_path)
        catalog_store: CatalogStore = FileSystemCatalogStore(data_path, data_warehouse_config or {})

        # 检查是否启用自动加载，是否缺少目录文件
        if auto_load:
            _auto_load_catalog_if_needed(catalog_store)

        return catalog_store
    elif store_type == "database":
        catalog_store = DatabaseCatalogStore(
            connection_string=kwargs.get("connection_string"),
            engine=kwargs.get("engine"),
            data_warehouse_config=data_warehouse_config or {},
            auto_create_schema=kwargs.get("auto_create_schema", True),
            echo=kwargs.get("echo", False),
        )

        # 当目录 DB 为空时，可选择从数据仓库中加载目录。
        if auto_load:
            _auto_load_catalog_if_needed(catalog_store)

        return catalog_store
    else:
        raise ValueError(f"Unsupported storage type: {store_type}")


def _auto_load_catalog_if_needed(catalog_store: CatalogStore) -> None:
    """如果目录文件缺失或为空，从数据仓库自动装入目录

    Args:
        catalog_store (CatalogStore): 目录存储实例
    """

    # 检查目录存储是否有使用该商店自己的 check_exists 方法的现有数据
    if not catalog_store.check_exists():
        logger.info("Catalog files missing or empty, attempting to load from data warehouse...")

        try:
            # 从装入的配置中获取数据仓库配置
            data_warehouse_config = catalog_store.get_data_warehouse_config()
            if not data_warehouse_config:
                logger.warning("No data warehouse configuration found, skipping autoload")
                return

            warehouse_uri = data_warehouse_config.get("uri")
            if not warehouse_uri:
                logger.warning("No data warehouse URI found in configuration, skipping autoload")
                return

            # 从数据仓库装入目录
            success = load_catalog_from_data_warehouse(catalog_store)

            if success:
                logger.info("Successfully loaded catalog from data warehouse")
            else:
                logger.error("Failed to load catalog from data warehouse")
                raise Exception("Failed to load catalog from data warehouse")

        except Exception as e:
            logger.warning(f"Autoload from data warehouse failed: {e}")
            raise Exception("Failed to load catalog from data warehouse") from e
