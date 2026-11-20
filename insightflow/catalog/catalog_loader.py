import logging
from typing import Any, cast

from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine

from .catalog_store import CatalogStore

logger = logging.getLogger(__name__)


class DataCatalogLoader:
    """从数据仓库元数据中装入数据目录并保存到目录存储的加载器。"""

    def __init__(self, engine: Engine, include_tables: list[str] | None = None):
        """初始化目录加载器。

        Args:
            engine (Engine): SQLAlchemy 引擎实例
            include_tables (Optional[List[str]]): 要包含的表格名列表，None 适用于所有人
        """
        self.engine = engine
        self.include_tables = include_tables
        self.metadata = MetaData()
        self.inspector = inspect(engine)

    def get_tables_and_columns(self) -> dict[str, list[dict[str, Any]]]:
        """使用 SQLAlchemy 检查器提取表格和列元数据，包括注释。

        Returns:
            Dict[str, List[Dict[str, Any]]]: 用于列信息列表的词典映射表名
        """
        try:
            tables_columns = {}

            # 获取全部表名
            table_names = self.inspector.get_table_names()

            # 配置后过滤到特定表格
            if self.include_tables:
                table_names = [name for name in table_names if name in self.include_tables]

            logger.info(f"Found {len(table_names)} tables to process")

            for table_name in table_names:
                try:
                    # 获取表字段信息
                    columns = self.inspector.get_columns(table_name)
                    column_list = []
                    for column in columns:
                        is_common_column = column["name"] not in ("id", "name", "type", "status")
                        column_info = {
                            "column_name": column["name"],
                            "display_name": "",
                            "alias": "",
                            "type": str(column["type"]),
                            "category": "",
                            "tag": "",
                            "description": column.get("comment", "") or "",
                            "dimension_table": "",
                            "default": str(column.get("default", "")) if column.get("default") is not None else "",
                            "is_common": is_common_column,
                        }
                        column_list.append(column_info)

                    tables_columns[table_name] = column_list
                    logger.debug(f"Processed table {table_name} with {len(column_list)} columns")

                except Exception as e:
                    logger.error(f"Failed to process table {table_name}: {e}")
                    continue

            logger.info(f"Successfully processed {len(tables_columns)} tables")
            return tables_columns

        except Exception as e:
            logger.error(f"Failed to get tables and columns from data warehouse: {e}")
            return {}

    def get_table_indexes(self, table_name: str) -> list[dict[str, Any]]:
        """获取特定表格的索引信息。

        Args:
            table_name (str): 表名

        Returns:
            List[Dict[str, Any]]: 索引资料清单
        """
        try:
            indexes = self.inspector.get_indexes(table_name)
            return cast(list[dict[str, Any]], indexes)
        except Exception as e:
            logger.warning(f"Failed to get indexes for table {table_name}: {e}")
            return []

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """获取特定表格的外国密钥信息。

        Args:
            table_name (str): 表名

        Returns:
            List[Dict[str, Any]]: 外国关键信息清单
        """
        try:
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            return cast(list[dict[str, Any]], foreign_keys)
        except Exception as e:
            logger.warning(f"Failed to get foreign keys for table {table_name}: {e}")
            return []

    def save_to_catalog_store(
        self, catalog_store: CatalogStore, database_name: str | None = None, update: bool = False
    ) -> bool:
        """提取仓库元数据并保存到目录存储。

        Args:
            catalog_store (CatalogStore): 目标目录存储以装入数据到
            database_name (Optional[str]): 目录中的数据库名称，默认为 'default'
            update (bool): 更新现有目录存储以同步数据仓库

        Returns:
            bool: 如果负载成功，则使用 True ,否则使用 False
        """
        try:
            if database_name is None:
                database_name = "default"

            # 从数据仓库获取表格和列
            tables_columns = self.get_tables_and_columns()

            if not tables_columns:
                logger.warning("No tables found in data warehouse")
                return True

            # 导入每个表格
            success_count = 0
            total_count = len(tables_columns)

            for table_name, columns in tables_columns.items():
                try:
                    # 如果可用，获取表格注释
                    table_comment = ""
                    try:
                        raw_comment = self.inspector.get_table_comment(table_name)
                        table_comment = (raw_comment.get("text") or "") if raw_comment else ""
                    except Exception as e:
                        # 有些数据库不支持表格注释
                        logger.info("Skipping table comment for %s: %s", table_name, e)

                    table_info: dict[str, Any] = {"description": table_comment, "selection_rule": "", "sql_rule": ""}
                    if catalog_store.save_table_information(table_name, table_info, columns, database_name):
                        success_count += 1
                        logger.info(f"Successfully loaded table: {database_name}.{table_name}")
                    else:
                        logger.error(f"Failed to load table: {database_name}.{table_name}")

                    # init 无效 SQL 示例
                    catalog_store.save_table_sql_examples(
                        table_name, [{"question": "null", "answer": "null"}], database_name
                    )

                except Exception as e:
                    logger.error(f"Error loading table {table_name}: {e}")

            # 输入空表格选择示例
            catalog_store.save_table_selection_examples([("", [])])

            logger.info(f"Load completed: {success_count}/{total_count} tables loaded successfully")
            return success_count == total_count

        except Exception as e:
            logger.error(f"Failed to load data warehouse to catalog store: {e}")
            return False


def load_catalog_from_data_warehouse(catalog_store: CatalogStore) -> bool:
    """根据数据仓库配置( URI)使用SQLAlchemy从数据仓库装入目录数据。

    目录加载的主要切入点。

    Args:
        catalog_store (CatalogStore): 目标目录库

    Returns:
        bool: 如果负载成功，则使用 True ,否则使用 False
    """
    try:
        data_warehouse_config = catalog_store.get_data_warehouse_config()
        database_uri = data_warehouse_config.get("uri")
        include_tables = data_warehouse_config.get("include_tables")
        database_name = data_warehouse_config.get("database_name", "default")
        engine = catalog_store.get_sql_engine()

        loader = DataCatalogLoader(engine, include_tables)
        return loader.save_to_catalog_store(catalog_store, database_name)

    except Exception as e:
        logger.error(f"Failed to import catalog from data warehouse URI {database_uri}: {e}")
        return False
