"""基于文件系统的目录存储执行。"""

import ast
import csv
import json
import logging
import os
import re
import traceback
from typing import Any

import yaml
from sqlalchemy import Engine

from ..catalog_store import CatalogStore, split_db_table_name
from ..helper import create_sqlalchemy_engine_instance

logger = logging.getLogger(__name__)


class FileSystemCatalogStore(CatalogStore):
    """基于文件系统的数据目录存储执行。

    在本地文件系统中存储 CSV 和 YAML 文件中的目录数据。
    """

    data_path: str
    table_info_file: str
    sql_example_file: str
    table_selection_example_file: str
    table_columns_file: str
    common_columns_file: str
    table_spec_columns_file: str

    _table_info_cache: dict | None
    _table_columns_cache: dict | None
    _common_columns_cache: dict | None
    _table_spec_columns_cache: dict | None
    _sql_example_cache: dict | None
    _table_selection_example_cache: list[tuple[str, list[str]]] | None

    _data_warehouse_config: dict
    _sql_engine: Engine | None

    def __init__(self, data_path: str, data_warehouse_config: dict | None = None):
        """初始化文件系统目录存储。

        Args:
            data_path (str): 目录存储目录文件的绝对路径。
            data_warehouse_config (dict): 带有密钥的数据仓库配置词典 :
                - uri (str): 数据库连接 URI
                - include_tables (Optional[List[str]]): 要包含的表格列表，如果 None 包含全部
                - database_name (Optional[str]): 目录中使用的数据库名称
        """
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("data_path must be a non-empty string")

        if data_warehouse_config is None:
            data_warehouse_config = {}
        elif not isinstance(data_warehouse_config, dict):
            raise ValueError("data_warehouse_config must be a dictionary")

        self.data_path = data_path.strip()
        self.table_info_file = os.path.join(data_path, "table_info.yaml")
        self.sql_example_file = os.path.join(data_path, "sql_example.yaml")
        self.table_selection_example_file = os.path.join(data_path, "table_selection_example.csv")
        self.table_columns_file = os.path.join(data_path, "table_columns.csv")
        self.common_columns_file = os.path.join(data_path, "common_columns.csv")
        self.table_spec_columns_file = os.path.join(data_path, "table_spec_columns.csv")

        # 确保目录存在并妥善处理错误
        try:
            os.makedirs(self.data_path, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to create data directory '{self.data_path}': {e}") from e

        # 初始化缓存
        self._table_info_cache = None
        self._table_columns_cache = None
        self._common_columns_cache = None
        self._table_spec_columns_cache = None
        self._sql_example_cache = None
        self._table_selection_example_cache = None

        self._data_warehouse_config = data_warehouse_config
        try:
            self._sql_engine = create_sqlalchemy_engine_instance(data_warehouse_config)
        except Exception as e:
            logger.warning(f"Failed to create SQL engine: {e}. Some catalog operations may not work.")
            self._sql_engine = None

    def _clear_cache(self) -> None:
        """清除所有缓存数据，以确保数据修改后的一致性"""
        self._table_info_cache = None
        self._table_columns_cache = None
        self._common_columns_cache = None
        self._table_spec_columns_cache = None
        self._sql_example_cache = None
        self._table_selection_example_cache = None
        logger.debug("Cleared all caches")

    def get_data_warehouse_config(self) -> dict:
        return self._data_warehouse_config

    def get_sql_engine(self) -> Engine:
        if self._sql_engine is None:
            raise RuntimeError("SQL engine is not available. Check data warehouse configuration.")
        return self._sql_engine

    def _validate_table_name(self, table: str) -> bool:
        """验证表名

        Args:
            table (str): 表名

        Returns:
            bool: 表名是否有效

        Raises:
            ValueError :如果表名无效
        """
        if not table or not isinstance(table, str):
            raise ValueError("Table name must be a non-empty string")

        # 检查无效字符( db.table 格式的allow dots)
        invalid_chars = ["/", "\\", "*", "?", "<", ">", "|", '"', "'"]
        if any(char in table for char in invalid_chars):
            raise ValueError(f"Table name contains invalid characters: {table}")

        return True

    def _validate_column_data(self, columns: list[dict[str, Any]]) -> bool:
        """校验列数据格式

        Args:
            columns (List[Dict[str, Any]]): 栏次信息列表

        Returns:
            bool: 列数据是否有效

        Raises:
            ValueError :如果列数据无效
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")

        required_fields = {"column_name", "type"}

        for i, column in enumerate(columns):
            if not isinstance(column, dict):
                raise ValueError(f"Column {i} must be a dictionary")

            # 检查需要的字段
            missing_fields = required_fields - set(column.keys())
            if missing_fields:
                raise ValueError(f"Column {i} missing required fields: {missing_fields}")

            # 验证 column_name
            column_name = column.get("column_name")
            if not isinstance(column_name, str) or not column_name.strip():
                raise ValueError(f"Column {i}: column_name must be a non-empty string")

            # 校验类型
            column_type = column.get("type")
            if not isinstance(column_type, str) or not column_type.strip():
                raise ValueError(f"Column {i}: type must be a non-empty string")

        return True

    def _validate_table_information(self, information: dict[str, Any]) -> bool:
        """验证表格信息格式

        Args:
            information (Dict[str, Any]): 表格资料

        Returns:
            bool: 表格信息是否有效

        Raises:
            ValueError :如果表格信息无效
        """
        if not isinstance(information, dict):
            raise ValueError("Table information must be a dictionary")

        # 验证可选字符串字段
        string_fields = ["description", "selection_rule"]
        for field in string_fields:
            if field in information:
                value = information[field]
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"Table information field '{field}' must be a string or None")

        return True

    def _validate_sql_examples(self, examples: list[dict[str, str]]) -> bool:
        """验证 SQL 示例格式

        Args:
            examples (List[Dict[str, str]]): SQL 实例列表

        Returns:
            bool: SQL 示例是否有效

        Raises:
            ValueError :如果 SQL 实例无效
        """
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list")

        required_fields = {"question", "answer"}

        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")

            # 检查需要的字段
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                raise ValueError(f"Example {i} missing required fields: {missing_fields}")

            # 校验字段为非空字符串
            for field in required_fields:
                value = example.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Example {i}: {field} must be a non-empty string")

        return True

    @staticmethod
    def _load_yaml_file(file_path: str) -> dict:
        """加载 YAML 文件

        Args:
            file_path (str): 文件路径

        Returns:
            命令: YAML 内容
        """
        if not os.path.exists(file_path):
            logger.debug(f"YAML file does not exist: {file_path}")
            return {}

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"Successfully loaded YAML file: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return {}

    @staticmethod
    def _load_csv_file(file_path: str) -> list[dict[str, str]]:
        """加载 CSV 文件

        Args:
            file_path (str): 文件路径

        Returns:
            List[Dict[str, str]]: 作为词典的行列表
        """
        if not os.path.exists(file_path):
            logger.debug(f"CSV file does not exist: {file_path}")
            return []

        try:
            result = []
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    result.append(row)
            logger.debug(f"Successfully loaded CSV file: {file_path} with {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return []

    @staticmethod
    def _save_yaml_file(file_path: str, data: dict) -> bool:
        """保存 YAML 文件

        Args:
            file_path (str): 文件路径
            data (Dict): 要保存的数据

        Returns:
            bool: 保存是否成功
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return False

    @staticmethod
    def _save_csv_file(file_path: str, data: list[dict[str, Any]], headers: list[str] | None = None) -> bool:
        """保存 CSV 文件

        Args:
            file_path (str): 文件路径
            data (List[Dict[str, Any]]): 作为词典的行列表
            headers (List[str]): 以顺序排列的页眉名称列表

        Returns:
            bool: 保存是否成功
        """
        try:
            if not data:
                return True

            # 从所有行获取所有可能的标题
            all_headers: set[str] = set()
            for row in data:
                all_headers.update(row.keys())

            # 如果指定 field_names ,请确定所有密钥都在 field_names 中
            if headers is not None:
                for key in all_headers:
                    if key not in headers:
                        headers.append(key)

            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers or sorted(all_headers))
                writer.writeheader()
                for row in data:
                    writer.writerow(row)

            return True
        except Exception as e:
            logger.error(f"Failed to save CSV file {file_path}: {e}")
            logger.error(traceback.format_stack())
            return False

    def _load_tables(self) -> dict[str, list[str]]:
        # 加载 table_columns.csv
        table_columns_csv = self._load_csv_file(self.table_columns_file)

        # 获得独特的 db_name . table_name 组合
        table_dict: dict[str, list[str]] = {}
        for row in table_columns_csv:
            if "db_name" in row and "table_name" in row and "column_name" in row:
                db_name = row["db_name"]
                table_name = row["table_name"]
                column_name = row["column_name"]
                full_table_name, _, _ = split_db_table_name(table_name, db_name)
                if full_table_name not in table_dict:
                    table_dict[full_table_name] = []
                table_dict[full_table_name].append(column_name)
        return table_dict

    def _load_common_columns(self) -> dict[str, dict[str, Any]]:
        # 加载 common_columns.csv 以获取列细节
        columns_csv = self._load_csv_file(self.common_columns_file)

        # 过滤和返回列细节
        column_dict = {}
        for row in columns_csv:
            if row.get("column_name") and row.get("type"):
                # 将行转换为 Dict[str, Any]
                column_info = {}
                for key, value in row.items():
                    if key != "":
                        column_info[key] = value
                column_dict[row["column_name"]] = column_info

        return column_dict

    def _load_table_spec_columns(self) -> dict[str, dict[str, Any]]:
        """装入表格光谱列的信息
        Returns:
            Dict[str, Dict[str, Any]]: 表格特定列信息的词典，按 "full_table_name:column_name" 键
        """
        # 加载 table_spec_columns.csv 以获取表格特定列的细节
        columns_csv = self._load_csv_file(self.table_spec_columns_file)

        # 过滤和返回列细节
        column_dict = {}
        for row in columns_csv:
            if "db_name" in row and "table_name" in row and "column_name" in row and row["column_name"]:
                # 转换行为 Dict [( str,st, Any]
                full_table_name, _, _ = split_db_table_name(row["table_name"], row["db_name"])
                column_info = {}
                for key, value in row.items():
                    if key != "":
                        column_info[key] = value
                column_dict[f"{full_table_name}:{row['column_name']}"] = column_info

        return column_dict

    def _parse_example_text(self, example_text: str) -> list[tuple[str, str]]:
        """解析示例文本，格式为Q:... A:...

        Args:
            example_text (str): 示例文本

        Returns:
            List[Tuple[str, str]]: 解析的答题对列表
        """
        examples = []
        lines = example_text.strip().split("\n")

        question = ""
        answer = ""
        current_type = None

        for line in lines:
            if line.startswith("Q:"):
                # 如果已经有一个完整的回答对，请添加到结果中
                if question and answer:
                    examples.append((question.strip(), answer.strip()))
                    question = ""
                    answer = ""

                question = line[2:]
                current_type = "Q"
            elif line.startswith("A:"):
                answer = line[2:]
                current_type = "A"
            else:
                # 继续添加到当前类型
                if current_type == "Q":
                    question += "\n" + line
                elif current_type == "A":
                    answer += "\n" + line

        # 添加最后一个答题对
        if question and answer:
            examples.append((question.strip(), answer.strip()))

        return examples

    def get_database_list(self) -> list[str]:
        # 提取唯一数据库名称
        databases = set()
        for table in self._get_all_table_schema().keys():
            full_table_name, db_name, table_name = split_db_table_name(table)
            databases.add(db_name)

        return list(databases)

    def _get_all_table_schema(self) -> dict[str, list[str]]:
        """获取所有表结构(表列)
        Returns:
            Dict[str, List[str]]: 按表名键的表格模式(列) dict
        """
        if self._table_columns_cache is None:
            self._table_columns_cache = self._load_tables()
        # 返回深层副本以防止外部修改
        return {k: v.copy() for k, v in self._table_columns_cache.items()}

    def get_table_list(self, database: str | None = None) -> list[str]:
        tables = self._get_all_table_schema()
        if database is None:
            return list(tables.keys())

        # 通过数据库过滤
        filtered_tables = []
        for full_table_name in tables.keys():
            _, db_name, table_name = split_db_table_name(full_table_name)
            if db_name == database:
                filtered_tables.append(full_table_name)

        return filtered_tables

    def _get_common_columns(self) -> dict[str, dict[str, Any]]:
        """获取所有常见列的信息
        Returns:
            Dict[str, Dict[str, Any]]: 列信息词典，按列名键
        """
        if self._common_columns_cache is None:
            self._common_columns_cache = self._load_common_columns()
        # 返回深层副本以防止外部修改
        return {k: v.copy() for k, v in self._common_columns_cache.items()}

    def _get_table_spec_columns(self) -> dict[str, dict[str, Any]]:
        """获取所有表格特定列的信息
        Returns:
            Dict[str, Dict[str, Any]]: 表格特定列信息的词典，按 "full_table_name:column_name" 键
        """
        if self._table_spec_columns_cache is None:
            self._table_spec_columns_cache = self._load_table_spec_columns()
        # 返回深层副本以防止外部修改
        return {k: v.copy() for k, v in self._table_spec_columns_cache.items()}

    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        _common_columns = self._get_common_columns()
        if table is None:
            return list(_common_columns.values())

        # 获取完整的表名
        full_table_name, db_name, table_name = split_db_table_name(table, database)

        # 过滤表格列
        tables_dict = self._get_all_table_schema()
        if full_table_name not in tables_dict:
            return []

        table_columns = tables_dict[full_table_name]

        # 如果没有找到列，请返回空列表
        if not table_columns:
            return []

        # 过滤和返回列细节
        result = []
        _table_spec_columns = self._get_table_spec_columns()
        for column in table_columns:
            # 检查列是否为表格特定
            key = f"{full_table_name}:{column}"
            if key in _table_spec_columns:
                column_info = _table_spec_columns[key]
                column_info["is_common"] = False
                result.append(column_info)
            else:
                common_column_info = _common_columns.get(column)
                if common_column_info:
                    common_column_info["is_common"] = True
                    result.append(common_column_info)
        return result

    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        full_table_name, db_name, table_name = split_db_table_name(table, database)

        if self._table_info_cache is None:
            self._table_info_cache = self._load_yaml_file(self.table_info_file)

        if db_name in self._table_info_cache and table_name in self._table_info_cache[db_name]:
            # 返回副本以防止外部修改
            return dict(self._table_info_cache[db_name][table_name])

        return {}

    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        if self._sql_example_cache is None:
            self._sql_example_cache = self._load_yaml_file(self.sql_example_file)

        if table is None:
            # 如果没有指定表格，请返回所有示例
            examples = []
            for db_name, tables in self._sql_example_cache.items():
                for table_name, example_text in tables.items():
                    qa_pairs = self._parse_example_text(example_text)
                    full_table_name, _, _ = split_db_table_name(table_name, db_name)
                    examples.extend([(q, a, [full_table_name]) for (q, a) in qa_pairs])
            return examples

        full_table_name, db_name, table_name = split_db_table_name(table, database)

        # 查找包含此表格的示例
        examples = []

        # 检查事实部分
        if db_name in self._sql_example_cache:
            if table_name in self._sql_example_cache[db_name]:
                # 解析示例文本，格式为Q:... A:...
                qa_pairs = self._parse_example_text(self._sql_example_cache[db_name][table_name])
                examples.extend([(q, a, [full_table_name]) for (q, a) in qa_pairs])

        return examples

    @staticmethod
    def _parse_selected_tables(raw: str) -> list[str]:
        """将 ``selected_tables`` CSV 单元格分析为表名的干净列表。

        光碟上的光谱格式是一个 JSON 阵列(如 ``["Customers", "Orders"]``).
        对于后向兼容性，此选项也接受 Python 列表的字元
        (从旧的，丢失的序列化)和一个平面空间/comma/新线
        分离字符串。
        """
        raw = raw.strip()
        if not raw:
            return []

        # 首选: JSON 阵列。
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

        # 后向-compat: Python 列表词义(如"[ 'Customers' ]").
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (ValueError, SyntaxError):
                pass

        # 后向-对齐:平面分离的字符串。
        return [p.strip() for p in re.split(r"[ ,\n]", raw) if p.strip()]

    @classmethod
    def _load_table_selection_examples_from_csv(cls, file_path: str) -> list[tuple[str, list[str]]]:
        examples = []
        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row.get("question", "").strip()
                    selected_tables = row.get("selected_tables", "").strip()
                    if question and selected_tables:
                        examples.append((question, cls._parse_selected_tables(selected_tables)))
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to load table selection examples from {file_path}: {e}")
        return examples

    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        if self._table_selection_example_cache is None:
            self._table_selection_example_cache = self._load_table_selection_examples_from_csv(
                self.table_selection_example_file
            )
        return self._table_selection_example_cache

    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        # 验证输入数据(让验证错误被公布)
        self._validate_table_name(table)
        self._validate_table_information(information)
        self._validate_column_data(columns)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)

            table_info = self._load_yaml_file(self.table_info_file)

            # 首先保存列
            if not self._save_columns(table_name, columns, db_name, update_existing):
                logger.error(f"Failed to save columns for table {full_table_name}")
                return False

            # 保存表格信息(确保适当的结构)
            if db_name not in table_info:
                table_info[db_name] = {}
            if update_existing or table_name not in table_info[db_name]:
                table_info[db_name][table_name] = information
            success = self._save_yaml_file(self.table_info_file, table_info)

            if success:
                logger.info(f"Successfully saved table information for {full_table_name}")
                # 清除缓存以确保一致性
                self._clear_cache()

            return success
        except Exception as e:
            logger.error(f"Unexpected error when saving table information: {e}")
            logger.error(traceback.format_stack())
            return False

    def _save_columns(
        self, table_name: str, columns: list[dict[str, Any]], db_name: str = "", update_existing: bool = False
    ) -> bool:
        """将列信息保存到 common_columns.csv 和表列保存到 table_columns.csv

        Args:
            table_name (str): 表名
            columns (List[Dict[str, Any]]): 栏次信息列表
            db_name (str): 数据库名称
            update_existing (bool): 更新现有列信息

        Returns:
            bool: 保存是否成功
        """
        full_table_name, db_name, table_name = split_db_table_name(table_name, db_name)
        # 装入已有数据
        tables_data = self._load_csv_file(self.table_columns_file)
        common_columns_dict = self._load_common_columns()
        table_spec_columns_dict = self._load_table_spec_columns()

        # 创建一组现有的表格列组合
        existing_table_columns = set()
        for row in tables_data:
            if "db_name" in row and "table_name" in row and "column_name" in row:
                existing_full_table_name, _, _ = split_db_table_name(row["table_name"], row["db_name"])
                key = f"{existing_full_table_name}:{row['column_name']}"
                existing_table_columns.add(key)

        # 更新 table_columns.csv 并跟踪要添加的新列

        for column in columns:
            if "column_name" not in column:
                continue

            column_name = column["column_name"]
            is_common_column = column.get("is_common", False)

            key = f"{full_table_name}:{column_name}"
            column_info = {k: str(v) for k, v in column.items() if k != "is_common"}
            if not is_common_column:
                column_info["db_name"] = db_name
                column_info["table_name"] = table_name

            # 表格中的新列 -> 添加到 table_columns.csv
            if key not in existing_table_columns:
                tables_data.append({"db_name": db_name, "table_name": table_name, "column_name": column_name})
                existing_table_columns.add(key)
                if is_common_column:
                    # 处理 common_columns.csv 并去重。
                    if column_name not in common_columns_dict:
                        # 将新列添加到 columns_data
                        logger.info(f"Add new column column {column_name}")
                        common_columns_dict[column_name] = column_info
                else:
                    table_spec_columns_dict[key] = column_info
            # 应用更新到 columns_data 中的现有列
            elif update_existing:
                if is_common_column:
                    common_columns_dict[column_name] = column_info
                else:
                    table_spec_columns_dict[key] = column_info

        # 保存更新的数据
        tables_success = self._save_csv_file(
            self.table_columns_file, tables_data, ["db_name", "table_name", "column_name"]
        )
        common_columns_success = self._save_csv_file(
            self.common_columns_file,
            list(common_columns_dict.values()),
            ["column_name", "display_name", "alias", "type", "category", "tag", "description"],
        )
        table_spec_columns_success = self._save_csv_file(
            self.table_spec_columns_file,
            list(table_spec_columns_dict.values()),
            ["db_name", "table_name", "column_name", "display_name", "alias", "type", "category", "tag", "description"],
        )

        success = tables_success and common_columns_success and table_spec_columns_success
        if success:
            # 清除缓存以确保一致性
            self._clear_cache()
            logger.debug(f"Successfully saved columns for table {table_name}")

        return success

    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        # 验证输入数据(让验证错误被公布)
        self._validate_table_name(table)
        self._validate_sql_examples(examples)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)

            sql_examples = self._load_yaml_file(self.sql_example_file)

            # 确保结构中存在数据库
            if db_name not in sql_examples:
                sql_examples[db_name] = {}

            # 示例文本
            example_text = ""
            for example in examples:
                example_text += f"Q: {example['question']}\nA: {example['answer']}\n\n"

            sql_examples[db_name][table_name] = example_text.strip()

            success = self._save_yaml_file(self.sql_example_file, sql_examples)

            if success:
                logger.info(f"Successfully saved {len(examples)} examples for table {full_table_name}")
                # 更新缓存
                self._sql_example_cache = sql_examples

            return success
        except Exception as e:
            logger.error(f"Unexpected error when saving table examples: {e}")
            logger.error(traceback.format_stack())
            return False

    def append_sql_example(
        self,
        question: str,
        sql: str,
        tables: list[str],
        source: str = "golden",
        database: str | None = None,
    ) -> bool:
        self._validate_sql_examples([{"question": question, "answer": sql}])
        try:
            target_table = tables[0] if tables else ""
            full_table_name, db_name, table_name = split_db_table_name(target_table, database)

            sql_examples = self._load_yaml_file(self.sql_example_file)
            if db_name not in sql_examples:
                sql_examples[db_name] = {}
            existing = sql_examples[db_name].get(table_name, "")

            # 解答问题文本("Q:..."行)——只有新内容才附加。
            if f"Q: {question}\n" in existing:
                logger.info(f"Golden SQL example already present for table {full_table_name}; skipping append.")
                return True

            new_block = f"Q: {question}\nA: {sql}\n"
            sql_examples[db_name][table_name] = (existing + "\n\n" + new_block).strip() if existing else new_block

            success = self._save_yaml_file(self.sql_example_file, sql_examples)
            if success:
                logger.info(f"Appended {source} SQL example for table {full_table_name}")
                self._sql_example_cache = sql_examples
            return success
        except Exception as e:
            logger.error(f"Unexpected error when appending SQL example: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        example_data = []
        for example in examples:
            # 将表格列表序列化为 JSON 阵列，这样它可以毫无损失地绕行。
            example_data.append({"question": example[0], "selected_tables": json.dumps(list(example[1]))})
        save_success = self._save_csv_file(
            self.table_selection_example_file, example_data, ["question", "selected_tables"]
        )
        if save_success:
            logger.info(f"Successfully saved {len(examples)} table selection examples.")
        return save_success

    def check_exists(self) -> bool:
        try:
            # 检查必要的目录文件是否存在并有内容
            files_missing = (
                not os.path.exists(self.table_columns_file)
                or not os.path.exists(self.common_columns_file)
                or os.path.getsize(self.table_columns_file) <= 1  # 空或只是页眉
                or os.path.getsize(self.common_columns_file) <= 1
            )

            return not files_missing

        except Exception as e:
            logger.warning(f"Error checking catalog existence: {e}")
            logger.error(traceback.format_stack())
            return False
