from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import Engine


class CatalogStore(ABC):
    """抽象基础类定义数据目录的存储界面(数据库，表格，列定义，描述，和附加提示).

    各表之间含义相同的常见栏目将集中储存，以避免数据重复。

    列属性 :

        - column_name: 栏名
        - display_name: 列的显示名称
        - 类型:列的数据类型
        - 类别:尺寸或度数
        - 说明:栏的说明
        - is_common: 是否为普通列
    """

    @abstractmethod
    def get_data_warehouse_config(self) -> dict:
        """获取数据仓库配置

        Returns:
            dict: 数据仓库配置
        """
        pass

    @abstractmethod
    def get_sql_engine(self) -> Engine:
        """获取目录的 SQLAlchemy 引擎

        Returns:
            引擎: SQLAlchemy 引擎
        """
        pass

    @abstractmethod
    def get_database_list(self) -> list[str]:
        """获取所有数据库的列表

        Returns:
            List[str]: 数据库名称清单
        """
        pass

    @abstractmethod
    def get_table_list(self, database: str | None = None) -> list[str]:
        """获取指定数据库中所有表字段表，如果数据库为 None ,请返回所有表格

        Args:
            database (Optional[str]): 数据库名称

        Returns:
            List[str]: 表名列表
        """
        pass

    @abstractmethod
    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        """获取指定表格的所有列信息，如果表格为 None , 返回目录中所有常见列

        Args:
            table (Optional[str]): 表名
            database (Optional[str]): 数据库名称

        Returns:
            List[Dict[str, Any]]: 列信息列表，每列包含名称，类型，说明等内容。
        """
        pass

    @abstractmethod
    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        """获取指定表格的信息

        Args:
            table (str): 表名
            database (Optional[str]): 数据库名称

        Returns:
            Dict[str, Any]: 表格信息，包括描述文本，选择规则等。
        """
        pass

    @abstractmethod
    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        """获取 SQL 示例

        Args:
            table (Optional[str]): 表名
            database (Optional[str]): 数据库名称

        Returns:
            List[Tuple[str, str, List[str]]]: SQL 实例列表，每个实例都是图普-3作为(问题，SQL , full_table_names).
        """
        pass

    @abstractmethod
    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        """获取表格选择示例

        Returns:
            List[Tuple[str, List[str]]]: 表格选择示例列表，每个示例为 Tuple-2作为(问题，选定表格)
        """
        pass

    @abstractmethod
    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        """将信息和列保存到表格中

        Args:
            table (str): 表名
            information (Dict[str, Any]): 表格资料
            columns (List[Dict[str, Any]]): 各栏信息列表，每一栏信息在租赁时包含
                column_name 类型、类别、描述
            database (Optional[str]): 数据库名称
            update_existing (bool): 更新现有表格和列信息

        Returns:
            bool: 保存是否成功
        """
        pass

    @abstractmethod
    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        """将 SQL 示例保存到表格

        Args:
            table (str): 表名
            examples (List[Dict[str, str]]): SQL 实例列表
            database (Optional[str]): 数据库名称

        Returns:
            bool: 保存是否成功
        """
        pass

    @abstractmethod
    def append_sql_example(
        self,
        question: str,
        sql: str,
        tables: list[str],
        source: str = "golden",
        database: str | None = None,
    ) -> bool:
        """附加一个单Q-> SQL 的例子，解析这个问题。

        与 ``save_table_sql_examples`` 不同的是(它覆盖了每张表格)
        示例块),这保存了已有示例，只添加
        当问题还没有出现时，新的问题。

        Args:
            question (str): 自然语问。
            sql (str): 经验证的 SQL 答案。
            tables (list[str]): 完整表名引用 SQL。
            source (str): 证明标记( 'golden' 为人类批准的标记).
            database (Optional[str]): 数据库名称覆盖。

        Returns:
            bool: 附件是否成功。
        """
        pass

    @abstractmethod
    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        """保存表格选择示例

        Args:
            examples (List[Tuple[str, List[str]]]): 表格选择示例列表

        Returns:
            bool: 保存是否成功
        """
        pass

    @abstractmethod
    def check_exists(self) -> bool:
        """检查目录存储是否有已有的数据/内容

        Returns:
            bool: True 如果目录存储有已有数据，如果为空或缺少基本文件，False
        """
        pass


def split_db_table_name(table: str, database: str | None = None) -> tuple[str, str, str]:
    """将完整表名拆分为 db 名称和表名
    Args:
        table (str): 如果数据库是 None ,则应该像 `db.table` 那样是完整的表名，否则应该只为表名
        database (Optional[str]): 数据库名称
    Returns:
        Tuple[str, str, str]: full_table_name, db_name, table_name
    """
    full_table_name = table
    if database and "." not in table:
        full_table_name = f"{database}.{table}"
    if "." in full_table_name:
        db_name, table_name = full_table_name.rsplit(".", 1)
    else:
        db_name = ""
        table_name = full_table_name
    return full_table_name, db_name, table_name
