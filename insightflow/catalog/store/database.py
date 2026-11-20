"""数据库支持的目录存储执行。

此模块提供: class: `DatabaseCatalogStore` ,一个 SQLAlchemy 2.0 备份
执行: class: `~insightflow.catalog.catalog_store.CatalogStore` .

与 : 类: `~insightflow.catalog.store.file_system.FileSystemCatalogStore` 相比
这个后端存储了 CSV / YAML 文件。
目录数据包括数据库、表、字段、SQL 示例和选表示例。
在一个关系数据库内，允许同时访问、交易、
指数化和跨线共享。

支持的连接样式 :

- ``connection_string`` :任意 SQLAlchemy 数据库 URI ,如 ``sqlite:///catalog.db`` ,,
  ``postgresql+psycopg://user:pass@host/db`` or ``mysql+pymysql://user:pass@host/db``.
- ``engine`` :一个外部创建的 SQLAlchemy : class: `~sqlalchemy.Engine` :
  例如，用于连接池共享和测试。

重要: 目录持久性数据库的配置独立于
``data_warehouse_config`` . ``data_warehouse_config`` 仅用于构建
数据仓库执行引擎返回 :meth: `get_sql_engine` ;它从不
重复使用为目录 DB 连接。
"""

import logging
import traceback
from typing import Any

from sqlalchemy import (
    JSON,
    Engine,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..catalog_store import CatalogStore, split_db_table_name
from ..helper import create_sqlalchemy_engine_instance

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """所有目录 ORM 模型的宣告基础。"""


class CatalogDatabase(Base):
    """逻辑数据库(namespace),用于分组目录表。"""

    __tablename__ = "catalog_databases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    tables: Mapped[list["CatalogTable"]] = relationship(back_populates="database")


class CatalogTable(Base):
    """带有描述性元数据的目录表。"""

    __tablename__ = "catalog_tables"
    __table_args__ = (UniqueConstraint("database_id", "table_name", name="uq_catalog_table_db_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    database_id: Mapped[int] = mapped_column(ForeignKey("catalog_databases.id"), index=True)
    table_name: Mapped[str] = mapped_column(String(255), index=True)

    description: Mapped[str] = mapped_column(Text, default="")
    selection_rule: Mapped[str] = mapped_column(Text, default="")
    sql_rule: Mapped[str] = mapped_column(Text, default="")
    derived_metric: Mapped[str] = mapped_column(Text, default="")
    # 保存任何额外的表格信息密钥，用于圆通忠贞。
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    database: Mapped["CatalogDatabase"] = relationship(back_populates="tables")
    sql_examples: Mapped[list["CatalogSqlExample"]] = relationship(back_populates="table", cascade="all, delete-orphan")


class CatalogColumn(Base):
    """专栏定义。

    专栏定义本身并不意味着"本表拥有此专栏".
    表栏成员通过: 类: `CatalogTableColumn` 表示，故 a.
    单一的共同栏定义可以在多个表格中重复使用。

    纵列是* common*(全球共享)还是* 表格特有* 得出
    仅来自 ``scope_table_id`` :

    - ``scope_table_id IS NULL`` -> 常见列( ``is_common=True`` in API).
    - ``scope_table_id = <id>`` ->该表的专栏范围
      ( ``is_common=False``),因此它从未被其他表格意外重复使用。

    故意没有存储的 ``is_common`` 布尔:它会重复
    信息已被 ``scope_table_id`` 编码，并冒险让两人离开
    。 ``is_common`` 旗帜在输入时仍然被接受，并返回到
    输出后向兼容性。
    """

    __tablename__ = "catalog_columns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    column_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    alias: Mapped[str] = mapped_column(String(255), default="")
    type: Mapped[str] = mapped_column(String(255), default="")
    category: Mapped[str] = mapped_column(String(255), default="")
    tag: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    default: Mapped[str] = mapped_column("default_value", String(255), default="")
    # NULL 表示全局共享字段，设定表后该字段仅在对应表内适用。
    # 到自有表格。此单字段编码了普通/表格特定拆分。
    scope_table_id: Mapped[int | None] = mapped_column(ForeignKey("catalog_tables.id"), nullable=True, index=True)
    # 保存任何额外的列元数据密钥，用于回合忠诚。
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class CatalogTableColumn(Base):
    """明确表栏成员(图示遗产 ``table_columns.csv``)."""

    __tablename__ = "catalog_table_columns"
    __table_args__ = (UniqueConstraint("table_id", "column_id", name="uq_catalog_table_column"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("catalog_tables.id"), index=True)
    column_id: Mapped[int] = mapped_column(ForeignKey("catalog_columns.id"), index=True)


class CatalogSqlExample(Base):
    """每表自然语言问题 / SQL 答例。"""

    __tablename__ = "catalog_sql_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("catalog_tables.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)

    table: Mapped["CatalogTable"] = relationship(back_populates="sql_examples")


class CatalogTableSelectionExample(Base):
    """全局表格选择示例(问题 -> 选定表格)。"""

    __tablename__ = "catalog_table_selection_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    selected_tables: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


# 列元数据字段为一级 ORM 列建模。
_KNOWN_COLUMN_FIELDS = {
    "column_name",
    "display_name",
    "alias",
    "type",
    "category",
    "tag",
    "description",
    "default",
    "is_common",
}

# 表格信息字段为一等的 ORM 列模式。
_KNOWN_TABLE_FIELDS = {"description", "selection_rule", "sql_rule", "derived_metric"}


class DatabaseCatalogStore(CatalogStore):
    """SQLAlchemy后置目录存储。

    Args:
        connection_string (Optional[str]): 用于目录的 SQLAlchemy 数据库 URI
            与 ``engine`` (需要一个) 互用。
        engine (Optional[Engine]): 为目录 DB 预建的SQLAlchemy引擎。
        data_warehouse_config (Optional[dict]): 使用的数据仓库配置
            *只用于构建数据仓库执行引擎
            :meth: `get_sql_engine` . 从未作为目录 DB 连接。
        auto_create_schema (bool): ``True`` ( 默认) 运行时
            ``Base.metadata.create_all()`` 初始化后需要
            为开发和测试方便而创建表格。设置到
            ``False`` 用于外部管理的方案。
        echo (bool): SQLAlchemy 引擎 ``echo`` 旗 (仅在创建时使用)
            引擎来自 ``connection_string`` .).
    """

    def __init__(
        self,
        connection_string: str | None = None,
        engine: Engine | None = None,
        data_warehouse_config: dict[str, Any] | None = None,
        auto_create_schema: bool = True,
        echo: bool = False,
    ) -> None:
        if engine is None and (not connection_string or not connection_string.strip()):
            raise ValueError("Either 'engine' or a non-empty 'connection_string' must be provided")

        if data_warehouse_config is None:
            data_warehouse_config = {}
        elif not isinstance(data_warehouse_config, dict):
            raise ValueError("data_warehouse_config must be a dictionary")

        self._engine: Engine = engine if engine is not None else create_engine(connection_string, echo=echo)  # type: ignore[arg-type]
        self._session_factory = sessionmaker(bind=self._engine)

        self._data_warehouse_config = data_warehouse_config
        self._sql_engine: Engine | None = None

        if auto_create_schema:
            Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # 数据仓库访问( 独立于目录 DB)
    # ------------------------------------------------------------------
    def get_data_warehouse_config(self) -> dict:
        return self._data_warehouse_config

    def get_sql_engine(self) -> Engine:
        if self._sql_engine is None:
            try:
                self._sql_engine = create_sqlalchemy_engine_instance(self._data_warehouse_config)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("SQL engine is not available. Check data warehouse configuration.") from e
        return self._sql_engine

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _full_name(db_name: str, table_name: str) -> str:
        """建立完整的表名，避免为空数据库设置主点。"""
        return f"{db_name}.{table_name}" if db_name else table_name

    @staticmethod
    def _validate_table_name(table: str) -> None:
        if not table or not isinstance(table, str):
            raise ValueError("Table name must be a non-empty string")
        invalid_chars = ["/", "\\", "*", "?", "<", ">", "|", '"', "'"]
        if any(char in table for char in invalid_chars):
            raise ValueError(f"Table name contains invalid characters: {table}")

    @staticmethod
    def _validate_sql_examples(examples: list[dict[str, str]]) -> None:
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list")
        required_fields = {"question", "answer"}
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                raise ValueError(f"Example {i} missing required fields: {missing_fields}")
            for field in required_fields:
                value = example.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Example {i}: {field} must be a non-empty string")

    def _get_database(self, session: Session, db_name: str) -> CatalogDatabase | None:
        return session.scalars(select(CatalogDatabase).where(CatalogDatabase.name == db_name)).first()

    def _get_or_create_database(self, session: Session, db_name: str) -> CatalogDatabase:
        db_row = self._get_database(session, db_name)
        if db_row is None:
            db_row = CatalogDatabase(name=db_name)
            session.add(db_row)
            session.flush()
        return db_row

    def _get_table_row(self, session: Session, table: str, database: str | None) -> CatalogTable | None:
        _, db_name, table_name = split_db_table_name(table, database)
        stmt = (
            select(CatalogTable)
            .join(CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id)
            .where(CatalogDatabase.name == db_name, CatalogTable.table_name == table_name)
        )
        return session.scalars(stmt).first()

    def _get_or_create_table(self, session: Session, db_name: str, table_name: str) -> CatalogTable:
        db_row = self._get_or_create_database(session, db_name)
        session.flush()
        row = session.scalars(
            select(CatalogTable).where(CatalogTable.database_id == db_row.id, CatalogTable.table_name == table_name)
        ).first()
        if row is None:
            row = CatalogTable(database_id=db_row.id, table_name=table_name)
            session.add(row)
            session.flush()
        return row

    @staticmethod
    def _column_to_dict(col: CatalogColumn) -> dict[str, Any]:
        data: dict[str, Any] = {
            "column_name": col.column_name,
            "display_name": col.display_name,
            "alias": col.alias,
            "type": col.type,
            "category": col.category,
            "tag": col.tag,
            "description": col.description,
            "default": col.default,
        }
        if col.extra:
            for key, value in col.extra.items():
                data.setdefault(key, value)
        # 常见 vs 表格特异性来源于 scope_table_id .
        data["is_common"] = col.scope_table_id is None
        return data

    # ------------------------------------------------------------------
    # 读取操作
    # ------------------------------------------------------------------
    def get_database_list(self) -> list[str]:
        with self._session_factory() as session:
            stmt = (
                select(CatalogDatabase.name)
                .join(CatalogTable, CatalogTable.database_id == CatalogDatabase.id)
                .distinct()
            )
            return list(session.scalars(stmt).all())

    def get_table_list(self, database: str | None = None) -> list[str]:
        with self._session_factory() as session:
            stmt = select(CatalogDatabase.name, CatalogTable.table_name).join(
                CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id
            )
            if database is not None:
                stmt = stmt.where(CatalogDatabase.name == database)
            stmt = stmt.order_by(CatalogTable.id)
            rows = session.execute(stmt).all()
            return [self._full_name(db_name, table_name) for db_name, table_name in rows]

    def get_column_list(self, table: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            if table is None:
                # 常见的栏目是没有列入任何具体表格的栏目。
                stmt = select(CatalogColumn).where(CatalogColumn.scope_table_id.is_(None)).order_by(CatalogColumn.id)
                return [self._column_to_dict(col) for col in session.scalars(stmt).all()]

            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return []

            stmt = (
                select(CatalogColumn)
                .join(CatalogTableColumn, CatalogTableColumn.column_id == CatalogColumn.id)
                .where(CatalogTableColumn.table_id == table_row.id)
                .order_by(CatalogTableColumn.id)
            )
            return [self._column_to_dict(col) for col in session.scalars(stmt).all()]

    def get_table_information(self, table: str, database: str | None = None) -> dict[str, Any]:
        with self._session_factory() as session:
            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return {}
            info: dict[str, Any] = {
                "description": table_row.description or "",
                "selection_rule": table_row.selection_rule or "",
                "sql_rule": table_row.sql_rule or "",
                "derived_metric": table_row.derived_metric or "",
            }
            if table_row.extra:
                for key, value in table_row.extra.items():
                    info.setdefault(key, value)
            return info

    def get_sql_examples(
        self, table: str | None = None, database: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        with self._session_factory() as session:
            if table is None:
                stmt = (
                    select(
                        CatalogDatabase.name,
                        CatalogTable.table_name,
                        CatalogSqlExample.question,
                        CatalogSqlExample.answer,
                    )
                    .join(CatalogTable, CatalogSqlExample.table_id == CatalogTable.id)
                    .join(CatalogDatabase, CatalogTable.database_id == CatalogDatabase.id)
                    .order_by(CatalogSqlExample.id)
                )
                rows = session.execute(stmt).all()
                return [
                    (question, answer, [self._full_name(db_name, table_name)])
                    for db_name, table_name, question, answer in rows
                ]

            table_row = self._get_table_row(session, table, database)
            if table_row is None:
                return []
            full_table_name = self._full_name(table_row.database.name, table_row.table_name)
            table_stmt = (
                select(CatalogSqlExample.question, CatalogSqlExample.answer)
                .where(CatalogSqlExample.table_id == table_row.id)
                .order_by(CatalogSqlExample.id)
            )
            table_rows = session.execute(table_stmt).all()
            return [(question, answer, [full_table_name]) for question, answer in table_rows]

    def get_table_selection_examples(self) -> list[tuple[str, list[str]]]:
        with self._session_factory() as session:
            stmt = select(CatalogTableSelectionExample).order_by(CatalogTableSelectionExample.id)
            return [(row.question, list(row.selected_tables or [])) for row in session.scalars(stmt).all()]

    # ------------------------------------------------------------------
    # 写入操作
    # ------------------------------------------------------------------
    def _find_column(self, session: Session, column_name: str, is_common: bool, table_id: int) -> CatalogColumn | None:
        stmt = select(CatalogColumn).where(CatalogColumn.column_name == column_name)
        if is_common:
            # 常见栏为全局性栏:不限定于任何表格。
            stmt = stmt.where(CatalogColumn.scope_table_id.is_(None))
        else:
            # 表格特定栏目范围为自有表。
            stmt = stmt.where(CatalogColumn.scope_table_id == table_id)
        return session.scalars(stmt).first()

    @staticmethod
    def _build_column_kwargs(column: dict[str, Any]) -> dict[str, Any]:
        extra = {k: str(v) for k, v in column.items() if k not in _KNOWN_COLUMN_FIELDS}
        return {
            "display_name": str(column.get("display_name", "") or ""),
            "alias": str(column.get("alias", "") or ""),
            "type": str(column.get("type", "") or ""),
            "category": str(column.get("category", "") or ""),
            "tag": str(column.get("tag", "") or ""),
            "description": str(column.get("description", "") or ""),
            "default": str(column.get("default", "") or ""),
            "extra": extra or None,
        }

    def _ensure_membership(self, session: Session, table_id: int, column_id: int) -> None:
        exists = session.scalars(
            select(CatalogTableColumn).where(
                CatalogTableColumn.table_id == table_id, CatalogTableColumn.column_id == column_id
            )
        ).first()
        if exists is None:
            session.add(CatalogTableColumn(table_id=table_id, column_id=column_id))

    def _save_columns(
        self, session: Session, table_row: CatalogTable, columns: list[dict[str, Any]], update_existing: bool
    ) -> None:
        for column in columns:
            if "column_name" not in column:
                continue
            column_name = column["column_name"]
            is_common = bool(column.get("is_common", False))

            col_row = self._find_column(session, column_name, is_common, table_row.id)
            if col_row is None:
                col_row = CatalogColumn(
                    column_name=column_name,
                    scope_table_id=None if is_common else table_row.id,
                    **self._build_column_kwargs(column),
                )
                session.add(col_row)
                session.flush()
            elif update_existing:
                for key, value in self._build_column_kwargs(column).items():
                    setattr(col_row, key, value)

            self._ensure_membership(session, table_row.id, col_row.id)

    def save_table_information(
        self,
        table: str,
        information: dict[str, Any],
        columns: list[dict[str, Any]],
        database: str | None = None,
        update_existing: bool = False,
    ) -> bool:
        # 向上传递校验错误，与 FileSystemCatalogStore 的行为一致。
        self._validate_table_name(table)
        if not isinstance(information, dict):
            raise ValueError("Table information must be a dictionary")
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)
            with self._session_factory() as session:
                db_row = self._get_or_create_database(session, db_name)
                session.flush()

                table_row = session.scalars(
                    select(CatalogTable).where(
                        CatalogTable.database_id == db_row.id, CatalogTable.table_name == table_name
                    )
                ).first()

                extra = {k: v for k, v in information.items() if k not in _KNOWN_TABLE_FIELDS}
                if table_row is None:
                    table_row = CatalogTable(
                        database_id=db_row.id,
                        table_name=table_name,
                        description=str(information.get("description", "") or ""),
                        selection_rule=str(information.get("selection_rule", "") or ""),
                        sql_rule=str(information.get("sql_rule", "") or ""),
                        derived_metric=str(information.get("derived_metric", "") or ""),
                        extra=extra or None,
                    )
                    session.add(table_row)
                    session.flush()
                elif update_existing:
                    table_row.description = str(information.get("description", "") or "")
                    table_row.selection_rule = str(information.get("selection_rule", "") or "")
                    table_row.sql_rule = str(information.get("sql_rule", "") or "")
                    table_row.derived_metric = str(information.get("derived_metric", "") or "")
                    table_row.extra = extra or None

                self._save_columns(session, table_row, columns, update_existing)
                session.commit()

            logger.info(f"Successfully saved table information for {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when saving table information: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_sql_examples(self, table: str, examples: list[dict[str, str]], database: str | None = None) -> bool:
        self._validate_table_name(table)
        self._validate_sql_examples(examples)

        try:
            full_table_name, db_name, table_name = split_db_table_name(table, database)
            with self._session_factory() as session:
                table_row = self._get_or_create_table(session, db_name, table_name)
                # 覆盖每个表格的示例块。
                session.execute(delete(CatalogSqlExample).where(CatalogSqlExample.table_id == table_row.id))
                for example in examples:
                    session.add(
                        CatalogSqlExample(table_id=table_row.id, question=example["question"], answer=example["answer"])
                    )
                session.commit()

            logger.info(f"Successfully saved {len(examples)} examples for table {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
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
        if not tables:
            logger.warning("No target tables provided for SQL example; skipping append.")
            return False
        try:
            target_table = tables[0]
            full_table_name, db_name, table_name = split_db_table_name(target_table, database)
            with self._session_factory() as session:
                table_row = self._get_or_create_table(session, db_name, table_name)

                existing = session.scalars(
                    select(CatalogSqlExample).where(
                        CatalogSqlExample.table_id == table_row.id, CatalogSqlExample.question == question
                    )
                ).first()
                if existing is not None:
                    logger.info(f"Golden SQL example already present for table {full_table_name}; skipping append.")
                    return True

                session.add(CatalogSqlExample(table_id=table_row.id, question=question, answer=sql))
                session.commit()

            logger.info(f"Appended {source} SQL example for table {full_table_name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when appending SQL example: {e}")
            logger.error(traceback.format_stack())
            return False

    def save_table_selection_examples(self, examples: list[tuple[str, list[str]]]) -> bool:
        try:
            with self._session_factory() as session:
                # 覆盖所有已有的选择示例。
                session.execute(delete(CatalogTableSelectionExample))
                for question, selected_tables in examples:
                    session.add(CatalogTableSelectionExample(question=question, selected_tables=list(selected_tables)))
                session.commit()

            logger.info(f"Successfully saved {len(examples)} table selection examples.")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error when saving table selection examples: {e}")
            logger.error(traceback.format_stack())
            return False

    def check_exists(self) -> bool:
        try:
            with self._session_factory() as session:
                return session.scalars(select(CatalogTable.id).limit(1)).first() is not None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error checking catalog existence: {e}")
            return False
