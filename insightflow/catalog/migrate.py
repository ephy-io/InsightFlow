"""分类移徙公用设施。

提供简单、后端不可知的迁移帮助以复制目录数据
一个: 类: `~insightflow.catalog.catalog_store.CatalogStore` 到另一个。 最多
常见用法正在将一个现有的文件系统目录迁移到一个
数据库支持的目录存储。

因为两个后端都执行相同的 ``CatalogStore`` 接口，所以迁移是
仅仅"从源头读到，写到目的地"在公共方法。

CLI 用途 :

    python -m 洞察flow.catalog.migrate --source-path ./例 --dest-connection-string sqlite://./catalog.db.

添加 ``--dry-run`` 不写入预览，和 ``--no-update-existing`` 跳过
覆盖目的地中已经存在的行。
"""

import argparse
import logging
from typing import Any

from insightflow.catalog.catalog_store import CatalogStore, split_db_table_name

logger = logging.getLogger(__name__)


def migrate_catalog(
    source: CatalogStore,
    dest: CatalogStore,
    *,
    update_existing: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """将所有目录数据从 ``source`` 复制到 ``dest``。

    复制表格元数据+列，每表 SQL 示例，以及全局
    表格选择示例。

    Args:
        source: 拟读目录店。
        dest: 拟写目录店。
        update_existing: 当 ``True`` (默认) 时，覆盖表格元数据/列
            在目的地已经存在。
        dry_run: 当 ``True`` 时，只登录计划中的操作而不写入。

    Returns:
        以数字表示的数据: ``tables``、 ``failed_tables``、
        ``sql_example_tables`` 和 ``selection_examples`` 相声。
    """
    stats: dict[str, int] = {
        "tables": 0,
        "failed_tables": 0,
        "sql_example_tables": 0,
        "selection_examples": 0,
    }

    full_table_names = source.get_table_list()
    logger.info("Found %d table(s) to migrate%s", len(full_table_names), " (dry-run)" if dry_run else "")

    for full_table_name in full_table_names:
        _, db_name, table_name = split_db_table_name(full_table_name)
        information: dict[str, Any] = source.get_table_information(full_table_name)
        columns: list[dict[str, Any]] = source.get_column_list(full_table_name)

        logger.info(
            "Migrating table '%s' (%d column(s))%s",
            full_table_name,
            len(columns),
            " [dry-run]" if dry_run else "",
        )

        if not dry_run:
            ok = dest.save_table_information(
                table_name,
                information,
                columns,
                database=db_name,
                update_existing=update_existing,
            )
            if not ok:
                logger.error("Failed to migrate table '%s'", full_table_name)
                stats["failed_tables"] += 1
                continue
        stats["tables"] += 1

        # per-table SQL 实例。
        examples = source.get_sql_examples(full_table_name)
        if examples:
            sql_examples = [{"question": question, "answer": answer} for question, answer, _ in examples]
            logger.info("  -> %d SQL example(s)", len(sql_examples))
            if not dry_run:
                dest.save_table_sql_examples(table_name, sql_examples, database=db_name)
            stats["sql_example_tables"] += 1

    # 全局表格选择示例。 丢弃空占位符行。
    selection_examples = [(q, tables) for q, tables in source.get_table_selection_examples() if q]
    if selection_examples:
        logger.info("Migrating %d table-selection example(s)", len(selection_examples))
        if not dry_run:
            dest.save_table_selection_examples(selection_examples)
        stats["selection_examples"] = len(selection_examples)

    logger.info("Migration complete: %s", stats)
    return stats


def _build_source(args: argparse.Namespace) -> CatalogStore:
    from insightflow.catalog.store.file_system import FileSystemCatalogStore

    return FileSystemCatalogStore(data_path=args.source_path)


def _build_dest(args: argparse.Namespace) -> CatalogStore:
    from insightflow.catalog.store.database import DatabaseCatalogStore

    return DatabaseCatalogStore(
        connection_string=args.dest_connection_string,
        auto_create_schema=True,
        echo=args.echo,
    )


def main(argv: list[str] | None = None) -> dict[str, int]:
    """CLI 入口:将一个文件系统目录迁移到一个数据库目录中。"""
    parser = argparse.ArgumentParser(description="Migrate a file-system catalog store into a database catalog store.")
    parser.add_argument(
        "--source-path",
        required=True,
        help="Path to the file-system catalog directory (e.g. ./example).",
    )
    parser.add_argument(
        "--dest-connection-string",
        required=True,
        help="SQLAlchemy URI for the destination catalog DB (e.g. sqlite:///./catalog.db).",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="Do not overwrite table metadata/columns that already exist in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the migration without writing to the destination.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy engine echo for the destination DB (debugging).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source = _build_source(args)
    dest = _build_dest(args)
    stats = migrate_catalog(
        source,
        dest,
        update_existing=not args.no_update_existing,
        dry_run=args.dry_run,
    )
    print(
        "Migration {}finished: {} tables, {} failed, {} tables with SQL examples, {} selection examples".format(
            "(dry-run) " if args.dry_run else "",
            stats["tables"],
            stats["failed_tables"],
            stats["sql_example_tables"],
            stats["selection_examples"],
        )
    )
    return stats


if __name__ == "__main__":
    main()
