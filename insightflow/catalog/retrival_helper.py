"""构建字段检索索引的辅助函数。"""

import hashlib
import json

from rank_bm25 import BM25Okapi

from insightflow.llm.llm import get_embedding_model
from insightflow.text_segmenter import _segmenter
from insightflow.utils import create_vector_db, log


def get_columns_metadata(catalog):
    """提取字段级索引所需的元数据。

    Args:
        catalog: Schema 目录存储实例。

    Returns:
        tuple: 字段列表、按 Schema 标识索引的元数据、分词列表和向量化文本，依次为 (columns, col_dict, column_tokens, embedding_keys)。
    """
    columns = []
    col_dict = {}
    column_tokens = []
    embedding_keys = []
    for table_name in catalog.get_table_list():
        for raw_column in catalog.get_column_list(table_name):
            # 同一个完整标识写入 BM25 文档、向量元数据和详情字典，避免同名字段串表。
            schema_id = f"{table_name}.{raw_column['column_name']}"
            column = {**raw_column, "table_name": table_name, "schema_id": schema_id}
            columns.append(column)
            col_dict[schema_id] = column
            text_parts = [
                column.get("column_name", ""),
                column.get("display_name", ""),
                column.get("alias", ""),
                column.get("tag", ""),
                column.get("description", ""),
                table_name,
            ]
            text = " ".join(text_parts)
            tokens = [token for token in _segmenter.cut(text) if token not in ("_", " ")]
            column_tokens.append(tokens)
            embedding_key = (
                f"{schema_id}: {column.get('display_name', '')} "
                f"{column.get('alias', '')} {column.get('description', '')}"
            )
            embedding_keys.append(embedding_key)
    return columns, col_dict, column_tokens, embedding_keys


def build_column_tables_mapping(catalog):
    """建立完整字段标识到所属表的映射。"""
    column_tables_mapping: dict[str, list[str]] = {}
    for table_name in catalog.get_table_list():
        for column in catalog.get_column_list(table_name):
            column_name = column["column_name"]
            column_tables_mapping[f"{table_name}.{column_name}"] = [table_name]
    return column_tables_mapping


def build_columns_retriever(catalog, vector_db_path: str | None = None):
    """构建字段级 BM25 和向量检索器。

    Args:
        catalog: Schema 目录存储实例。
        vector_db_path: 向量库的持久化目录。

    Returns:
        tuple: (bm25, vector_db, columns, col_dict)。
    """
    columns, col_dict, column_tokens, embedding_keys = get_columns_metadata(catalog)

    bm25 = BM25Okapi(column_tokens)

    log("Building vector database for columns...")
    vector_db = create_vector_db(
        embedding_keys,
        get_embedding_model(),
        metadatas=columns,
        collection_name="columns",
        collection_metadata={
            "hnsw:space": "cosine",
            "schema_sha256": hashlib.sha256(
                json.dumps(embedding_keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        chroma_db_path=vector_db_path,
        ids=[column["schema_id"] for column in columns],
    )

    return bm25, vector_db, columns, col_dict
