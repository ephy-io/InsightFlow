"""按业务关键词查找相关的数据库表与字段。"""

import os
import re
import time

import Levenshtein

from insightflow import config
from insightflow.catalog.retrival_helper import build_column_tables_mapping, build_columns_retriever
from insightflow.text_segmenter import _segmenter
from insightflow.utils import log

# 文档构建时跳过构建
if not os.environ.get("SPHINX_BUILD"):
    try:
        _catalog_store = config.get().catalog_store
    except ValueError:
        _catalog_store = None
else:
    _catalog_store = None

if _catalog_store:
    bm25, vector_db, columns, col_dict = build_columns_retriever(_catalog_store, config.get().vector_db_path)
    column_tables_mapping = build_column_tables_mapping(_catalog_store)
else:
    bm25, vector_db, columns, col_dict = None, None, [], {}
    column_tables_mapping = {}


def column_retrieval(query, db, k=10, threshold=0.5, filter=None):
    """通过向量相似度召回相关字段。

    Args:
        query (str): 检索文本。
        db: 用于查询的向量库。
        k (int, optional): 召回条数上限，默认 10。
        threshold (float, optional): 筛选结果的阈值，默认 0.5。
        filter (dict, optional): 可选的元数据过滤条件，默认 None。

    Returns:
        list: 相关字段的 Schema 标识。
    """
    log(f"检索 Schema 字段，问题：{query}")
    similar_column_key_scores = db.similarity_search_with_score(query, k=k, filter=filter)
    # log(f"similar_column_key_scores: {similar_column_key_scores}")
    schema_ids = [key.metadata.get("schema_id", key.metadata["column_name"]) for (key, score) in similar_column_key_scores if score < threshold]
    log(f"过滤后的候选字段：{schema_ids}")
    return schema_ids


def merge_list(list1, list2):
    """按首次出现顺序合并候选，避免 set 造成结果抖动。"""
    return list(dict.fromkeys([*list1, *list2]))


def edit_distance_score(key1, key2):
    """计算两个字符串的归一化编辑距离。

    Returns:
        float: 相同字符串为 0，完全不同时为 1。
    """
    dist = Levenshtein.distance(key1, key2)
    max_len = max(len(key1), len(key2))
    return dist / max_len if max_len > 0 else 1


def edit_distance_search(keywords_list, top_k=10, threshold=0.5):
    """通过归一化编辑距离匹配字段。

    Args:
        keywords_list (list): 用于匹配的关键词。
        top_k (int, optional): 每个关键词最多保留的候选数，默认 10。
        threshold (float, optional): 编辑距离上限，默认 0.5。

    Returns:
        list: 匹配到的字段标识。
    """
    keys = {re.sub(r"(_id|_name| id| name)$", "", key.lower()) for key in keywords_list}
    column_similarity_score: dict[str, float] = {}
    for key in sorted(keys):
        key_column_similarity_score = {}
        for column_name, row in col_dict.items():
            column_name_score = edit_distance_score(
                key, re.sub(r"(_id|_name| id| name)$", "", row.get("column_name", "").lower())
            )
            display_score = edit_distance_score(
                key, re.sub(r"(_id|_name| id| name)$", "", row.get("display_name", "").lower())
            )
            if column_name_score < threshold or display_score < threshold:
                key_column_similarity_score[column_name] = min(column_name_score, display_score)
        # 编辑距离越小越相似；稳定排序可让相同分数按 Schema 标识决定。
        for schema_id, score in sorted(key_column_similarity_score.items(), key=lambda item: (item[1], item[0]))[:top_k]:
            column_similarity_score[schema_id] = min(column_similarity_score.get(schema_id, 1.0), score)
    return [schema_id for schema_id, _ in sorted(column_similarity_score.items(), key=lambda item: (item[1], item[0]))]


def bm25_search(query_list, top_k=5, score_threshold=0.5):
    """使用 BM25 对字段文档进行关键词检索。

    Args:
        query_list (list): 查询词列表。
        top_k (int, optional): 最多返回的候选数，默认 5。
        score_threshold (float, optional): BM25 分数下限，默认 0.5。

    Returns:
        list: 匹配到的字段标识。
    """
    query_tokens = [token for token in _segmenter.cut(" ".join(query_list)) if token not in ("_", " ")]
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score_threshold and score < score_threshold:
            continue
        results.append(columns[idx].get("schema_id", columns[idx]["column_name"]))
    return results


def get_relevant_columns(keywords_list, dimensions, metrics):
    """为关键词、维度和指标召回相关字段。

    根据检索配置选择 BM25、编辑距离、向量相似度及重排序。

    Args:
        keywords_list (list): 通用关键词。
        dimensions (list): 维度相关的关键词。
        metrics (list): 指标相关的关键词。

    Returns:
        list: 相关字段的 Schema 标识。
    """
    started = time.perf_counter()
    query_terms = list(dict.fromkeys([*keywords_list, *dimensions, *metrics]))
    if not query_terms or bm25 is None:
        return []

    # Baseline 保留历史召回入口，便于和后续混合策略做同集对比。
    try:
        strategy = getattr(config.get(), "schema_retrieval_strategy", "rrf")
    except ValueError:
        strategy = "baseline"
    bm25_results = bm25_search(query_terms, top_k=max(len(query_terms) * 4, 10))
    if strategy == "baseline":
        results = merge_list(bm25_results, edit_distance_search(query_terms, top_k=len(query_terms), threshold=0.3))
        if vector_db is not None and dimensions:
            results = merge_list(results, column_retrieval(" ".join(dimensions), vector_db, k=10, filter={"category": "dimension"}))
        if vector_db is not None and metrics:
            results = merge_list(results, column_retrieval(" ".join(metrics), vector_db, k=10, threshold=0.55, filter={"category": "metric"}))
        log(f"Schema retrieval strategy=baseline candidates={len(results)} latency_ms={(time.perf_counter()-started)*1000:.2f}")
        return results

    dense_results = []
    if vector_db is not None:
        dense_results = column_retrieval(" ".join(query_terms), vector_db, k=30, threshold=2.0)
    if strategy == "bm25_dense":
        results = merge_list(bm25_results, dense_results)
    else:
        # RRF 不依赖不同检索器的原始分数尺度，只融合各自的名次。
        ranks: dict[str, float] = {}
        for ranking in (bm25_results, dense_results):
            for rank, item in enumerate(ranking, start=1):
                ranks[item] = ranks.get(item, 0.0) + 1.0 / (60 + rank)
        results = sorted(ranks, key=lambda item: (-ranks[item], item))

    if strategy == "rrf_reranker":
        try:
            from insightflow.catalog.reranker import rerank_schema_candidates

            results = rerank_schema_candidates(" ".join(query_terms), results, col_dict)
        except Exception as exc:  # reranker 故障时保留可用的 RRF 结果，并记录原因。
            log(f"Schema reranker failed; using RRF order: {exc}")
    log(
        f"Schema retrieval strategy={strategy} candidates={len(results)} "
        f"rrf_scores={dict(list(sorted(ranks.items(), key=lambda item: (-item[1], item[0]))[:10])) if strategy != 'bm25_dense' else {}} "
        f"latency_ms={(time.perf_counter()-started)*1000:.2f}"
    )
    return results
