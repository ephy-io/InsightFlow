"""阿里云百炼 Reranker 适配器。"""

from __future__ import annotations

import os
from typing import Any

import requests

from insightflow import config
from insightflow.utils import log


def rerank_schema_candidates(query: str, candidate_ids: list[str], col_dict: dict[str, dict[str, Any]]) -> list[str]:
    """调用百炼 Reranker 重排候选 Schema；配置或服务不可用时显式抛错供调用方降级。"""
    if not candidate_ids:
        return []
    cfg = getattr(config.get(), "reranker_config", {}) or {}
    endpoint = cfg.get("endpoint") or os.getenv("DASHSCOPE_RERANK_URL")
    api_key = cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("DASHSCOPE_RERANK_MODEL") or cfg.get("model") or "qwen3-rerank"
    if not endpoint or not api_key or "${" in str(endpoint) or "${" in str(api_key):
        raise RuntimeError("未配置 DASHSCOPE_RERANK_URL 或 DASHSCOPE_API_KEY")

    documents = []
    for candidate_id in candidate_ids:
        column = col_dict.get(candidate_id, {})
        documents.append(
            " | ".join(
                str(column.get(key, ""))
                for key in ("table_name", "column_name", "display_name", "alias", "description", "category")
            )
        )
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": documents, "top_n": len(documents)},
        timeout=float(cfg.get("timeout", 15)),
    )
    response.raise_for_status()
    payload = response.json()
    ranked = payload.get("results")
    if not isinstance(ranked, list):
        raise ValueError("Reranker 响应缺少 results 数组")
    ordered: list[str] = []
    scored_candidates: list[tuple[str, float]] = []
    for item in ranked:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(candidate_ids):
            ordered.append(candidate_ids[index])
            score = item.get("relevance_score")
            if isinstance(score, int | float):
                scored_candidates.append((candidate_ids[index], float(score)))
    # 保留服务响应中未覆盖的合法候选，避免意外丢字段。
    ordered.extend(item for item in candidate_ids if item not in ordered)
    log(
        f"Qwen reranker returned {len(ranked)} scores for {len(candidate_ids)} candidates; "
        f"top_scores={scored_candidates[:10]}"
    )
    return ordered
