"""统一检索入口：关键词检索、向量检索与排名融合。"""

from math import sqrt

import notes_sqlite


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """余弦相似度 = 点积 / 两个向量的长度乘积，分数越大越相似。"""
    if len(left) != len(right):
        raise ValueError("两个向量的维度必须相同")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("不能用零向量计算余弦相似度")
    return dot_product / (left_norm * right_norm)


def search_notes_by_vector(query_vector: list[float], model: str,
                           top_k: int = 5) -> list[dict]:
    """对已索引的笔记排序；问题向量也必须由 model 对应的模型生成。"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    notes = notes_sqlite.list_note_embeddings(model)
    results = []
    for note in notes:
        score = cosine_similarity(query_vector, note["vector"])
        results.append({"id": note["id"], "title": note["title"], "score": score})

    # 只将编号、标题和分数交给 Agent；正文仍通过 read_notes 获取。
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def merge_search_results(keyword_results: list[dict], vector_results: list[dict],
                         top_k: int = 5) -> list[dict]:
    """用 RRF 合并两份已排序的结果；融合分数越大越靠前。"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    merged = {}
    for results in (keyword_results, vector_results):
        for rank, note in enumerate(results, start=1):
            note_id = note["id"]
            if note_id not in merged:
                merged[note_id] = {
                    "id": note_id, "title": note["title"], "score": 0.0,
                }
            # 两路原始分数的尺度不同，只使用排名贡献，不直接相加。
            merged[note_id]["score"] += 1 / (60 + rank)

    results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def search_notes(query: str, top_k: int = 5, *,
                 query_vector: list[float] = None, model: str = None) -> list[dict]:
    """没有问题向量时沿用 BM25；提供同一模型的问题向量时融合两路结果。"""
    if query_vector is None:
        return notes_sqlite.search_notes(query, top_k)

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    if not model:
        raise ValueError("提供问题向量时需要指定 embedding 模型")

    # 两路先各取更多候选，融合排序后再截取最终数量。
    candidate_count = top_k * 3
    keyword_results = notes_sqlite.search_notes(query, candidate_count)
    vector_results = search_notes_by_vector(query_vector, model, candidate_count)
    return merge_search_results(keyword_results, vector_results, top_k)
