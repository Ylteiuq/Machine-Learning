"""向量检索：接收已经生成的向量，暂不负责调用 embedding 服务。"""

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
