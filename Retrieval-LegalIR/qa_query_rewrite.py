#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative, model-free query rewrites for Task 2 legal retrieval.

The rewrite is only a second search query. It is never treated as legal evidence
or copied into an answer; the original question remains the primary query.
"""

import re

import numpy as np


_RULES = (
    (r'\b(?:bị xử phạt thế nào|bị phạt thế nào)\b',
     'mức xử phạt đối với hành vi'),
    (r'\b(?:bị xử phạt bao nhiêu năm tù|bị phạt bao nhiêu năm tù)\b',
     'hình phạt tù'),
    (r'\b(?:được hưởng án treo|có được hưởng án treo)\b',
     'điều kiện cho hưởng án treo'),
    (r'\b(?:cần đáp ứng những (?:điều kiện|yêu cầu) gì|cần đáp ứng yêu cầu gì)\b',
     'điều kiện tiêu chuẩn'),
    (r'\b(?:trình tự,? thủ tục)\b', 'trình tự thủ tục hồ sơ'),
    (r'\b(?:có được|có thuộc trường hợp được)\b', 'trường hợp được'),
    (r'\b(?:có nộp|phải nộp)\b', 'nghĩa vụ nộp'),
    (r'\b(?:được miễn)\b', 'miễn'),
)

_QUESTION_END = re.compile(
    r'\s*(?:là gì|gồm những gì|như thế nào|thế nào|bao lâu|bao nhiêu|gì|'
    r'đúng không|hay không|không)\s*[?.!]*\s*$', re.IGNORECASE)
_LEAD = re.compile(
    r'^\s*(?:hướng dẫn(?: về việc)?|quy định(?: về)?|xin|việc)\s+',
    re.IGNORECASE)


def rewrite_query(question):
    """Return one legal-register variant, or None when no useful change exists.

    This deliberately does not introduce law numbers, dates, amounts, or facts.
    The optional rules only change the question's retrieval vocabulary.
    """
    original = re.sub(r'\s+', ' ', question).strip()
    if not original:
        return None
    rewritten = _LEAD.sub('', original)
    for pattern, replacement in _RULES:
        if replacement is not None:
            rewritten = re.sub(pattern, replacement, rewritten,
                               flags=re.IGNORECASE)
    rewritten = _QUESTION_END.sub('', rewritten)
    rewritten = re.sub(r'\s+(?:là|thì)\s*$', '', rewritten,
                       flags=re.IGNORECASE)
    rewritten = rewritten.strip(' ?.!,;:')
    if not rewritten or rewritten.casefold() == original.strip(' ?.!,;:').casefold():
        return None
    return rewritten


def blend_search_results(original, rewritten, rewrite_weight=0.25):
    """Blend two `(chunk IDs, scores)` results, protecting original ranking.

    Each branch is min-max normalized within its retrieval pool. A rewritten-only
    hit cannot beat the original branch's top hit with the default weight.
    """
    original_ids, original_scores = original
    rewritten_ids, rewritten_scores = rewritten
    if rewritten_ids.size == 0:
        return original

    def normalized(scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        low, high = float(scores.min()), float(scores.max())
        if high - low < 1e-9:
            return np.ones_like(scores)
        return (scores - low) / (high - low)

    fused = {}
    for chunk_id, score in zip(original_ids, normalized(original_scores)):
        fused[int(chunk_id)] = float(score)
    for chunk_id, score in zip(rewritten_ids, normalized(rewritten_scores)):
        fused[int(chunk_id)] = fused.get(int(chunk_id), 0.0) + rewrite_weight * float(score)
    ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
    return (np.asarray(ranked, dtype=np.int64),
            np.asarray([fused[chunk_id] for chunk_id in ranked], dtype=np.float32))
