#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structure-anchored pseudo-relevance feedback for Task 2 LegalQA.

This is a model-free, conservative CSQE-inspired prototype, not the paper's
LLM relevance assessor. Expansion words come only from a retrieved article
heading; feedback is skipped when the first retrieval is too uncertain.
"""

import re

import numpy as np

from bm25 import tokenize


ARTICLE_RE = re.compile(r'Điều\s+\d{1,4}(?:\s*[-.]\s*\d{1,3})?\s*\.\s*([^\n]{3,160})',
                        re.IGNORECASE)
STOP = {
    'có', 'được', 'là', 'gì', 'như', 'thế', 'nào', 'theo', 'về', 'việc',
    'của', 'và', 'đối', 'với', 'trong', 'để', 'phải', 'thì', 'khi', 'tại',
    'bao', 'nhiêu', 'không', 'những', 'các', 'một', 'năm', 'điều', 'khoản',
    'quy', 'định', 'thực', 'hiện',
}


def _article_heading(chunk):
    body = chunk['text'].split('\n', 1)[-1][:350]
    match = ARTICLE_RE.search(body)
    if not match:
        return None
    heading = re.sub(r'\s+', ' ', match.group(1)).strip()
    # Long lines with semicolons/ellipsis are usually clause or form text, not
    # a clean Điều heading. Never expand from such ambiguous structure.
    if (len(tokenize(heading)) > 16 or ';' in heading or
            '...' in heading or '…' in heading):
        return None
    return heading


def _query_anchors(bm25, question):
    anchors = {}
    for token in set(tokenize(question)):
        index = bm25.vocab.get(token)
        if (index is not None and token not in STOP and
                not any(c.isdigit() for c in token) and
                bm25.idf[index] >= 2.0):
            anchors[token] = float(bm25.idf[index])
    return anchors


def _coverage(heading, anchors):
    if not anchors:
        return 0.0
    words = set(tokenize(heading))
    return sum(weight for word, weight in anchors.items() if word in words) / sum(anchors.values())


def _novel_phrase(bm25, heading, question):
    question_words = set(tokenize(question))
    words = tokenize(heading)
    spans, active = [], []
    for word in words + ['']:
        if (word and word not in question_words and word not in STOP and
                word.isalpha() and word in bm25.vocab):
            active.append(word)
        elif active:
            spans.append(active[:4])
            active = []
    phrases = []
    for span in spans:
        if len(span) < 2:
            continue
        rarity = sum(float(bm25.idf[bm25.vocab[word]]) for word in span)
        if rarity >= 5.0:
            phrases.append((rarity, ' '.join(span)))
    return max(phrases)[1] if phrases else None


def corpus_query(bm25, question, order, scores, feedback=12):
    """Return `(expanded query, source doc ID, source heading)`, or None.

    Required: repeated high-ranked evidence from the top document, improved
    rare-query-term coverage in another article heading, and a novel phrase
    present in that real heading. This cannot guarantee that the top document
    is correct; it only limits propagation of unverified first-stage evidence.
    """
    if len(order) < 3:
        return None
    ranked = np.argsort(-scores, kind='stable')[:feedback]
    source_doc = int(bm25.chunk_doc[int(order[ranked[0]])])
    source_positions = [position for position in ranked
                        if int(bm25.chunk_doc[int(order[position])]) == source_doc]
    if len(source_positions) < 3:
        return None

    anchors = _query_anchors(bm25, question)
    if len(anchors) < 2 or sum(anchors.values()) < 6.0:
        return None
    first_heading = _article_heading(bm25.read_chunk(int(order[ranked[0]])))
    first_coverage = _coverage(first_heading or '', anchors)
    best = None
    top_score = float(scores[ranked[0]])
    for position in source_positions:
        if top_score > 0 and float(scores[position]) < 0.85 * top_score:
            continue
        heading = _article_heading(bm25.read_chunk(int(order[position])))
        if not heading:
            continue
        coverage = _coverage(heading, anchors)
        if best is None or coverage > best[0]:
            best = (coverage, heading)
    if best is None or best[0] < 0.75 or best[0] < first_coverage + 0.15:
        return None

    phrase = _novel_phrase(bm25, best[1], question)
    if not phrase:
        return None
    return (question.rstrip(' ?.!,;:') + ' ' + phrase, source_doc, best[1])
