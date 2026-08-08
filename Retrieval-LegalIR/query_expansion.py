#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for Vietnamese legal query expansion.

The expansion is intended for BM25 only. Dense/embedding models should usually
keep the original user question, because stuffing keywords into a sentence can
move the semantic vector away from the natural query.
"""

import json
import re


SYSTEM_PROMPT = """You are a Vietnamese legal information retrieval query expansion module.

Your job is to rewrite and expand a user question into legal search terms for BM25 retrieval.

Rules:
- Use Vietnamese only.
- Preserve the original legal intent.
- Add only synonymous, abbreviated, or legally standard terms that could appear in Vietnamese legal documents.
- Do not answer the question.
- Do not invent facts.
- Do not add specific numbers, money amounts, deadlines, article numbers, clause numbers, document IDs, document titles, or law names unless they already appear in the input question.
- Prefer short legal terms and noun phrases, not full sentences.
- Expand common Vietnamese legal abbreviations such as CCCD, CMND, BHXH, BHYT, UBND, TAND, VKSND, GTGT, TNCN when present.
- Convert informal question phrases into legal wording:
  "bao nhiêu tiền" -> "mức thu", "lệ phí", "phí", "đồng", "không có"
  "bao lâu" -> "thời hạn", "thời gian", "ngày làm việc", "không quá"
  "giấy tờ gì" -> "hồ sơ", "văn bản", "bản chính", "bản sao", "tờ khai", "tài liệu"
  "có được không" -> "được phép", "không được", "bị cấm", "điều kiện"
  "bị phạt" -> "xử phạt", "mức phạt", "phạt tiền", "vi phạm hành chính"
- Output strict JSON only.
- The JSON schema is:
{
  "normalized_query": "string",
  "expansion_terms": ["string", "..."]
}
- expansion_terms must contain at most 12 terms.
- Each expansion term must contain at most 6 words.
"""


ABBREVIATIONS = {
    'cccd': 'căn cước công dân',
    'cmnd': 'chứng minh nhân dân',
    'bhxh': 'bảo hiểm xã hội',
    'bhyt': 'bảo hiểm y tế',
    'ubnd': 'ủy ban nhân dân',
    'tand': 'tòa án nhân dân',
    'vksnd': 'viện kiểm sát nhân dân',
    'gtgt': 'giá trị gia tăng',
    'tncn': 'thu nhập cá nhân',
    'atgt': 'an toàn giao thông',
}


PHRASE_EXPANSIONS = [
    (r'bao nhiêu tiền|tốn bao nhiêu|chi phí|mất bao nhiêu', [
        'mức thu', 'lệ phí', 'phí', 'đồng', 'không có'
    ]),
    (r'bao nhiêu ngày|bao lâu|trong khoảng thời gian nào', [
        'thời hạn', 'thời gian', 'ngày làm việc', 'không quá', 'trong ngày'
    ]),
    (r'giấy tờ gì|cần chuẩn bị những giấy tờ gì|hồ sơ gồm|hồ sơ.*cần', [
        'hồ sơ', 'văn bản', 'bản chính', 'bản sao', 'tờ khai', 'tài liệu'
    ]),
    (r'có được|được phép|có phải|có bắt buộc', [
        'được phép', 'không được', 'bị cấm', 'điều kiện', 'trách nhiệm'
    ]),
    (r'bị phạt|mức phạt|xử phạt|phạt tiền', [
        'xử phạt', 'mức phạt', 'phạt tiền', 'vi phạm hành chính'
    ]),
    (r'thủ tục', [
        'trình tự', 'cách thức thực hiện', 'hồ sơ', 'cơ quan giải quyết'
    ]),
]


def _tokens(text):
    return re.findall(r'[0-9A-Za-zÀ-ỹ]+', str(text).lower())


def _dedupe(items):
    out, seen = [], set()
    for item in items:
        item = re.sub(r'\s+', ' ', str(item)).strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def rule_expand(question):
    """Small deterministic fallback. Useful for ablation and JSON repair."""
    qlow = str(question).lower()
    terms = []
    for abbr, full in ABBREVIATIONS.items():
        if re.search(rf'\b{re.escape(abbr)}\b', qlow):
            terms.append(full)
    for pattern, extra in PHRASE_EXPANSIONS:
        if re.search(pattern, qlow, flags=re.I):
            terms.extend(extra)
    terms = _dedupe(terms)[:12]
    normalized = ' '.join([question] + terms[:4]).strip()
    return {'normalized_query': normalized, 'expansion_terms': terms}


def extract_json(text):
    """Parse strict JSON, or recover the first JSON object from model chatter."""
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError('No JSON object found in model output')


def filter_expansion(question, expansion, max_terms=12):
    """Remove hallucination-prone terms, especially newly invented numbers."""
    q = str(question)
    qlow = q.lower()
    q_numbers = set(re.findall(r'\d+', qlow))
    normalized = str(expansion.get('normalized_query') or q)
    raw_terms = expansion.get('expansion_terms') or []
    if not isinstance(raw_terms, list):
        raw_terms = []

    terms = []
    for term in raw_terms:
        term = re.sub(r'\s+', ' ', str(term)).strip()
        if not term:
            continue
        # Do not let the LLM answer the question by inventing concrete values.
        nums = set(re.findall(r'\d+', term.lower()))
        if nums and not nums <= q_numbers:
            continue
        if re.search(r'\bcontext_\d+\b', term, flags=re.I):
            continue
        if re.search(r'\b(điều|khoản)\s+\d+', term, flags=re.I) and not re.search(
                r'\b(điều|khoản)\s+\d+', qlow, flags=re.I):
            continue
        if len(_tokens(term)) > 6:
            continue
        terms.append(term)

    terms = _dedupe(terms)[:max_terms]
    if not terms:
        terms = rule_expand(q)['expansion_terms']
    return {'normalized_query': normalized, 'expansion_terms': terms}


def build_expanded_query(question, expansion, repeat_top=4):
    """Build a BM25-friendly query string from a cached expansion record."""
    if not expansion:
        return str(question)
    normalized = expansion.get('normalized_query') or ''
    terms = expansion.get('expansion_terms') or []
    terms = [str(t) for t in terms if str(t).strip()]
    repeated = terms[:max(0, repeat_top)]
    return ' '.join([str(question), normalized, ' '.join(terms), ' '.join(repeated)]).strip()


def load_expansions(path):
    if not path:
        return {}
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)

