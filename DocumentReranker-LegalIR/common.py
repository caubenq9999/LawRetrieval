#!/usr/bin/env python3
"""Shared I/O and evaluation helpers for the document-ranker experiment."""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np


FEATURE_NAMES = [
    'hybrid_rank_inv',
    'hybrid_max',
    'hybrid_second',
    'hybrid_mean',
    'bm25_max',
    'bm25_second',
    'dense_max',
    'dense_second',
    'ce_max',
    'ce_second',
    'ce_gap',
    'ce_mean',
    'baseline_score',
    'candidate_chunk_count',
    'corpus_chunk_count_log',
    'popularity',
]


def load_json(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def refuse_overwrite(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f'{path} da ton tai. Dung --overwrite neu thuc su muon ghi de.')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)


def meta_path(feature_path):
    return feature_path + '.meta.json'


def write_meta(feature_path, meta):
    with open(meta_path(feature_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_rows(path, require_labels=False):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [x for x in FEATURE_NAMES if x not in row.get('features', {})]
            if missing:
                raise ValueError(f'{path}:{line_no}: thieu feature {missing}')
            if require_labels and row.get('label') is None:
                raise ValueError(f'{path}:{line_no}: thieu label')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: khong co dong du lieu nao')
    return rows


def group_rows(rows):
    """Return ordered query groups and reject interleaved qids."""
    groups = OrderedDict()
    closed = set()
    previous = None
    for row in rows:
        qid = str(row['qid'])
        if previous is not None and qid != previous:
            closed.add(previous)
        if qid in closed:
            raise ValueError(f'qid {qid} xuat hien o nhieu doan khong lien tiep')
        groups.setdefault(qid, []).append(row)
        previous = qid
    return groups


def matrix(rows):
    return np.asarray(
        [[float(row['features'][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )


def labels(rows):
    return np.asarray([int(row['label']) for row in rows], dtype=np.int32)


def group_sizes(rows):
    return np.asarray([len(v) for v in group_rows(rows).values()], dtype=np.int32)


def score_groups(rows, scores, k=5):
    """Per-query Recall@k plus ranked document ids.

    gold_count is stored explicitly because a gold document outside the candidate
    set still counts in the denominator and cannot be recovered by the ranker.
    """
    if len(rows) != len(scores):
        raise ValueError('So row va so score khong khop')
    out = OrderedDict()
    offset = 0
    for qid, items in group_rows(rows).items():
        part = np.asarray(scores[offset:offset + len(items)], dtype=np.float64)
        order = np.argsort(-part, kind='stable')
        chosen = [items[int(i)] for i in order[:k]]
        gold_count = int(items[0].get('gold_count') or 0)
        hits = sum(int(x.get('label') or 0) for x in chosen)
        recall = hits / gold_count if gold_count else None
        out[qid] = {
            'recall': recall,
            'hits': hits,
            'gold_count': gold_count,
            'docs': [str(x['doc_id']) for x in chosen],
        }
        offset += len(items)
    return out


def mean_recall(result):
    values = [v['recall'] for v in result.values() if v['recall'] is not None]
    return float(np.mean(values)) if values else None


def compare_results(base, candidate, bootstrap=4000, seed=2026):
    qids = [q for q in base if q in candidate and base[q]['recall'] is not None]
    before = np.asarray([base[q]['recall'] for q in qids], dtype=np.float64)
    after = np.asarray([candidate[q]['recall'] for q in qids], dtype=np.float64)
    delta = after - before
    report = {
        'n_queries': len(qids),
        'baseline_recall_at_5': float(before.mean()),
        'candidate_recall_at_5': float(after.mean()),
        'delta': float(delta.mean()),
        'improved_queries': int((delta > 0).sum()),
        'damaged_queries': int((delta < 0).sum()),
        'unchanged_queries': int((delta == 0).sum()),
    }
    if len(delta) and bootstrap:
        rng = np.random.default_rng(seed)
        chunk = 250
        means = []
        for start in range(0, bootstrap, chunk):
            n = min(chunk, bootstrap - start)
            idx = rng.integers(0, len(delta), size=(n, len(delta)))
            means.append(delta[idx].mean(axis=1))
        boot = np.concatenate(means)
        report['bootstrap_95_ci'] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    return report


def baseline_scores(rows):
    return np.asarray([float(r['features']['baseline_score']) for r in rows],
                      dtype=np.float32)

