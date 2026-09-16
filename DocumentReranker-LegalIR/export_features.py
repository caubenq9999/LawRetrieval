#!/usr/bin/env python3
"""Export one inspectable JSONL row per (query, candidate document)."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RETRIEVAL = os.path.join(ROOT, 'Retrieval-LegalIR')
sys.path.insert(0, RETRIEVAL)

from hybrid import Hybrid, minmax  # noqa: E402
from rerank import Reranker, _agg  # noqa: E402

from common import FEATURE_NAMES, load_json, refuse_overwrite, write_meta  # noqa: E402


def _select_qids(questions, split_path=None, split_name=None):
    if not split_path:
        return list(questions), None
    split = load_json(split_path)
    if split_name not in split:
        raise KeyError(f'{split_name!r} khong co trong {split_path}; co {list(split)}')
    qids = [str(q) for q in split[split_name] if str(q) in questions]
    return qids, split


def _prior_counts(prior_train, split, split_name):
    if not prior_train:
        return collections.Counter(), set()
    if split:
        if 'train' in split:
            source_qids = {str(q) for q in split['train']}
        elif 'val' in split:
            source_qids = set(prior_train) - {str(q) for q in split['val']}
        else:
            raise ValueError('split can co key train hoac val de tranh leakage cua prior')
    else:
        source_qids = set(prior_train)
    freq = collections.Counter(
        str(doc) for qid, item in prior_train.items() if str(qid) in source_qids
        for doc in item.get('answer', []))
    return freq, source_qids


def _popularity(freq, max_log, doc_id, own_gold, leave_one_out):
    count = freq.get(str(doc_id), 0)
    if leave_one_out and str(doc_id) in own_gold:
        count = max(0, count - 1)
    return math.log1p(count) / max_log if max_log else 0.0


def _components(h, query, pool, alpha, qvec):
    bm_all, _ = h.bm25.score(query)
    nz = np.flatnonzero(bm_all)
    if not len(nz):
        empty = np.empty(0, dtype=np.float32)
        return np.empty(0, dtype=np.int64), empty, empty, empty
    if len(nz) > pool:
        nz = nz[np.argpartition(-bm_all[nz], pool - 1)[:pool]]
    order = nz[np.argsort(-bm_all[nz])]
    bm_raw = bm_all[order].astype(np.float32)
    srt = np.argsort(order)
    vecs = np.empty((len(order), h.dim), dtype=np.float32)
    vecs[srt] = h.emb[order[srt]].astype(np.float32)
    dense_raw = vecs @ qvec
    bm = minmax(bm_raw)
    dense = minmax(dense_raw)
    hybrid = alpha * dense + (1.0 - alpha) * bm
    return order, bm, dense, hybrid


def _candidate_chunks(h, order, hybrid, ndocs, mchunks):
    per_doc = collections.OrderedDict()
    hmap = {}
    for j in np.argsort(-hybrid, kind='stable'):
        ci = int(order[j])
        doc = int(h.bm25.chunk_doc[ci])
        if doc not in per_doc and len(per_doc) >= ndocs:
            break
        per_doc.setdefault(doc, []).append(ci)
        hmap[ci] = float(hybrid[j])
    return per_doc, hmap


def _top_values(values, missing_second='repeat'):
    values = sorted((float(x) for x in values), reverse=True)
    first = values[0]
    if len(values) > 1:
        second = values[1]
    else:
        second = first if missing_second == 'repeat' else 0.0
    return first, second, float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', required=True, help='train.json hoac public-official.json')
    p.add_argument('--output', '-o', required=True, help='JSONL feature output')
    p.add_argument('--split', help='split.json; tuy chon')
    p.add_argument('--split-name', help='train/val trong split.json')
    p.add_argument('--prior-train', help='train.json dung tinh popularity; bo qua = popularity 0')
    p.add_argument('--index', default=os.path.join(RETRIEVAL, 'index'))
    p.add_argument('--emb', default=os.path.join(RETRIEVAL, 'emb_v2ft'))
    p.add_argument('--reranker', default=os.path.join(ROOT, 'Finetune-LegalIR',
                                                      'models', 'reranker-ft'))
    p.add_argument('--pool', type=int, default=2000)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--agg', default='max+0.4')
    p.add_argument('--prior-lambda', type=float, default=0.2,
                   help='Popularity weight cua baseline hien tai')
    p.add_argument('--ndocs', type=int, default=50)
    p.add_argument('--mchunks', type=int, default=2)
    p.add_argument('--k1', type=float, default=2.5)
    p.add_argument('--b', type=float, default=0.9)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--encode-batch', type=int, default=64)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if bool(args.split) != bool(args.split_name):
        p.error('--split va --split-name phai di cung nhau')
    refuse_overwrite(args.output, args.overwrite)
    refuse_overwrite(args.output + '.meta.json', args.overwrite)

    questions = {str(k): v for k, v in load_json(args.questions).items()}
    qids, split = _select_qids(questions, args.split, args.split_name)
    if args.limit:
        qids = qids[:args.limit]
    prior_train = ({str(k): v for k, v in load_json(args.prior_train).items()}
                   if args.prior_train else None)
    freq, prior_source_qids = _prior_counts(prior_train, split, args.split_name)
    max_log = math.log1p(max(freq.values())) if freq else 0.0

    print(f'Nap hybrid: index={args.index} | emb={args.emb}')
    h = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
    print(f'Nap reranker: {args.reranker} | max_length={args.max_length}')
    rr = Reranker(args.reranker, batch=args.batch, max_length=args.max_length)

    texts = [questions[q]['question'] for q in qids]
    print(f'Encode {len(texts)} query ...')
    qvecs = []
    for start in range(0, len(texts), args.encode_batch):
        qvecs.append(h.encode_query(texts[start:start + args.encode_batch]))
    qvecs = np.concatenate(qvecs) if qvecs else np.empty((0, h.dim), dtype=np.float32)

    n_rows = n_no_cand = 0
    candidate_recalls = []
    t0 = time.time()
    with open(args.output, 'w', encoding='utf-8') as out:
        for qi, qid in enumerate(qids):
            item = questions[qid]
            query = item['question']
            gold = {str(x) for x in item.get('answer', [])}
            order, bm, dense, hybrid = _components(
                h, query, args.pool, args.alpha, qvecs[qi])
            if not len(order):
                n_no_cand += 1
                continue
            per_doc, hmap = _candidate_chunks(
                h, order, hybrid, args.ndocs, args.mchunks)
            lookup = {int(ci): j for j, ci in enumerate(order)}
            cand = [(doc, ci) for doc, chunks in per_doc.items()
                    for ci in chunks[:args.mchunks]]
            chunk_texts = [h.bm25.read_chunk(ci)['text'] for _, ci in cand]
            ce_raw = rr.score(query, chunk_texts)
            ce_norm = minmax(ce_raw)
            hybrid_cand = np.asarray([hmap[ci] for _, ci in cand], dtype=np.float32)
            blended = args.beta * ce_norm + (1.0 - args.beta) * minmax(hybrid_cand)

            by_doc = collections.OrderedDict()
            for idx, (doc, ci) in enumerate(cand):
                j = lookup[ci]
                by_doc.setdefault(doc, []).append({
                    'ci': ci,
                    'hybrid': float(hybrid[j]),
                    'bm25': float(bm[j]),
                    'dense': float(dense[j]),
                    'ce': float(ce_raw[idx]),
                    'blend': float(blended[idx]),
                })
            doc_rank = {doc: rank + 1 for rank, doc in enumerate(by_doc)}
            if gold:
                candidate_recalls.append(len(gold & {str(d) for d in by_doc}) / len(gold))

            leave_one_out = qid in prior_source_qids
            query_rows = []
            raw_baseline = []
            for doc, chunks in by_doc.items():
                hy1, hy2, hymean = _top_values(x['hybrid'] for x in chunks)
                bm1, bm2, _ = _top_values(x['bm25'] for x in chunks)
                dn1, dn2, _ = _top_values(x['dense'] for x in chunks)
                ce1, ce2, cemean = _top_values(x['ce'] for x in chunks)
                blend_values = sorted((x['blend'] for x in chunks), reverse=True)
                popularity = _popularity(freq, max_log, doc, gold, leave_one_out)
                raw_score = float(_agg(blend_values, args.agg))
                features = {
                    'hybrid_rank_inv': 1.0 / doc_rank[doc],
                    'hybrid_max': hy1,
                    'hybrid_second': hy2,
                    'hybrid_mean': hymean,
                    'bm25_max': bm1,
                    'bm25_second': bm2,
                    'dense_max': dn1,
                    'dense_second': dn2,
                    'ce_max': ce1,
                    'ce_second': ce2,
                    'ce_gap': ce1 - ce2 if len(chunks) > 1 else 0.0,
                    'ce_mean': cemean,
                    # Gan score that sau khi min-max o muc document, ben duoi.
                    'baseline_score': raw_score,
                    'candidate_chunk_count': float(len(chunks)),
                    'corpus_chunk_count_log': math.log1p(int(h.bm25.doc_nchunks[doc])),
                    'popularity': popularity,
                }
                assert list(features) == FEATURE_NAMES
                row = {
                    'qid': qid,
                    'doc_id': str(doc),
                    'label': int(str(doc) in gold) if gold else None,
                    'gold_count': len(gold),
                    'features': features,
                }
                query_rows.append(row)
                raw_baseline.append(raw_score)
            # prior.py chuan hoa diem document trong top-50 truoc khi cong prior.
            normalized = minmax(np.asarray(raw_baseline, dtype=np.float32))
            for row, score in zip(query_rows, normalized):
                row['features']['baseline_score'] = (
                    float(score) + args.prior_lambda * row['features']['popularity'])
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                n_rows += 1
            if (qi + 1) % 50 == 0:
                elapsed = time.time() - t0
                left = elapsed / (qi + 1) * (len(qids) - qi - 1) / 60
                print(f'  {qi + 1}/{len(qids)} | {n_rows} rows | con ~{left:.0f} phut',
                      flush=True)

    meta = {
        'format_version': 1,
        'questions': os.path.abspath(args.questions),
        'split': os.path.abspath(args.split) if args.split else None,
        'split_name': args.split_name,
        'n_queries_requested': len(qids),
        'n_queries_without_candidates': n_no_cand,
        'n_rows': n_rows,
        'feature_names': FEATURE_NAMES,
        'candidate_recall': float(np.mean(candidate_recalls)) if candidate_recalls else None,
        'config': {k: getattr(args, k) for k in (
            'pool', 'alpha', 'beta', 'agg', 'prior_lambda', 'ndocs', 'mchunks',
            'k1', 'b', 'max_length')},
    }
    write_meta(args.output, meta)
    print(f'Xong: {n_rows} rows -> {args.output}')
    if candidate_recalls:
        print(f'Candidate Recall@{args.ndocs}: {np.mean(candidate_recalls):.4f}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
