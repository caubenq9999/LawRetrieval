#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine Task-2-only pseudo labels for cross-encoder fine-tuning.

The candidates match qa_predict.py: hybrid linear retrieval, top documents,
then at most N chunks per document. The official Task 2 training answer acts
only as a teacher for selecting the best candidate chunk. It is never an input
to the reranker at inference time.

Keep a disjoint validation prefix, for example:
    python mine_qa_reranker.py --offset 1000 --limit 2000 ...

This script intentionally cannot read Task 1 train pairs or checkpoints.
"""

import argparse
from collections import Counter
import json
import os
import re
import sys
import time

import numpy as np


TOKEN_RE = re.compile(r'\w+', re.UNICODE)


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def weighted_overlap(reference, passage, vocab, idf):
    """Return answer coverage and a coverage-heavy weighted F1 score."""
    ref = Counter(tokens(reference))
    cand = Counter(tokens(passage))
    if not ref or not cand:
        return 0.0, 0.0

    def weight(term):
        index = vocab.get(term)
        return float(idf[index]) if index is not None else 1.0

    overlap = sum(min(count, cand.get(term, 0)) * weight(term)
                  for term, count in ref.items())
    ref_weight = sum(count * weight(term) for term, count in ref.items())
    cand_weight = sum(count * weight(term) for term, count in cand.items())
    recall = overlap / max(ref_weight, 1e-9)
    precision = overlap / max(cand_weight, 1e-9)
    f1 = (2 * recall * precision / (recall + precision)
          if recall + precision else 0.0)
    return recall, 0.7 * recall + 0.3 * f1


def candidate_pool(retriever, question, qvec, pool, alpha, ndocs, mchunks):
    bm25 = retriever.bm25
    order, fused = retriever.fused_chunk_scores(
        question, pool=pool, fusion='linear', alpha=alpha, qvec=qvec)
    if order.size == 0:
        return []

    per_doc = {}
    first_stage = {}
    for position in np.argsort(-fused):
        chunk_index = int(order[position])
        doc_id = int(bm25.chunk_doc[chunk_index])
        if doc_id not in per_doc:
            if len(per_doc) >= ndocs:
                break
            per_doc[doc_id] = []
        if len(per_doc[doc_id]) < mchunks:
            per_doc[doc_id].append(chunk_index)
            first_stage[chunk_index] = float(fused[position])

    return [(doc_id, chunk_index, first_stage[chunk_index])
            for doc_id, chunk_indices in per_doc.items()
            for chunk_index in chunk_indices]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Mine Task 2 reranker pairs from Task 2 train answers only.')
    parser.add_argument('--qa-dir', required=True,
                        help='Directory containing train.json and selected-contexts/')
    parser.add_argument('--index', required=True, help='Task 2 BM25 index')
    parser.add_argument('--emb', required=True, help='Task 2 dense embeddings')
    parser.add_argument('--out', required=True, help='Output JSONL pair file')
    parser.add_argument('--offset', type=int, default=1000,
                        help='First 1000 are held out for validation by default')
    parser.add_argument('--limit', type=int, default=2000,
                        help='Questions to mine; 0 means all after offset')
    parser.add_argument('--pool', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--ndocs', type=int, default=20)
    parser.add_argument('--mchunks', type=int, default=3)
    parser.add_argument('--neg', type=int, default=4)
    parser.add_argument('--min-positive', type=float, default=0.08,
                        help='Minimum weighted teacher score for a usable positive')
    parser.add_argument('--negative-ratio', type=float, default=0.55,
                        help='Negative score must be <= positive * this ratio')
    parser.add_argument('--qbatch', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = os.path.join(args.qa_dir, 'train.json')
    if not os.path.isfile(train_path):
        raise SystemExit(f'Task 2 train file not found: {train_path}')

    retrieval_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'Retrieval-LegalIR'))
    sys.path.insert(0, retrieval_dir)
    from hybrid import Hybrid

    with open(train_path, encoding='utf-8-sig') as stream:
        all_items = list(json.load(stream).items())
    end = None if args.limit == 0 else args.offset + args.limit
    items = all_items[args.offset:end]
    if not items:
        raise SystemExit('No training questions selected; check --offset/--limit.')

    print(f'Loading Task 2 hybrid retriever | alpha={args.alpha} ...', flush=True)
    retriever = Hybrid(args.index, args.emb, k1=1.5, b=0.75)
    bm25 = retriever.bm25
    if len(retriever.emb) != bm25.meta['n_chunks']:
        raise SystemExit('Task 2 chunks and embeddings have different row counts: '
                         f'{bm25.meta["n_chunks"]} != {len(retriever.emb)}')

    questions = [value['question'] for _, value in items]
    print(f'Encoding {len(questions)} training questions ...', flush=True)
    qvecs = np.concatenate([
        retriever.encode_query(questions[start:start + args.qbatch])
        for start in range(0, len(questions), args.qbatch)
    ])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = skipped_weak = skipped_neg = 0
    positive_scores = []
    started = time.time()
    with open(args.out, 'w', encoding='utf-8') as output:
        for position, ((qid, item), qvec) in enumerate(zip(items, qvecs), 1):
            pool = candidate_pool(retriever, item['question'], qvec, args.pool,
                                  args.alpha, args.ndocs, args.mchunks)
            candidates = []
            for doc_id, chunk_index, retrieval_score in pool:
                chunk = bm25.read_chunk(chunk_index)
                coverage, teacher_score = weighted_overlap(
                    item['answer'], chunk['text'], bm25.vocab, bm25.idf)
                candidates.append({
                    'doc_id': doc_id,
                    'chunk_index': chunk_index,
                    'text': chunk['text'],
                    'retrieval_score': retrieval_score,
                    'coverage': coverage,
                    'teacher_score': teacher_score,
                })

            if not candidates:
                skipped_weak += 1
                continue
            positive = max(candidates, key=lambda value: value['teacher_score'])
            if positive['teacher_score'] < args.min_positive:
                skipped_weak += 1
                continue

            max_negative = positive['teacher_score'] * args.negative_ratio
            eligible = [value for value in candidates
                        if value['chunk_index'] != positive['chunk_index']
                        and value['doc_id'] != positive['doc_id']
                        and value['teacher_score'] <= max_negative]
            eligible.sort(key=lambda value: -value['retrieval_score'])
            if len(eligible) < args.neg:
                # Adjacent chunks from the same document are more likely to be false
                # negatives, so use this fallback only when necessary.
                used = {value['chunk_index'] for value in eligible}
                fallback = [value for value in candidates
                            if value['chunk_index'] != positive['chunk_index']
                            and value['chunk_index'] not in used
                            and value['teacher_score'] <= max_negative]
                fallback.sort(key=lambda value: -value['retrieval_score'])
                eligible.extend(fallback)
            negatives = eligible[:args.neg]
            if len(negatives) < args.neg:
                skipped_neg += 1
                continue

            row = {
                'qid': qid,
                'query': item['question'],
                'positive': positive['text'],
                'negatives': [value['text'] for value in negatives],
                'positive_doc': str(positive['doc_id']),
                'positive_chunk': positive['chunk_index'],
                'positive_teacher_score': round(positive['teacher_score'], 6),
                'positive_answer_coverage': round(positive['coverage'], 6),
                'negative_teacher_scores': [round(value['teacher_score'], 6)
                                            for value in negatives],
            }
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
            positive_scores.append(positive['teacher_score'])

            if position % 100 == 0:
                elapsed = time.time() - started
                eta = elapsed / position * (len(items) - position) / 60
                print(f'  {position}/{len(items)} | pairs={written} '
                      f'weak={skipped_weak} neg={skipped_neg} | ~{eta:.0f}m left',
                      flush=True)

    mean_score = float(np.mean(positive_scores)) if positive_scores else 0.0
    metadata = {
        'source': 'Task 2 train.json only',
        'offset': args.offset,
        'requested': len(items),
        'written': written,
        'skipped_weak': skipped_weak,
        'skipped_negatives': skipped_neg,
        'alpha': args.alpha,
        'pool': args.pool,
        'ndocs': args.ndocs,
        'mchunks': args.mchunks,
        'negatives': args.neg,
        'mean_positive_teacher_score': mean_score,
    }
    with open(args.out + '.meta.json', 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f'Pairs saved to: {args.out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
