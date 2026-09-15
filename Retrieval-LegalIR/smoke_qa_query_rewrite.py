#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare paired qa_predict.py --candidates-out files with the BTC metric.

Use the same --eval questions, --index, --pool, and answer settings in both
retrieval runs. The only intended difference is --query-rewrite.
"""

import argparse
import json

import numpy as np
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from qa_query_rewrite import rewrite_query


def load(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--rewrite', required=True)
    args = parser.parse_args()

    baseline, rewritten = load(args.baseline), load(args.rewrite)
    if baseline.keys() != rewritten.keys():
        raise SystemExit('The two candidate files must contain identical question IDs.')

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rows = []
    for question_id, base in baseline.items():
        changed = rewritten[question_id]
        if (base['question'] != changed['question'] or
                base['gold_answer'] != changed['gold_answer']):
            raise SystemExit(f'Mismatched paired question: {question_id}')
        gold = base['gold_answer']
        metrics = []
        for candidate in (base, changed):
            answer = candidate['rule_answer']
            metrics.append((
                meteor_score([str(gold).split()], str(answer).split()),
                rouge.score(str(gold), str(answer))['rougeL'].fmeasure))
        old_docs = [c['doc_id'] for c in base['chunks']]
        new_docs = [c['doc_id'] for c in changed['chunks']]
        applied_query = changed.get('retrieval_query')
        if 'rewrite_mode' not in changed:
            applied_query = rewrite_query(base['question'])
        rows.append((question_id, metrics[0], metrics[1],
                     old_docs != new_docs, applied_query))

    mean = lambda index, metric: float(np.mean([row[index][metric] for row in rows]))
    delta = [row[2][0] - row[1][0] for row in rows]
    sampled = np.asarray(delta)[np.random.default_rng(42).integers(
        0, len(rows), size=(10000, len(rows)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    print(f'Paired train questions: {len(rows)}')
    print(f'Rewritten queries: {sum(row[4] is not None for row in rows)}')
    print(f'Changed selected document IDs: {sum(row[3] for row in rows)}')
    print('                 baseline  rewrite   delta')
    print(f'METEOR           {mean(1, 0):.4f}    {mean(2, 0):.4f}  '
          f'{mean(2, 0) - mean(1, 0):+.4f}')
    print(f'ROUGE-L          {mean(1, 1):.4f}    {mean(2, 1):.4f}  '
          f'{mean(2, 1) - mean(1, 1):+.4f}')
    print(f'METEOR improved/worse/tied: '
          f'{sum(d > 1e-6 for d in delta)}/'
          f'{sum(d < -1e-6 for d in delta)}/'
          f'{sum(abs(d) <= 1e-6 for d in delta)}')
    print(f'Paired bootstrap 95% interval for METEOR delta: '
          f'[{low:+.4f}, {high:+.4f}]')
    changed_rows = [row for row in rows if abs(row[2][0] - row[1][0]) > 1e-6]
    print('Largest absolute METEOR changes:')
    if not changed_rows:
        print('  none')
    for question_id, old, new, docs_changed, query in sorted(
            changed_rows, key=lambda row: -abs(row[2][0] - row[1][0]))[:8]:
        print(f'  {question_id}: {new[0] - old[0]:+.4f} '
              f'(doc IDs changed={docs_changed}) | rewrite={query or "<none>"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
