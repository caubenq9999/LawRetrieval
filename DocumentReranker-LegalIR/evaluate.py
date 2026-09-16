#!/usr/bin/env python3
"""Evaluate baseline and LambdaMART on exactly the same candidate rows."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import (FEATURE_NAMES, baseline_scores, compare_results, group_rows,
                    load_json, load_rows, matrix, refuse_overwrite, score_groups)


def _load_model(path):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit('Thieu lightgbm; xem requirements.txt') from exc
    meta = load_json(path + '.meta.json')
    if meta.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Feature order cua model khong khop code hien tai')
    return lgb.Booster(model_file=path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output')
    p.add_argument('--bootstrap', type=int, default=4000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.output:
        refuse_overwrite(args.output, args.overwrite)

    rows = load_rows(args.features, require_labels=True)
    model = _load_model(args.model)
    pred = model.predict(matrix(rows), num_iteration=model.best_iteration)
    base_result = score_groups(rows, baseline_scores(rows), k=5)
    model_result = score_groups(rows, pred, k=5)
    report = compare_results(base_result, model_result, args.bootstrap, args.seed)

    # Candidate ceiling uses all candidate documents, not only top five.
    ceilings = []
    for items in group_rows(rows).values():
        gold_count = int(items[0].get('gold_count') or 0)
        if gold_count:
            ceilings.append(sum(int(x['label']) for x in items) / gold_count)
    report['candidate_recall'] = float(np.mean(ceilings)) if ceilings else None
    report['features'] = args.features
    report['model'] = args.model
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'-> {args.output}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
