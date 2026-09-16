#!/usr/bin/env python3
"""Train a document-level LightGBM LambdaMART ranker."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import (FEATURE_NAMES, baseline_scores, compare_results, group_sizes,
                    labels, load_rows, matrix, refuse_overwrite, score_groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--model-out', required=True)
    p.add_argument('--metrics-out')
    p.add_argument('--trees', type=int, default=1000)
    p.add_argument('--learning-rate', type=float, default=0.03)
    p.add_argument('--num-leaves', type=int, default=15)
    p.add_argument('--min-data-in-leaf', type=int, default=30)
    p.add_argument('--feature-fraction', type=float, default=0.9)
    p.add_argument('--bagging-fraction', type=float, default=0.9)
    p.add_argument('--early-stopping', type=int, default=80)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.metrics_out:
        args.metrics_out = args.model_out + '.metrics.json'
    refuse_overwrite(args.model_out, args.overwrite)
    refuse_overwrite(args.model_out + '.meta.json', args.overwrite)
    refuse_overwrite(args.metrics_out, args.overwrite)

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            'Thieu lightgbm. Cai bang: pip install -r '
            'DocumentReranker-LegalIR/requirements.txt') from exc

    train_rows = load_rows(args.train, require_labels=True)
    valid_rows = load_rows(args.valid, require_labels=True)
    x_train, y_train = matrix(train_rows), labels(train_rows)
    x_valid, y_valid = matrix(valid_rows), labels(valid_rows)
    g_train, g_valid = group_sizes(train_rows), group_sizes(valid_rows)
    print(f'Train: {len(g_train)} query, {len(train_rows)} document rows, '
          f'{int(y_train.sum())} positive')
    print(f'Valid: {len(g_valid)} query, {len(valid_rows)} document rows, '
          f'{int(y_valid.sum())} positive')

    dtrain = lgb.Dataset(x_train, label=y_train, group=g_train,
                         feature_name=FEATURE_NAMES, free_raw_data=False)
    dvalid = lgb.Dataset(x_valid, label=y_valid, group=g_valid,
                         feature_name=FEATURE_NAMES, reference=dtrain, free_raw_data=False)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'min_data_in_leaf': args.min_data_in_leaf,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': 1,
        'verbosity': -1,
        'seed': args.seed,
        'feature_fraction_seed': args.seed,
        'bagging_seed': args.seed,
        'deterministic': True,
        'force_col_wise': True,
    }
    callbacks = [lgb.log_evaluation(period=25)]
    if args.early_stopping:
        callbacks.append(lgb.early_stopping(args.early_stopping, verbose=True))
    model = lgb.train(params, dtrain, num_boost_round=args.trees,
                      valid_sets=[dvalid], valid_names=['valid'], callbacks=callbacks)
    model.save_model(args.model_out, num_iteration=model.best_iteration)

    pred = model.predict(x_valid, num_iteration=model.best_iteration)
    baseline = score_groups(valid_rows, baseline_scores(valid_rows), k=5)
    candidate = score_groups(valid_rows, pred, k=5)
    report = compare_results(baseline, candidate, seed=args.seed)
    report.update({
        'best_iteration': int(model.best_iteration or model.current_iteration()),
        'train_rows': len(train_rows),
        'valid_rows': len(valid_rows),
        'feature_importance_gain': {
            name: float(value) for name, value in sorted(
                zip(FEATURE_NAMES, model.feature_importance(importance_type='gain')),
                key=lambda x: -x[1])
        },
    })
    with open(args.model_out + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'format_version': 1, 'feature_names': FEATURE_NAMES,
                   'params': params, 'best_iteration': report['best_iteration']},
                  f, ensure_ascii=False, indent=2)
    with open(args.metrics_out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Model   -> {args.model_out}')
    print(f'Metrics -> {args.metrics_out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
