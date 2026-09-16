#!/usr/bin/env python3
"""Create a Task 1 submission from exported public candidate features."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile

from common import (FEATURE_NAMES, baseline_scores, load_json, load_rows, matrix,
                    refuse_overwrite, score_groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--method', choices=['lambdamart', 'baseline'], default='lambdamart')
    p.add_argument('--model', help='Bat buoc khi --method lambdamart')
    p.add_argument('--output', required=True, help='Duong dan .zip')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.output.endswith('.zip'):
        p.error('--output phai co duoi .zip')
    if args.method == 'lambdamart' and not args.model:
        p.error('--model bat buoc khi --method lambdamart')
    json_path = args.output[:-4] + '.json'
    refuse_overwrite(args.output, args.overwrite)
    refuse_overwrite(json_path, args.overwrite)

    rows = load_rows(args.features, require_labels=False)
    if args.method == 'baseline':
        scores = baseline_scores(rows)
    else:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise SystemExit('Thieu lightgbm; xem requirements.txt') from exc
        meta = load_json(args.model + '.meta.json')
        if meta.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Feature order cua model khong khop code hien tai')
        model = lgb.Booster(model_file=args.model)
        scores = model.predict(matrix(rows), num_iteration=model.best_iteration)

    ranked = score_groups(rows, scores, k=5)
    submission = {qid: {'answer': item['docs']} for qid, item in ranked.items()}
    short = [qid for qid, item in submission.items() if len(item['answer']) != 5]
    if short:
        raise ValueError(f'{len(short)} query khong du 5 document; vi du {short[:5]}')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, 'submission.json')
    print(f'{len(submission)} query -> {args.output}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
