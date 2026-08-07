#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Chay chuong trinh cham diem GOC cua BTC tren bo du lieu mock.

Script KHONG sua scoring.py. No doc nguyen van file do, exec trong mot namespace
rieng roi ghi de 3 bien duong dan module-level (reference_dir / prediction_dir /
score_dir) sang thu muc mock, truoc khi goi main(). Nho vay ket qua chay o local
phan anh dung hanh vi that tren he thong - ke ca cac truong hop scoring.py crash.

Usage:
    python run_scoring.py --mock mock --submission mock/samples/perfect.json
    python run_scoring.py --mock mock --submission my_submission.zip
    python run_scoring.py --mock mock --all
"""

import argparse
import io
import json
import os
import shutil
import sys
import traceback
import zipfile

DEFAULT_SCORING = os.path.join('..', 'Scoring-Program-Task-LegalIR', 'scoring.py')


def load_scoring_module(path):
    """Exec scoring.py cua BTC trong namespace rieng, khong chay khoi __main__."""
    if not os.path.exists(path):
        sys.exit(f'Khong tim thay scoring.py tai: {path}')
    with open(path, encoding='utf-8-sig') as f:
        source = f.read()
    namespace = {'__name__': 'btc_scoring', '__file__': os.path.abspath(path)}
    exec(compile(source, path, 'exec'), namespace)
    return namespace


def stage_submission(sub_path, res_dir):
    """Dat submission vao input/res/submission.json (nhan ca .json lan .zip)."""
    os.makedirs(res_dir, exist_ok=True)
    target = os.path.join(res_dir, 'submission.json')

    if sub_path.lower().endswith('.zip'):
        if not zipfile.is_zipfile(sub_path):
            sys.exit(f'Khong phai zip hop le: {sub_path}')
        with zipfile.ZipFile(sub_path) as zf:
            names = [n for n in zf.namelist()
                     if os.path.basename(n) == 'submission.json'
                     and not n.startswith('__MACOSX/')]
            if not names:
                sys.exit(f'Zip khong chua submission.json: {sub_path}')
            with open(target, 'wb') as out:
                out.write(zf.read(names[0]))
    else:
        shutil.copyfile(sub_path, target)
    return target


def run_once(scoring, mock_dir, sub_path, verbose=True):
    """Cham mot bai nop. Tra ve (scores | None, thong_diep_loi | None)."""
    ref_dir = os.path.join(mock_dir, 'input', 'ref')
    res_dir = os.path.join(mock_dir, 'input', 'res')
    score_dir = os.path.join(mock_dir, 'output')
    os.makedirs(score_dir, exist_ok=True)

    stage_submission(sub_path, res_dir)

    scores_path = os.path.join(score_dir, 'scores.json')
    if os.path.exists(scores_path):
        os.remove(scores_path)

    # Ghi de duong dan /app/... cua BTC bang thu muc mock.
    scoring['reference_dir'] = ref_dir
    scoring['prediction_dir'] = res_dir
    scoring['score_dir'] = score_dir

    captured = io.StringIO()
    real_stdout = sys.stdout
    try:
        sys.stdout = captured if not verbose else Tee(real_stdout, captured)
        scoring['main']()
    except Exception:
        sys.stdout = real_stdout
        tb = traceback.format_exc().strip().splitlines()
        return None, tb[-1]
    finally:
        sys.stdout = real_stdout

    if not os.path.exists(scores_path):
        return None, 'scoring.py ket thuc nhung khong ghi ra scores.json'
    with open(scores_path, encoding='utf-8') as f:
        return json.load(f), None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(
        description='Chay scoring.py goc cua BTC tren bo mock sinh tu train.json.')
    parser.add_argument('--mock', '-m', default='mock', help='Thu muc mock (mac dinh: mock)')
    parser.add_argument('--submission', '-s', help='File bai nop (.json hoac .zip)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Cham lan luot moi file trong mock/samples/ va so sanh')
    parser.add_argument('--scoring', default=DEFAULT_SCORING,
                        help='Duong dan scoring.py cua BTC')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.mock, 'input', 'ref')):
        sys.exit(f'Thu muc mock chua duoc tao: {args.mock}\n'
                 f'Chay truoc: python make_mock.py --train <train.json> --n 50')

    scoring = load_scoring_module(args.scoring)

    if args.all:
        sample_dir = os.path.join(args.mock, 'samples')
        samples = sorted(f for f in os.listdir(sample_dir) if f.endswith('.json'))
        rows = []
        for name in samples:
            scores, err = run_once(scoring, args.mock,
                                   os.path.join(sample_dir, name), verbose=False)
            rows.append((name[:-5], scores, err))

        print('=' * 72)
        print(f'KET QUA CHAM TREN BO MOCK: {args.mock}')
        print('=' * 72)
        print(f'{"Bai nop mau":<18}{"Recall":>10}{"Precision":>12}   Ghi chu')
        print('-' * 72)
        for name, scores, err in rows:
            if err:
                print(f'{name:<18}{"CRASH":>10}{"CRASH":>12}   {err}')
            else:
                print(f'{name:<18}{scores["recall"]:>10.4f}{scores["precision"]:>12.4f}')
        print('-' * 72)
        print('Recall la do do chinh; Precision chi dung khi bang Recall.')
        return 0

    if not args.submission:
        parser.error('Can --submission <file> hoac --all')

    print('=' * 72)
    print(f'Bai nop  : {args.submission}')
    print(f'Mock     : {args.mock}')
    print(f'Scoring  : {args.scoring}')
    print('-' * 72)
    scores, err = run_once(scoring, args.mock, args.submission, verbose=True)
    print('-' * 72)
    if err:
        print(f'THAT BAI: {err}')
        print('Bai nop nay se lam chuong trinh cham diem loi tren he thong that.')
        return 1
    print(f'Recall    : {scores["recall"]:.4f}   (do do chinh)')
    print(f'Precision : {scores["precision"]:.4f}   (do do phu)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass
    raise SystemExit(main())
