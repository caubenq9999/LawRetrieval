#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Tao bo du lieu mock tu train.json de chay thu chuong trinh cham diem cua BTC.

Sinh ra cay thu muc mo phong moi truong Codabench:

    mock/
      input/ref/metadata.json     <- tro toi truth.json va submission.json
      input/ref/truth.json        <- dap an, dinh dang PHANG {qid: [doc_id]}
      input/res/                  <- noi dat submission.json can cham
      output/                     <- noi scoring.py ghi scores.json
      questions.json              <- "de bai" gia lap (answer = null)
      answer_key.json             <- dap an kem cau hoi, de tra cuu
      samples/*.json              <- cac bai nop mau de kiem thu

Usage:
    python make_mock.py --train "../LegalIR - Public Test/train.json" --n 50
    python make_mock.py --train train.json --n 200 --seed 7 --out mock_big/
"""

import argparse
import json
import os
import random
import shutil

MAX_DOCS_PER_QUESTION = 5


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_samples(subset, noise_pool, rng):
    """Sinh cac bai nop mau, moi bai mo phong mot kieu he thong retrieval."""

    def noise(exclude, k):
        picked = []
        while len(picked) < k:
            d = rng.choice(noise_pool)
            if d not in exclude and d not in picked:
                picked.append(d)
        return picked

    samples = {}

    # Du doan hoan hao - moc tren cua thang do.
    samples['perfect'] = {q: {'answer': list(v)} for q, v in subset.items()}

    # Chi tra ve 1 ket qua: Precision = 1, Recall tut o cau co nhieu dap an.
    samples['top1'] = {q: {'answer': [v[0]]} for q, v in subset.items()}

    # Dap an dung + nhieu cho du 5 ID: Recall = 1, Precision thap.
    samples['padded'] = {
        q: {'answer': list(v) + noise(v, MAX_DOCS_PER_QUESTION - len(v))}
        for q, v in subset.items()
    }

    # Dung khoang mot nua so cau.
    half = {}
    for q, v in subset.items():
        half[q] = {'answer': list(v) if rng.random() < 0.5 else noise(v, len(v))}
    samples['half'] = half

    # Sai hoan toan - moc duoi.
    samples['random'] = {q: {'answer': noise(v, 1)} for q, v in subset.items()}

    # Vi pham rang buoc 5 ID o ~10% so cau -> nhung cau do phai bi 0 diem.
    violate = {}
    for q, v in subset.items():
        if rng.random() < 0.1:
            violate[q] = {'answer': list(v) + noise(v, 6 - len(v))}
        else:
            violate[q] = {'answer': list(v)}
    samples['violate_limit'] = violate

    # Loi kieu du lieu: document_id de nguyen int (loi am tham hay gap nhat).
    samples['int_ids'] = {q: {'answer': [int(d) for d in v]} for q, v in subset.items()}

    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Tao bo mock tu train.json de chay thu scoring.py cua BTC.')
    parser.add_argument('--train', '-t', required=True, help='Duong dan train.json')
    parser.add_argument('--n', '-n', type=int, default=50,
                        help='So cau hoi trich ra (mac dinh 50, dung 0 de lay het)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Seed ngau nhien')
    parser.add_argument('--out', '-o', default='mock', help='Thu muc dau ra (mac dinh: mock/)')
    parser.add_argument('--keep', action='store_true',
                        help='Giu lai thu muc cu thay vi xoa tao lai')
    args = parser.parse_args()

    with open(args.train, encoding='utf-8-sig') as f:
        train = json.load(f)

    qids = sorted(train.keys())
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    if args.n > 0:
        qids = qids[:args.n]
    qids.sort()

    subset = {q: [str(d) for d in train[q]['answer']] for q in qids}
    noise_pool = sorted({d for v in train.values() for d in map(str, v['answer'])})

    out = args.out
    if os.path.exists(out) and not args.keep:
        shutil.rmtree(out)

    # --- reference bundle (input/ref) ------------------------------------
    # truth.json PHANG: scoring.py doc y_true[k] truc tiep la list, khong qua ['answer'].
    write_json(os.path.join(out, 'input', 'ref', 'truth.json'), subset)
    write_json(os.path.join(out, 'input', 'ref', 'metadata.json'),
               {'files': {'reference': 'truth.json', 'input': 'submission.json'}})

    os.makedirs(os.path.join(out, 'input', 'res'), exist_ok=True)
    os.makedirs(os.path.join(out, 'output'), exist_ok=True)

    # --- de bai + dap an de tra cuu --------------------------------------
    write_json(os.path.join(out, 'questions.json'),
               {q: {'question': train[q]['question'], 'answer': None} for q in qids})
    write_json(os.path.join(out, 'answer_key.json'),
               {q: {'question': train[q]['question'], 'answer': subset[q]} for q in qids})

    # --- bai nop mau ------------------------------------------------------
    samples = build_samples(subset, noise_pool, rng)
    for name, sub in samples.items():
        write_json(os.path.join(out, 'samples', f'{name}.json'), sub)

    dist = {}
    for v in subset.values():
        dist[len(v)] = dist.get(len(v), 0) + 1

    print('=' * 68)
    print('DA TAO BO MOCK')
    print('=' * 68)
    print(f'Thu muc          : {os.path.abspath(out)}')
    print(f'So cau hoi       : {len(subset)} (seed {args.seed})')
    print('Phan bo dap an   : ' + ', '.join(f'{k} ID: {v} cau' for k, v in sorted(dist.items())))
    print(f'Kho ID lam nhieu : {len(noise_pool)}')
    print()
    print('Bai nop mau (mock/samples/):')
    for name in samples:
        print(f'  - {name}.json')
    print()
    print('Buoc tiep theo:')
    print(f'  python run_scoring.py --mock {out} --submission {out}/samples/perfect.json')
    print(f'  python run_scoring.py --mock {out} --all')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
