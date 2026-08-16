#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Gop hai bi-encoder o tang hybrid.

Tren dia dang co hai bo vector day du cua hai model khac ho:
    emb_ft    halong-ft   (nen intfloat/multilingual-e5-base, 768 chieu, mean pooling)
    emb_v2ft  v2-ft       (nen BAAI/bge-m3,                  1024 chieu, CLS pooling)
Diem dense cua chung tuong quan 0.798 - du khac de bo sung cho nhau.

    diem_dense = w*minmax(dense_B) + (1-w)*minmax(dense_A)
    fused      = alpha*diem_dense + (1-alpha)*minmax(bm25)

Meo giu chi phi thap: pool 2000 chunk do BM25 chon nen KHONG phu thuoc w hay alpha.
Dump diem tho mot lan cho moi model (sweep_alpha.py dump), roi chi cham cross-encoder
tren HOP cac tap ung vien sinh boi ca luoi - mot lan chay cho ca luoi.

    python sweep_alpha.py dump -n 500 --offset 500 --emb emb_ft    -o pool_ft_b.npz
    python sweep_alpha.py dump -n 500 --offset 500 --emb emb_v2ft  -o pool_v2ft_b.npz
    python ens_dense.py union -a pool_ft_b.npz -b pool_v2ft_b.npz -o union_ens.json
    python ens_dense.py score -u union_ens.json -o sc_ens.json
    python ens_dense.py sweep -a pool_ft_b.npz -b pool_v2ft_b.npz -s sc_ens.json
"""

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'LegalIR - Public Test-20260806T081424Z-1-001', 'LegalIR - Public Test')
TRAIN = os.path.join(BASE, 'train.json')
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'Finetune-LegalIR', 'data', 'split.json')
RERANKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'Finetune-LegalIR', 'models', 'reranker-ft-fp16')

NDOCS, MCHUNKS = 50, 2
WS = [0.0, 0.3, 0.5, 0.7, 1.0]          # trong so cua model B (v2-ft)
ALPHAS = [0.6, 0.7, 0.8]


def load_json(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def mm(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _load2(pa, pb):
    with np.load(pa, allow_pickle=False) as f:
        A = {k: f[k] for k in f.files}
    with np.load(pb, allow_pickle=False) as f:
        B = {k: f[k] for k in f.files}
    assert (A['idx'] == B['idx']).all(), 'hai pool khong khop - phai cung k1/b'
    gold = load_json(pa + '.gold.json')
    return A, B, gold


def _cands(A, B, i, w, alpha):
    s, e = A['offs'][i], A['offs'][i + 1]
    if e <= s:
        return []
    idx = A['idx'][s:e]
    dense = w * mm(B['dense'][s:e]) + (1 - w) * mm(A['dense'][s:e])
    fused = alpha * mm(dense) + (1 - alpha) * mm(A['bm25'][s:e])
    per_doc, out = {}, []
    for j in np.argsort(-fused):
        ci = int(idx[j])
        d = int(A['chunk_doc'][ci])
        if d not in per_doc:
            if len(per_doc) >= NDOCS:
                break
            per_doc[d] = 0
        if per_doc[d] < MCHUNKS:
            per_doc[d] += 1
            out.append((str(d), ci, float(fused[j])))
    return out


def cmd_union(args):
    A, B, gold = _load2(args.a, args.b)
    qids = [str(q) for q in A['qids']]
    union = {}
    for i, q in enumerate(qids):
        u = set()
        for w in WS:
            for al in ALPHAS:
                u.update(ci for _, ci, _ in _cands(A, B, i, w, al))
        union[q] = sorted(u)
    tot = sum(len(v) for v in union.values())
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(union, f)
    print(f'{len(qids)} cau | hop qua {len(WS)}x{len(ALPHAS)} to hop: '
          f'{tot} cap ({tot / len(qids):.0f}/cau) -> {args.out}')
    return 0


def cmd_score(args):
    from bm25 import BM25Index
    from rerank import Reranker
    union = load_json(args.union)
    train = load_json(TRAIN)
    bm = BM25Index('index')
    rr = Reranker(args.model, batch=16)
    out, t0 = {}, time.time()
    for i, (q, cis) in enumerate(union.items()):
        if not cis:
            out[q] = {}
            continue
        txt = [bm.read_chunk(int(c))['text'] for c in cis]
        sc = rr.score(train[q]['question'], txt)
        out[q] = {str(c): float(s) for c, s in zip(cis, sc)}
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(union)} ({el:.0f}s, con ~{el / (i + 1) * (len(union) - i - 1) / 60:.0f}p)',
                  flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'scores': out}, f)
    print(f'\nxong {time.time() - t0:.0f}s -> {args.out}')
    return 0


def _prior():
    train = load_json(TRAIN)
    ex = set(map(str, load_json(SPLIT)['val']))
    freq = collections.Counter(str(d) for q, v in train.items() if q not in ex
                               for d in v['answer'])
    mx = math.log1p(max(freq.values()))
    return {d: math.log1p(c) / mx for d, c in freq.items()}


def cmd_sweep(args):
    A, B, gold = _load2(args.a, args.b)
    ce = load_json(args.scores)['scores']
    qids = [str(q) for q in A['qids']]
    n = len(qids)
    W = _prior()

    def recall(w, alpha, beta, lam, aw):
        out = []
        for i, q in enumerate(qids):
            cand = _cands(A, B, i, w, alpha)
            if not cand:
                out.append(0.0)
                continue
            fh = np.array([c[2] for c in cand])
            cs = np.array([ce[q].get(str(c[1]), 0.0) for c in cand])
            s = beta * mm(cs) + (1 - beta) * mm(fh)
            vals = {}
            for j in np.argsort(-s):
                vals.setdefault(cand[j][0], []).append(float(s[j]))
            ds = sorted(((d, x[0] + aw * (x[1] if len(x) > 1 else 0.0))
                         for d, x in vals.items()), key=lambda t: -t[1])
            v = mm(np.array([x for _, x in ds]))
            if lam:
                v = v + lam * np.array([W.get(d, 0.0) for d, _ in ds])
            top = [ds[j][0] for j in np.argsort(-v)[:5]]
            g = set(gold[q])
            out.append(len(g & set(top)) / len(g))
        return np.array(out)

    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(8000, n))
    ref = recall(1.0, args.alpha, args.beta, args.lam, args.aggw)   # w=1.0: chi v2-ft
    print(f'{n} cau | moc: chi v2-ft (w=1.0), alpha={args.alpha} beta={args.beta} '
          f'-> {ref.mean():.4f}\n')
    print(f'{"w(v2ft)":>8}{"alpha":>7}{"Recall@5":>11}{"delta":>9}{"boot 95% CI":>22}')
    print('-' * 57)
    rows = []
    for w in args.ws:
        for al in args.alphas:
            r = recall(w, al, args.beta, args.lam, args.aggw)
            d = r - ref
            bs = d[boot].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((w, al, r))
            print(f'{w:>8}{al:>7}{r.mean():>11.4f}{d.mean():>+9.4f}'
                  f'{f"[{lo:+.4f}, {hi:+.4f}]":>22}')
    print('-' * 57)
    best = max(rows, key=lambda t: t[2].mean())
    d = best[2] - ref
    bs = d[boot].mean(axis=1)
    print(f'\nTot nhat: w={best[0]} alpha={best[1]} -> {best[2].mean():.4f} '
          f'({d.mean():+.4f}, thang {(bs > 0).mean() * 100:.0f}%)')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('union')
    u.add_argument('-a', required=True, help='pool cua model A (halong-ft)')
    u.add_argument('-b', required=True, help='pool cua model B (v2-ft)')
    u.add_argument('-o', '--out', required=True)
    u.set_defaults(fn=cmd_union)
    s = sub.add_parser('score')
    s.add_argument('-u', '--union', required=True)
    s.add_argument('--model', default=RERANKER)
    s.add_argument('-o', '--out', required=True)
    s.set_defaults(fn=cmd_score)
    w = sub.add_parser('sweep')
    w.add_argument('-a', required=True)
    w.add_argument('-b', required=True)
    w.add_argument('-s', '--scores', required=True)
    w.add_argument('--ws', type=float, nargs='+', default=WS)
    w.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    w.add_argument('--alpha', type=float, default=0.7)
    w.add_argument('--beta', type=float, default=0.5)
    w.add_argument('--lam', type=float, default=0.2)
    w.add_argument('--aggw', type=float, default=0.4)
    w.set_defaults(fn=cmd_sweep)
    return p.parse_args().fn(p.parse_args())


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
