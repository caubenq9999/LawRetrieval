#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Fine-tune hiieu/halong_embedding tren du lieu cuoc thi.

Loss: MultipleNegativesRankingLoss boc trong MatryoshkaLoss.
  - MNRL: hoc phan biet positive voi hard negative + negative cung batch.
    Batch cang lon cang nhieu negative "mien phi" -> chat luong cao hon.
  - Matryoshka: model goc duoc train bang MatryoshkaLoss, giu lai de van cat
    duoc chieu 768->256 sau nay ma khong hong.

Rang buoc phan cung: 6.4 GB VRAM. Model 278M + AdamW (2 trang thai fp32) da ton
~4.4 GB truoc khi tinh activation. Nen bat gradient checkpointing va giu
max_seq_length 256 (chunk median 230 token, nen cat o 256 mat khong nhieu).

Usage:
    python train.py --data data/pairs.jsonl --out models/halong-ft --smoke
    python train.py --data data/pairs.jsonl --out models/halong-ft --epochs 1
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

BASE_MODEL = 'hiieu/halong_embedding'
MATRYOSHKA_DIMS = [768, 512, 256, 128]


def load_pairs(path, n_neg, limit=0):
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            negs = r['negatives'][:n_neg]
            if len(negs) < n_neg:            # bo cau thieu negative de cot dong deu
                continue
            item = {'anchor': r['query'], 'positive': r['positive']}
            for j, ng in enumerate(negs):
                item[f'negative_{j + 1}'] = ng
            rows.append(item)
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='data/pairs.jsonl')
    p.add_argument('--out', '-o', default='models/halong-ft')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--neg', type=int, default=3, help='So hard negative moi cau hoi')
    p.add_argument('--maxlen', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--smoke', action='store_true', help='Chay 30 buoc de do VRAM/toc do')
    args = p.parse_args()

    ds = load_pairs(args.data, args.neg, limit=400 if args.smoke else 0)
    print(f'Du lieu: {len(ds)} mau | {args.neg} hard negative/cau')
    print(f'Cot: {ds.column_names}')

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = args.maxlen
    model[0].auto_model.gradient_checkpointing_enable()

    inner = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, inner, matryoshka_dims=MATRYOSHKA_DIMS)

    seqs = args.batch * (2 + args.neg)
    print(f'Moi buoc xu ly {seqs} chuoi x {args.maxlen} token '
          f'(batch {args.batch}, accum {args.accum})')

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        fp16=True,
        max_steps=30 if args.smoke else -1,
        logging_steps=10,
        save_strategy='no' if args.smoke else 'epoch',
        save_total_limit=1,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=ds, loss=loss)
    t0 = time.time()
    trainer.train()
    el = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    steps = 30 if args.smoke else len(ds) * args.epochs / (args.batch * args.accum)
    print()
    print(f'Xong {steps:.0f} buoc trong {el / 60:.1f} phut ({el / max(steps, 1):.2f}s/buoc)')
    print(f'VRAM dinh: {peak:.2f} GB / 6.4 GB')
    if args.smoke:
        full = len(load_pairs(args.data, args.neg)) * args.epochs / (args.batch * args.accum)
        print(f'Uoc chay day du: {full:.0f} buoc -> {full * el / 30 / 60:.0f} phut')
    else:
        model.save_pretrained(args.out)
        print(f'Da luu: {args.out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
