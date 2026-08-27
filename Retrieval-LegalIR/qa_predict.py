#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSC 2026 Task 2 (LegalQA) — standalone extractive baseline.

The official metric rewards lexical overlap. This baseline therefore retrieves
legal chunks, copies them verbatim, and adds a deterministic citation header;
it does not use a generative LLM.

Competition rule: Task 1 and Task 2 data must not be mixed. Build index_qa and
emb_qa_v2 only from QA/selected-contexts. Do not pass a model fine-tuned on
Task 1 as --emb or --reranker.

Examples (run from Retrieval-LegalIR):

    python qa_predict.py --eval -n 200 --sweep --retriever bm25
    python qa_predict.py --eval -n 200 --sweep --retriever hybrid
    python qa_predict.py --questions "../LegalIR - Public Test-.../QA/public-official.json" \
        --out submission_qa
"""

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRIEVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QA_DIR = os.path.join(
    ROOT, 'LegalIR - Public Test-20260806T081424Z-1-001', 'QA')
DEFAULT_INDEX = os.path.join(RETRIEVAL_DIR, 'index_qa')
DEFAULT_EMB = os.path.join(RETRIEVAL_DIR, 'emb_qa_v2')

RE_SLUG = re.compile(
    r'(?<!\d)(\d{1,4})-(\d{4})-([A-Z][A-Z0-9\-]{1,12}?)(?=-[a-z]|-\d|$)')
DOC_TYPES = [
    ('Nghi-dinh', 'Nghị định'),
    ('Thong-tu', 'Thông tư'),
    ('Quyet-dinh', 'Quyết định'),
    ('Luat', 'Luật'),
    ('Phap-lenh', 'Pháp lệnh'),
    ('Nghi-quyet', 'Nghị quyết'),
    ('Chi-thi', 'Chỉ thị'),
    ('Cong-van', 'Công văn'),
]


def load_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def build_citation_map(context_dir):
    """Map document IDs to citations such as `Nghị định 117/2020/NĐ-CP`."""
    citations = {}
    for filename in os.listdir(context_dir):
        if not filename.startswith('context_') or not filename.endswith('.json'):
            continue
        doc_id = filename[8:-5]
        doc = load_json(os.path.join(context_dir, filename))
        blob = f"{doc.get('name') or ''} {doc.get('link') or ''}"
        match = RE_SLUG.search(blob)
        if not match:
            continue
        number, year, suffix = match.groups()
        reference = f'{int(number)}/{year}/{suffix.upper().rstrip("-")}'
        doc_type = next(
            (vi for slug, vi in DOC_TYPES if slug.lower() in blob.lower()),
            'Văn bản')
        citations[doc_id] = f'{doc_type} {reference}'
    return citations


def build_answer(chunks, citations, style='cite'):
    """Build an answer from ranked `(doc_id, path, text)` chunks."""
    if not chunks:
        return 'Không tìm thấy căn cứ pháp lý phù hợp.'

    parts = []
    for doc_id, path, text in chunks:
        body = text.split('\n', 1)[1].strip() if '\n' in text else text.strip()
        if style == 'raw':
            parts.append(body)
            continue

        reference = citations.get(doc_id, '')
        article = path.split(' > ')[0]
        header = f'Căn cứ {article}'
        if reference:
            header += f' {reference}'
        parts.append(f'{header} quy định như sau:\n{body}')
    return '\n'.join(parts)


def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def score_task2(gold, predictions):
    """Compute METEOR (primary) and ROUGE-L (secondary)."""
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    meteor, rouge_l = [], []
    for question_id, reference in gold.items():
        prediction = predictions.get(question_id, '')
        if not prediction:
            meteor.append(0.0)
            rouge_l.append(0.0)
            continue
        meteor.append(meteor_score(
            [word_tokenize(reference.lower(), preserve_line=True)],
            word_tokenize(prediction.lower(), preserve_line=True)))
        rouge_l.append(scorer.score(reference, prediction)['rougeL'].fmeasure)
    return {'meteor': float(np.mean(meteor)),
            'rougeL': float(np.mean(rouge_l))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', default='bm25',
                        choices=['bm25', 'hybrid'],
                        help='BM25 needs no model; hybrid also uses --emb')
    parser.add_argument('--index', default=DEFAULT_INDEX,
                        help='Task 2 BM25 index built from QA contexts')
    parser.add_argument('--emb', default=DEFAULT_EMB,
                        help='Task 2 embedding directory (hybrid only)')
    parser.add_argument('--reranker',
                        help='Optional external or Task-2-only cross-encoder')
    parser.add_argument('--qa-dir', default=DEFAULT_QA_DIR)
    parser.add_argument('--questions', help='Public/private question JSON')
    parser.add_argument('--out', default='submission_qa')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate against train.json in --qa-dir')
    parser.add_argument('-n', type=int, default=200,
                        help='Number of training questions for --eval')
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip this many training questions for --eval')
    parser.add_argument('--nchunk', type=int, default=3,
                        help='Number of ranked chunks copied into each answer')
    parser.add_argument('--style', default='cite', choices=['cite', 'raw'])
    parser.add_argument('--sweep', action='store_true',
                        help='Evaluate 1-4 chunks and both answer styles in one run')
    parser.add_argument('--k1', type=float, default=1.5)
    parser.add_argument('--b', type=float, default=0.75)
    parser.add_argument('--pool', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dense weight for hybrid retrieval')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Reranker weight when --reranker is supplied')
    parser.add_argument('--ndocs', type=int, default=20,
                        help='Number of first-stage documents sent to reranking')
    parser.add_argument('--mchunks', type=int, default=3,
                        help='Maximum candidate chunks per document')
    parser.add_argument('--batch', type=int, default=16,
                        help='Optional reranker batch size')
    return parser.parse_args()


def bm25_chunk_scores(bm25, query, pool):
    """Return ranked non-zero BM25 chunk IDs and their scores."""
    scores, _ = bm25.score(query)
    order = np.flatnonzero(scores)
    if order.size > pool:
        order = order[np.argpartition(-scores[order], pool - 1)[:pool]]
    order = order[np.argsort(-scores[order])]
    return order, scores[order]


def main():
    args = parse_args()
    if not args.eval and not args.questions:
        raise SystemExit('Provide --questions, or use --eval.')

    context_dir = os.path.join(args.qa_dir, 'selected-contexts')
    if not os.path.isdir(context_dir):
        raise SystemExit(f'Task 2 contexts not found: {context_dir}')
    print('Building citation table ...', flush=True)
    citations = build_citation_map(context_dir)
    print(f'  {len(citations)} documents have a legal reference')

    if args.eval:
        train = load_json(os.path.join(args.qa_dir, 'train.json'))
        items = list(train.items())[args.offset:args.offset + args.n]
        questions = [(key, value['question']) for key, value in items]
        gold = {key: value['answer'] for key, value in items}
    else:
        data = load_json(args.questions)
        questions = [(key, value['question']) for key, value in data.items()]
        gold = None

    if not os.path.isdir(args.index):
        raise SystemExit(
            f'Task 2 index not found: {args.index}\n'
            'Build index_qa from QA/selected-contexts before prediction.')

    if args.retriever == 'hybrid':
        if not os.path.isdir(args.emb):
            raise SystemExit(
                f'Task 2 embeddings not found: {args.emb}\n'
                'Build emb_qa_v2 from QA chunks, or use --retriever bm25.')
        from hybrid import Hybrid
        retriever = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
        bm25 = retriever.bm25
        texts = [question for _, question in questions]
        query_vectors = np.concatenate([
            retriever.encode_query(texts[start:start + 64])
            for start in range(0, len(texts), 64)
        ]) if texts else np.empty((0, retriever.dim), dtype=np.float32)
    else:
        from bm25 import BM25Index
        retriever = None
        bm25 = BM25Index(args.index, k1=args.k1, b=args.b)
        query_vectors = None

    reranker = None
    if args.reranker:
        from rerank import Reranker
        reranker = Reranker(args.reranker, batch=args.batch)

    print(f'{len(questions)} questions | retriever={args.retriever} | '
          f'reranker={args.reranker or "none"} | '
          f'ndocs={args.ndocs} mchunks={args.mchunks}')

    predictions, candidates_by_question = {}, {}
    started = time.time()
    keep_top = max(args.nchunk, 4 if args.sweep else args.nchunk)

    for index, (question_id, question) in enumerate(questions):
        if retriever is None:
            order, fused = bm25_chunk_scores(bm25, question, args.pool)
        else:
            order, fused = retriever.fused_chunk_scores(
                question, args.pool, 'linear', args.alpha,
                query_vectors[index])
        if order.size == 0:
            candidates_by_question[question_id] = []
            predictions[question_id] = build_answer([], citations, args.style)
            continue

        per_document, hybrid_scores = {}, {}
        for position in np.argsort(-fused):
            chunk_index = int(order[position])
            doc_id = int(bm25.chunk_doc[chunk_index])
            per_document.setdefault(doc_id, []).append(chunk_index)
            hybrid_scores[chunk_index] = float(fused[position])
            if len(per_document) > args.ndocs and len(per_document[doc_id]) == 1:
                per_document.pop(doc_id)
                break

        pairs = [
            (doc_id, chunk_index)
            for doc_id, chunk_indices in per_document.items()
            for chunk_index in chunk_indices[:args.mchunks]
        ]
        first_stage = np.array([
            hybrid_scores[chunk_index] for _, chunk_index in pairs
        ], dtype=np.float32)
        if reranker is None:
            scores = first_stage
        else:
            candidate_texts = [
                bm25.read_chunk(chunk_index)['text']
                for _, chunk_index in pairs
            ]
            ce_scores = reranker.score(question, candidate_texts)
            scores = (args.beta * minmax(ce_scores)
                      + (1 - args.beta) * minmax(first_stage))

        selected = []
        for position in np.argsort(-scores)[:keep_top]:
            doc_id, chunk_index = pairs[position]
            chunk = bm25.read_chunk(chunk_index)
            selected.append((str(doc_id), chunk['path'], chunk['text']))

        candidates_by_question[question_id] = selected
        predictions[question_id] = build_answer(
            selected[:args.nchunk], citations, args.style)

        if (index + 1) % 50 == 0:
            elapsed = time.time() - started
            remaining = elapsed / (index + 1) * (len(questions) - index - 1) / 60
            print(f'  {index + 1}/{len(questions)} '
                  f'({elapsed:.0f}s, ~{remaining:.0f}m remaining)', flush=True)

    if gold and args.sweep:
        print(f'\n=== Sweep on {len(questions)} training questions ===')
        print(f'{"chunks":>7}{"style":>7}{"METEOR":>10}{"ROUGE-L":>10}{"length":>9}')
        print('-' * 43)
        for style in ('cite', 'raw'):
            for chunk_count in (1, 2, 3, 4):
                variants = {
                    key: build_answer(value[:chunk_count], citations, style)
                    for key, value in candidates_by_question.items()
                }
                result = score_task2(gold, variants)
                average_length = np.mean([len(value) for value in variants.values()])
                print(f'{chunk_count:>7}{style:>7}{result["meteor"]:>10.4f}'
                      f'{result["rougeL"]:>10.4f}{average_length:>9.0f}')
        print('-' * 43)
        print(f'Average reference length: '
              f'{np.mean([len(value) for value in gold.values()]):.0f} characters')
        return 0

    if gold:
        result = score_task2(gold, predictions)
        print(f'\n=== {len(questions)} questions | chunks={args.nchunk} '
              f'style={args.style} ===')
        print(f'METEOR : {result["meteor"]:.4f}')
        print(f'ROUGE-L: {result["rougeL"]:.4f}')
        return 0

    json_path = f'{args.out}.json'
    zip_path = f'{args.out}.zip'
    payload = {key: {'answer': value} for key, value in predictions.items()}
    with io.open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, 'submission.json')
    print(f'\n{len(payload)} answers -> {zip_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
