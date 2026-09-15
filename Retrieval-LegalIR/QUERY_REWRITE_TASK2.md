# Task 2 LegalQA: query rewriting (experimental)

This feature adds a conservative, model-free second retrieval query. It does **not** rewrite the user's question for answer generation, and it does **not** treat a rewritten query as legal evidence. Only chunks returned from Task 2's `selected-contexts` can appear in the final answer.

## Scope and behavior

- `qa_query_rewrite.py` maps a few informal question patterns to legal-register search terms, removes generic question endings, and leaves law identifiers, dates, money amounts, and case facts untouched.
- `qa_predict.py --query-rewrite` retrieves with both the original question and the variant. The original result remains primary; the second result receives a bounded score weight (`--rewrite-weight`, default `0.25`). No rewrite is applied when no useful variant exists.
- Both BM25-only and hybrid modes are supported. In hybrid mode, a second query embedding is computed, so the option can add runtime cost.
- The generator in `qa_generate_vllm.py` still receives the original question and the selected legal chunks. Query rewriting can only help it indirectly by changing those chunks.
- `--corpus-rewrite` is a separate, more tightly gated exploratory option in `qa_corpus_rewrite.py`. It is **not** recommended as a default: a subsequent 200-question smoke test triggered no expansions after the structural safety gate was tightened.

## Paired evaluation

Run the two variants on exactly the same held-out train question IDs, index, embedding, and answer settings. Use a QA-only index built from Task 2 contexts; never mix Task 1 and Task 2 data or fine-tuned models.

```bash
cd Retrieval-LegalIR
python qa_predict.py --qa-dir /path/to/QA --index /path/to/index_qa \
  --eval -n 200 --offset 0 --retriever bm25 \
  --candidates-out /path/to/qa_baseline_candidates.json \
  --out /path/to/qa_baseline
python qa_predict.py --qa-dir /path/to/QA --index /path/to/index_qa \
  --eval -n 200 --offset 0 --retriever bm25 --query-rewrite \
  --candidates-out /path/to/qa_rewrite_candidates.json \
  --out /path/to/qa_rewrite
python smoke_qa_query_rewrite.py \
  --baseline /path/to/qa_baseline_candidates.json \
  --rewrite /path/to/qa_rewrite_candidates.json
```

The smoke script reports METEOR (the Task 2 primary metric), ROUGE-L, per-question wins/losses, changed selected documents, and a paired-bootstrap interval. It uses the scorer's whitespace tokenization for METEOR. For a final decision, evaluate at least one **fresh**, non-overlapping train slice and inspect the largest negative cases. Do not tune weights and report performance on the same slice.

## Preliminary observations, not a leaderboard claim

Earlier local BM25-only smoke runs suggested a small mean METEOR increase: about `+0.001` on 20 questions and about `+0.008` on a separate 50-question slice. These samples are too small to establish a reliable gain; individual questions degraded when the rewritten query promoted the wrong provision. The feature is therefore opt-in and should not be used for submission until a larger paired evaluation passes. These numbers concern the extractive `rule_answer`, **not** an isolated improvement in LLM answer generation.

## Answer-generation ablation

To determine whether rewriting helps final answers, compare four cells on identical question IDs: baseline retrieval + extractive answer; rewritten retrieval + extractive answer; baseline retrieval + generator; rewritten retrieval + generator. The first comparison measures retrieval-mediated effects; the same-context extractive-vs-generator comparison measures answer-generation effects. Keep the generator model, decoding temperature, and candidate count fixed.
