#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate cached LLM query expansions for LegalIR.

This script runs before prediction. It writes a JSON cache keyed by question_id,
so the final retrieval run is deterministic and does not call the LLM online.

Recommended model for DSC rules:
  Qwen/Qwen2.5-1.5B-Instruct

It is Apache-2.0, instruction-tuned, multilingual, and about 1.54B parameters.
Register the model with the organizers before using it in an official run.
"""

import argparse
import json
import os
import sys
import time

from query_expansion import SYSTEM_PROMPT, extract_json, filter_expansion, rule_expand


DEFAULT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'


def pick_device(name):
    import torch

    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class LocalExpander:
    def __init__(self, model_id, device='auto', max_new_tokens=160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = pick_device(device)
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device in ('cuda', 'mps') else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def expand(self, question):
        import torch

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Question:\n' + str(question).strip()},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        enc = self.tok([prompt], return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0][enc['input_ids'].shape[-1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return extract_json(text)


def load_questions(path):
    with open(path, encoding='utf-8-sig') as f:
        data = json.load(f)
    return [(qid, item.get('question', '')) for qid, item in data.items()]


def main():
    p = argparse.ArgumentParser(description='Generate LegalIR query expansions.')
    p.add_argument('--questions', '-q', required=True,
                   help='train.json / public-official.json / private-official.json')
    p.add_argument('--out', '-o', required=True,
                   help='Output expansion cache JSON')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--limit', '-n', type=int, default=0)
    p.add_argument('--rules-only', action='store_true',
                   help='Do not load an LLM; use deterministic fallback rules.')
    p.add_argument('--resume', action='store_true',
                   help='Reuse existing output records if present.')
    p.add_argument('--max-new-tokens', type=int, default=160)
    args = p.parse_args()

    qs = load_questions(args.questions)
    if args.limit:
        qs = qs[:args.limit]

    cache = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding='utf-8-sig') as f:
            cache = json.load(f)

    expander = None
    if not args.rules_only:
        print(f'Load expansion model: {args.model}')
        expander = LocalExpander(args.model, args.device, args.max_new_tokens)
        print(f'Device: {expander.device}')
    else:
        print('Mode: rules-only fallback')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    t0 = time.time()
    n_fail = 0
    for i, (qid, question) in enumerate(qs, 1):
        if qid in cache:
            continue
        try:
            raw = rule_expand(question) if args.rules_only else expander.expand(question)
            fixed = filter_expansion(question, raw)
            source = 'rules' if args.rules_only else args.model
        except Exception as e:
            n_fail += 1
            fixed = rule_expand(question)
            source = f'fallback_rules_after_error: {type(e).__name__}'
        cache[qid] = {
            'question': question,
            'normalized_query': fixed['normalized_query'],
            'expansion_terms': fixed['expansion_terms'],
            'source': source,
        }
        if i % 25 == 0 or i == len(qs):
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f'  {i}/{len(qs)} saved | fallback/errors {n_fail} '
                  f'| {time.time() - t0:.0f}s', flush=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(cache)} expansions -> {args.out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
