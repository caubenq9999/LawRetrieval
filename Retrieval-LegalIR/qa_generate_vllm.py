#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate grounded Task 2 answers from retrieved chunks with vLLM.

Input is produced by qa_predict.py --candidates-out. Retrieval and generation
run as separate processes so the embedding model/reranker release all VRAM
before vLLM reserves its KV cache.
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile

# FlashInfer 0.6.x does not recognize Blackwell SM 12.x correctly on the
# current Colab CUDA toolkit. Force vLLM's native PyTorch/Triton sampler; model
# attention still uses FlashAttention and generation quality is unchanged.
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'


DEFAULT_MODEL = 'Qwen/Qwen3.5-2B'

REWRITE_SYSTEM_PROMPT = """Bạn là bộ biên tập đáp án hỏi đáp pháp luật Việt Nam.
Chỉ được sử dụng thông tin trong BẢN NHÁP TRÍCH XUẤT và DANH SÁCH NGUỒN được cung cấp.

Quy tắc bắt buộc:
1. Giữ nguyên câu chữ pháp lý quan trọng, số tiền, thời hạn, Điều, Khoản và tên văn bản; không tự diễn giải chúng sang cách nói khác.
2. Xóa đoạn không trực tiếp trả lời câu hỏi, nhưng giữ đủ điều kiện, ngoại lệ, hình thức xử phạt và biện pháp khắc phục có liên quan.
3. Có thể thêm đúng một câu kết luận ngắn bắt đầu bằng "Theo đó," dựa hoàn toàn trên căn cứ.
4. Không được thêm kiến thức, con số hoặc căn cứ không xuất hiện trong dữ liệu đầu vào.
5. Không nói về quá trình suy luận, không dùng markdown, không viết lời chào.
6. Chỉ xuất ra đáp án cuối cùng bằng tiếng Việt."""

CONCLUSION_SYSTEM_PROMPT = """Bạn viết đúng MỘT câu kết luận cho đáp án hỏi đáp pháp luật Việt Nam.

Quy tắc bắt buộc:
1. Câu phải bắt đầu chính xác bằng "Theo đó," và trả lời trực tiếp câu hỏi.
2. Chỉ dùng thông tin có trong BẢN NHÁP TRÍCH XUẤT; không thêm kiến thức bên ngoài.
3. Mọi số tiền, thời hạn, Điều, Khoản và điều kiện phải giữ nguyên cách viết trong bản nháp.
4. Không nhắc lại toàn bộ căn cứ, không giải thích quá trình suy luận, không dùng markdown.
5. Tối đa 80 từ tiếng Việt và chỉ xuất đúng câu kết luận, không có lời dẫn."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', required=True, help='Output prefix without extension')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-tokens', type=int, default=1200)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--fallback-min-chars', type=int, default=80)
    parser.add_argument('--mode', default='conclusion',
                        choices=['conclusion', 'rewrite'])
    return parser.parse_args()


def build_user_prompt(row, mode):
    sources = []
    for index, chunk in enumerate(row.get('chunks', []), 1):
        sources.append(
            f'{index}. Văn bản {chunk.get("doc_id", "")}: '
            f'{chunk.get("path", "")}')
    instruction = ('Hãy viết đúng một câu kết luận theo các quy tắc hệ thống.'
                   if mode == 'conclusion' else
                   'Hãy biên tập thành đáp án cuối cùng theo đúng các quy tắc hệ thống.')
    return (
        f'CÂU HỎI:\n{row["question"].strip()}\n\n'
        f'BẢN NHÁP TRÍCH XUẤT:\n{row["rule_answer"].strip()}\n\n'
        f'DANH SÁCH NGUỒN:\n{chr(10).join(sources)}\n\n'
        + instruction)


def clean_answer(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'^Đáp án(?: cuối cùng)?\s*:\s*', '', text,
                  flags=re.IGNORECASE).strip()
    return text


def valid_conclusion(conclusion, row):
    """Reject unsupported numbers and malformed/overlong conclusions."""
    if not conclusion.startswith('Theo đó,'):
        return False
    if not 20 <= len(conclusion) <= 600:
        return False
    lowered = conclusion.lower()
    if any(marker in lowered for marker in (
            'không đủ thông tin', 'không có đủ thông tin', 'tôi không thể')):
        return False
    source = f'{row["question"]}\n{row["rule_answer"]}'
    source_numbers = set(re.findall(r'\d[\d./-]*', source))
    generated_numbers = set(re.findall(r'\d[\d./-]*', conclusion))
    return generated_numbers.issubset(source_numbers)


def write_submission(prefix, predictions):
    json_path = f'{prefix}.json'
    zip_path = f'{prefix}.zip'
    payload = {key: {'answer': value} for key, value in predictions.items()}
    with io.open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, 'submission.json')
    return zip_path


def main():
    args = parse_args()
    with open(args.candidates, encoding='utf-8') as stream:
        data = json.load(stream)
    if not data:
        raise SystemExit('Candidate file is empty.')

    from vllm import LLM, SamplingParams

    print(f'Loading vLLM model: {args.model}', flush=True)
    llm = LLM(
        model=args.model,
        dtype='bfloat16',
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    keys = list(data)
    prompts = []
    for key in keys:
        messages = [
            {'role': 'system', 'content': (CONCLUSION_SYSTEM_PROMPT
                                          if args.mode == 'conclusion'
                                          else REWRITE_SYSTEM_PROMPT)},
            {'role': 'user', 'content': build_user_prompt(data[key], args.mode)},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False))

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.03,
    )
    print(f'Generating {len(prompts)} answers ...', flush=True)
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    predictions = {}
    fallback_count = 0
    for key, output in zip(keys, outputs):
        generated = clean_answer(output.outputs[0].text)
        if args.mode == 'conclusion':
            # A newline usually indicates the model ignored the one-sentence rule.
            conclusion = next((line.strip() for line in generated.splitlines()
                               if line.strip()), '')
            if valid_conclusion(conclusion, data[key]):
                answer = data[key]['rule_answer'].rstrip() + '\n' + conclusion
            else:
                answer = data[key]['rule_answer']
                fallback_count += 1
        else:
            answer = generated
            if len(answer) < args.fallback_min_chars:
                answer = data[key]['rule_answer']
                fallback_count += 1
        predictions[key] = answer

    zip_path = write_submission(args.out, predictions)
    print(f'{len(predictions)} answers -> {zip_path}', flush=True)
    print(f'Rule fallback: {fallback_count}/{len(predictions)}', flush=True)

    gold = {key: row['gold_answer'] for key, row in data.items()
            if 'gold_answer' in row}
    if len(gold) == len(data):
        from qa_predict import score_task2
        rule_predictions = {key: row['rule_answer'] for key, row in data.items()}
        rule_score = score_task2(gold, rule_predictions)
        gen_score = score_task2(gold, predictions)
        print('\n=== SAME RETRIEVAL CONTEXTS ===')
        print(f'Rule METEOR : {rule_score["meteor"]:.4f}')
        print(f'vLLM METEOR : {gen_score["meteor"]:.4f}')
        print(f'Delta       : {gen_score["meteor"] - rule_score["meteor"]:+.4f}')
        print(f'Rule ROUGE-L: {rule_score["rougeL"]:.4f}')
        print(f'vLLM ROUGE-L: {gen_score["rougeL"]:.4f}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
