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
    parser.add_argument('--min-coverage', type=float, default=0.85,
                        help='Minimum extractive token coverage ratio against source')
    parser.add_argument('--max-conclusion-chars', type=int, default=300,
                        help='Maximum character length for conclusion sentence')
    parser.add_argument('--backend', default='auto', choices=['auto', 'vllm', 'hf'],
                        help='vllm (fast, GPU only), hf (transformers, supports Apple Silicon MPS/CPU), or auto')
    parser.add_argument('--device', default='auto',
                        help="Compute device for local execution: 'auto', 'cuda', 'mps', or 'cpu'")
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


TOKEN_RE = re.compile(r'\w+', re.UNICODE)
VI_STOPWORDS = {
    'theo', 'đó', 'thì', 'là', 'và', 'của', 'các', 'những', 'cho', 'được',
    'có', 'sẽ', 'phải', 'để', 'với', 'trong', 'khi', 'nếu', 'bị', 'do',
    'tại', 'như', 'về', 'này', 'trên', 'hoặc', 'hay', 'ra', 'vào', 'lại'
}


def valid_conclusion(conclusion, row, min_coverage=0.85, max_chars=300):
    """Reject unsupported numbers, hallucinations, and non-extractive conclusions."""
    if not conclusion.startswith('Theo đó,'):
        return False
    if not 20 <= len(conclusion) <= max_chars:
        return False
    lowered = conclusion.lower()
    if any(marker in lowered for marker in (
            'không đủ thông tin', 'không có đủ thông tin', 'tôi không thể')):
        return False

    source = f'{row["question"]}\n{row["rule_answer"]}'
    source_numbers = set(re.findall(r'\d[\d./-]*', source))
    generated_numbers = set(re.findall(r'\d[\d./-]*', conclusion))
    if not generated_numbers.issubset(source_numbers):
        return False

    source_tokens = set(TOKEN_RE.findall(source.lower()))
    gen_tokens = TOKEN_RE.findall(lowered)
    content_tokens = [t for t in gen_tokens if len(t) >= 2 and t not in VI_STOPWORDS]
    if not content_tokens:
        return False

    covered = sum(1 for t in content_tokens if t in source_tokens)
    coverage = covered / len(content_tokens)
    return coverage >= min_coverage


def generate_hf(prompts, model_name, device, max_tokens, temperature):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'Loading HuggingFace model on {device}: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if device == 'cuda' else (torch.float16 if device == 'mps' else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    outputs = []
    print(f'Generating {len(prompts)} answers via HuggingFace ({device}) ...', flush=True)
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': temperature > 0.0,
                'pad_token_id': tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs['temperature'] = temperature
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        outputs.append(text)
        if (i + 1) % 20 == 0 or (i + 1) == len(prompts):
            print(f'  Generated {i + 1}/{len(prompts)}', flush=True)
    return outputs


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

    device = args.device
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        except ImportError:
            device = 'cpu'

    backend = args.backend
    if backend == 'auto':
        try:
            import vllm
            import torch
            backend = 'vllm' if torch.cuda.is_available() else 'hf'
        except ImportError:
            backend = 'hf'

    keys = list(data)
    message_list = [
        [
            {'role': 'system', 'content': (CONCLUSION_SYSTEM_PROMPT
                                          if args.mode == 'conclusion'
                                          else REWRITE_SYSTEM_PROMPT)},
            {'role': 'user', 'content': build_user_prompt(data[key], args.mode)},
        ]
        for key in keys
    ]

    if backend == 'vllm':
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
        prompts = []
        for messages in message_list:
            try:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            except TypeError:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_tokens,
            repetition_penalty=1.03,
        )
        print(f'Generating {len(prompts)} answers via vLLM ...', flush=True)
        vllm_outputs = llm.generate(prompts, sampling, use_tqdm=True)
        raw_outputs = [out.outputs[0].text for out in vllm_outputs]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = []
        for messages in message_list:
            try:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            except TypeError:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
        raw_outputs = generate_hf(prompts, args.model, device, args.max_tokens, args.temperature)

    predictions = {}
    fallback_count = 0
    for key, output_text in zip(keys, raw_outputs):
        generated = clean_answer(output_text)
        if args.mode == 'conclusion':
            # A newline usually indicates the model ignored the one-sentence rule.
            conclusion = next((line.strip() for line in generated.splitlines()
                               if line.strip()), '')
            if valid_conclusion(conclusion, data[key],
                                min_coverage=args.min_coverage,
                                max_chars=args.max_conclusion_chars):
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
