#!/usr/bin/env python3
"""Run feature export, training, evaluation and optional public prediction."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def run(command):
    print('\n$', ' '.join(shlex.quote(x) for x in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def add_retrieval_args(command, cfg):
    mapping = {
        'pool': '--pool', 'alpha': '--alpha', 'beta': '--beta', 'agg': '--agg',
        'prior_lambda': '--prior-lambda', 'ndocs': '--ndocs',
        'mchunks': '--mchunks', 'k1': '--k1', 'b': '--b', 'batch': '--batch',
        'max_length': '--max-length', 'encode_batch': '--encode-batch',
    }
    for key, flag in mapping.items():
        if key in cfg:
            command.extend([flag, str(cfg[key])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', action='append',
                   choices=['export', 'train', 'evaluate', 'public'],
                   help='Lap lai de chon stage; mac dinh export+train+evaluate')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    cfg = json.load(open(args.config, encoding='utf-8'))
    paths = cfg['paths']
    retrieval = cfg.get('retrieval', {})
    training = cfg.get('training', {})
    stages = args.stage or ['export', 'train', 'evaluate']
    artifact_dir = resolve(paths.get(
        'artifacts', 'DocumentReranker-LegalIR/artifacts'))
    os.makedirs(artifact_dir, exist_ok=True)
    train_features = os.path.join(artifact_dir, 'train_features.jsonl')
    val_features = os.path.join(artifact_dir, 'val_features.jsonl')
    public_features = os.path.join(artifact_dir, 'public_features.jsonl')
    model_path = os.path.join(artifact_dir, 'document_ranker.txt')
    metrics_path = os.path.join(artifact_dir, 'metrics.json')
    evaluation_path = os.path.join(artifact_dir, 'evaluation.json')

    def export_command(questions, output, split_name=None):
        cmd = [sys.executable, os.path.join(HERE, 'export_features.py'),
               '--questions', resolve(questions), '--output', output,
               '--prior-train', resolve(paths['train_questions']),
               '--index', resolve(paths['index']), '--emb', resolve(paths['emb']),
               '--reranker', resolve(paths['reranker'])]
        if split_name:
            cmd.extend(['--split', resolve(paths['split']), '--split-name', split_name])
        add_retrieval_args(cmd, retrieval)
        if args.overwrite:
            cmd.append('--overwrite')
        return cmd

    if 'export' in stages:
        run(export_command(paths['train_questions'], train_features, 'train'))
        run(export_command(paths['train_questions'], val_features, 'val'))

    if 'train' in stages:
        cmd = [sys.executable, os.path.join(HERE, 'train_ranker.py'),
               '--train', train_features, '--valid', val_features,
               '--model-out', model_path, '--metrics-out', metrics_path]
        train_flags = {
            'trees': '--trees', 'learning_rate': '--learning-rate',
            'num_leaves': '--num-leaves', 'min_data_in_leaf': '--min-data-in-leaf',
            'feature_fraction': '--feature-fraction',
            'bagging_fraction': '--bagging-fraction',
            'early_stopping': '--early-stopping', 'seed': '--seed',
        }
        for key, flag in train_flags.items():
            if key in training:
                cmd.extend([flag, str(training[key])])
        if args.overwrite:
            cmd.append('--overwrite')
        run(cmd)

    if 'evaluate' in stages:
        cmd = [sys.executable, os.path.join(HERE, 'evaluate.py'),
               '--features', val_features, '--model', model_path,
               '--output', evaluation_path]
        if args.overwrite:
            cmd.append('--overwrite')
        run(cmd)

    if 'public' in stages:
        if not paths.get('public_questions'):
            p.error('config thieu paths.public_questions')
        run(export_command(paths['public_questions'], public_features))
        output = os.path.join(artifact_dir, 'submission_document_ranker.zip')
        cmd = [sys.executable, os.path.join(HERE, 'predict.py'),
               '--features', public_features, '--model', model_path,
               '--method', 'lambdamart', '--output', output]
        if args.overwrite:
            cmd.append('--overwrite')
        run(cmd)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
