import os
import ujson

import argparse
import pandas as pd
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to analyze different sampling strategies for calibration')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical')
    parser.add_argument('--metric', default='faithful')
    parser.add_argument('--max_num_rank', default=4, type=int)
    parser.add_argument('--max_examples', default=10000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()
    dummy = AutoTokenizer.from_pretrained('sshleifer/bart-tiny-random')

    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    faith_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
    relevance_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']
    if args.metric == 'faithful':
        strategies = [
            'random', 'max_margin', 'min_margin', 'avg_margin', 'max_diversity', 'min_diversity',
            'easy', 'hard', 'max_extractive_gap'  # Cross diversity?
        ]
        default_metrics = faith_metrics.copy()
    elif args.metric == 'relevance':
        strategies = [
            'random', 'max_margin', 'min_margin', 'max_diversity', 'min_diversity', 'top_beam', 'bottom_beam',
            'wide_beam', 'min_metric', 'max_metric', 'max_gap', 'min_gap',
            'max_surprise', 'min_surprise',
        ]
        default_metrics = relevance_metrics.copy()
    else:
        raise Exception('Unrecognized metric')


    pattern = os.path.join(args.data_dir, args.dataset, 'corruptions', args.split, '*.json')
    print(f'Looking for files matching {pattern}')
    fns = list(glob(pattern))

