from collections import defaultdict
import os
import ujson

import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


def score_candidate_fn(row, stats, contrast_metrics):
    norm_vals = []
    for metric in contrast_metrics:
        stat = stats[metric]
        norm_vals.append((row[metric] - stat['mean']) / stat['std'])
    return sum(norm_vals) / len(norm_vals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate')

    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--split', default='train')

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.dataset, 'corruptions', args.split)

    faith_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
    relevance_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']

    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    pattern = os.path.join(out_dir, '*.json')
    fns = list(glob(pattern))
    n = len(fns)
    print(f'Found {n} files matching {pattern}')
    method2metrics = defaultdict(lambda: defaultdict(list))
    for fn in tqdm(fns, total=n):
        with open(fn, 'r') as fd:
            corruptions = ujson.load(fd)
            for corruption in corruptions:
                method = corruption['method']
                if 'swap' in method:
                    method += '_' + str(corruption['swap_rate'])
                elif 'mask_and_fill' in method:
                    method += '_' + str(corruption['target_mask_rate'])
                # rel = score_candidate_fn(corruption, stats, relevance_metrics)
                # faith = score_candidate_fn(corruption, stats, faith_metrics)
                length = corruption['num_prediction_tokens']
                density = corruption['density']
                coverage = corruption['coverage']
                likely = corruption.get('primera_bertscore', corruption.get('primera_bartscore'))

                method2metrics[method]['relevance'].append(corruption['bs_ref_f1'])
                method2metrics[method]['faithful'].append(corruption['fact_score'])
                method2metrics[method]['length'].append(length)
                method2metrics[method]['density'].append(density)
                method2metrics[method]['coverage'].append(coverage)
                method2metrics[method]['likelihood'].append(likely)

    for method, obj in method2metrics.items():
        print(method)
        out_str = [method]
        keys = ['relevance', 'faithful', 'length', 'density', 'coverage', 'likelihood']
        for key in keys:
            mean_val = float(np.mean(obj[key]))
            print(f'\t{key}: {mean_val}')
            out_str.append(str(round(mean_val, 3)))
        print(','.join(out_str))
        print('\n')
