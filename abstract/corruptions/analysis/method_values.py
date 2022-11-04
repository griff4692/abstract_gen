from collections import defaultdict
import os
import ujson

import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from abstract.eval.run import METRIC_COLS

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate')

    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.dataset, 'corruptions', args.split)

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
                for col in METRIC_COLS:
                    method2metrics[method][col].append(corruption[col])

    for method, obj in method2metrics.items():
        print(method)
        for metric, values in obj.items():
            mean_val = np.mean(values)
            print(f'\t{metric}: {mean_val}')
        print('\n')
