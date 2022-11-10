from collections import defaultdict
import os
from glob import glob
import ujson

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from abstract.eval.run import METRIC_COLS


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summary Metric Statistics for Corruptions')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--datasets', default='pubmed,clinical,chemistry')

    args = parser.parse_args()

    out_fn = os.path.join(args.data_dir, 'metric_bounds.csv')

    out_df = []
    for dataset in args.datasets.split(','):
        corruption_dir = os.path.join(args.data_dir, dataset, 'corruptions', 'validation')
        pattern = os.path.join(corruption_dir, '*.json')
        fns = list(glob(pattern))
        metric_vals = defaultdict(list)
        for fn in tqdm(fns):
            with open(fn, 'r') as fd:
                ex = ujson.load(fd)
                for x in ex:
                    for col in METRIC_COLS:
                        metric_vals[col].append(x[col])

            for metric, vals in metric_vals.items():
                avg = np.mean(vals)
                std = np.std(vals)
                min_val, max_val = min(vals), max(vals)
                row = {
                    'dataset': dataset,
                    'metric': metric,
                    'avg': avg,
                    'std': std,
                    'min': min_val,
                    'max': max_val
                }

                out_df.append(row)

    out_df = pd.DataFrame(out_df)
    print(f'Saving metric bounds to {out_fn}')
    out_df.to_csv(out_fn, index=False)
