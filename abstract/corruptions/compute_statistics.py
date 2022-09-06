import os

import argparse
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


IGNORE_COLS = [
    'num_masks',
    'removed_tokens',
    'target_length',
    'num_swaps',
    'swap_rate',
    'numbers_swapped',
    'ents_swapped',
    'num_abstract_tokens',
    'target_mask_rate',
    'sample_idx',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Abstracts (real and synthetic corruptions)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--fp', default='abstract/corruptions_25000_with_metrics.csv')

    args = parser.parse_args()

    prediction_fn = os.path.join(args.data_dir, args.fp)
    print(f'Loading predictions from {prediction_fn}')
    predict_df = pd.read_csv(prediction_fn).sort_values(by='uuid')
    predict_df.dropna(subset=['prediction', 'uuid'], inplace=True)

    methods = predict_df['method'].unique().tolist()
    colnames_numerics_only = predict_df.select_dtypes(include=np.number).columns.tolist()
    colnames_numerics_only = [col for col in colnames_numerics_only if 'idx' not in col and col not in IGNORE_COLS]
    outputs = {}
    for col in colnames_numerics_only:
        row = {'all': predict_df[col].dropna().mean()}
        for method in methods:
            row[method] = predict_df[predict_df['method'] == method][col].dropna().mean()
        outputs[col] = row
    
    for metric, obj in outputs.items():
        print(metric)
        for k, v in obj.items():
            print(f'- \t{k}: {str(v)}')
        print('\n')
    
    method_to_metrics = []
    for method in methods:
        key = method.replace('_', ' ')
        method_df = predict_df[predict_df['method'] == method]
        row = {'method': key}
        for col in colnames_numerics_only:
            row[col] = method_df[col].dropna().mean()
        method_to_metrics.append(row)
    method_metrics_df = pd.DataFrame(method_to_metrics)

    corel_fn = 'corruption_metrics_correlations.png'
    corel_df = predict_df[colnames_numerics_only]
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corel_df.corr(), annot=True, linewidths=0.5, ax=ax, fmt='.2f', cmap='coolwarm')
    plt.tight_layout()
    print(f'Saving correlation matrix for {len(colnames_numerics_only)} variables to {corel_fn}')
    plt.savefig(corel_fn, bbox_inches='tight')
    fig.clear(True)
