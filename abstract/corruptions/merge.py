import os

import argparse
import pandas as pd
import numpy as np

from abstract.preprocess.preprocess import data_loader
from abstract.eval.run import METRIC_COLS

np.random.seed(1992)  # For reproducibility
REMOVE_COLS = ['abstract', 'masked_input', 'input', 'source', 'target']
ENSURE_COLS = ['uuid', 'prediction'] + METRIC_COLS


def load_corruption(fn, method, sign='positive'):
    print(f'Loading {method} corruptions from {fn}')
    df = pd.read_csv(fn)
    for col in ENSURE_COLS:
        assert col in df.columns

    keep_cols = [col for col in df.columns.tolist() if col not in REMOVE_COLS]
    removed_cols = [col for col in df.columns.tolist() if col in REMOVE_COLS]
    removed_str = ', '.join(removed_cols)
    print(f'Removing {len(removed_cols)} columns: {removed_str}')
    df['method'] = method
    df['sign'] = sign
    print(f'Loaded {len(df)} {method} corruptions from {fn}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Combine All Corruptions into a Single DataFrame')
    
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--diverse_experiments', default='primera_ft_pubmed,long_t5_ft_pubmed')
    parser.add_argument('-diverse_decoding', default=False, action='store_true')
    parser.add_argument('-paraphrase', default=False, action='store_true')
    parser.add_argument('-mask_and_fill', default=False, action='store_true')
    parser.add_argument('-entity_swap', default=False, action='store_true')

    args = parser.parse_args()
    args.entity_swap = True

    dataset = data_loader(args.dataset, contrast_subsample=True)

    train_uuids = dataset['train']['uuid']
    train_references = dataset['train']['target']
    validation_uuids = dataset['validation']['uuid']
    validation_references = dataset['validation']['target']
    num_train = len(train_uuids)
    num_val = len(validation_uuids)

    corrupt_dfs = [
        pd.DataFrame(
            {
                'uuid': train_uuids + validation_uuids,
                'prediction': train_references + validation_references,
                'split': ['train' for _ in range(num_train)] + ['validation' for _ in range(num_val)],
                'method': ['reference' for _ in range(num_train + num_val)],
                'sign': ['positive' for _ in range(num_train + num_val)],
            }
        ).drop_duplicates(subset='prediction', keep='first').reset_index(drop=True)
    ]

    mask_and_fill_fn = os.path.join(args.data_dir, args.dataset, 'mask_and_fill', 'span_fills.csv')
    intrinsic_swap_fn = os.path.join(args.data_dir, args.dataset, 'intrinsic_swaps.csv')
    extrinsic_swap_fn = os.path.join(args.data_dir, args.dataset, 'extrinsic_swaps.csv')
    paraphrase_fn = os.path.join(args.data_dir, args.dataset, 'paraphrases.csv')

    num_methods = 0
    if args.mask_and_fill:
        corrupt_dfs.append(load_corruption(mask_and_fill_fn, 'mask_and_fill', sign='negative'))
        num_methods += 1

    if args.diverse_decoding:
        for de in args.diverse_experiments.split(','):
            diverse_decoding_fn = os.path.join(
                args.data_dir, 'weights', de, 'results', 'diverse_decoding_train', 'train_predictions.csv'
            )
            train_corruption = load_corruption(
                diverse_decoding_fn, 'diverse_decoding_' + de, sign='mixed'
            )
            diverse_decoding_fn = os.path.join(
                args.data_dir, 'weights', de, 'results', 'diverse_decoding_validation', 'validation_predictions.csv'
            )
            validation_corruption = load_corruption(
                diverse_decoding_fn, 'diverse_decoding_' + de, sign='mixed'
            )
            full_corruption = pd.concat([train_corruption, validation_corruption])
            corrupt_dfs.append(full_corruption)
            num_methods += 1

    if args.entity_swap:
        corrupt_dfs.append(load_corruption(extrinsic_swap_fn, 'extrinsic_swap', sign='negative'))
        corrupt_dfs.append(load_corruption(intrinsic_swap_fn, 'intrinsic_swap', sign='negative'))
        num_methods += 2

    if args.paraphrase:
        corrupt_dfs.append(load_corruption(paraphrase_fn, 'paraphrase', sign='positive'))
        num_methods += 1

    shared_uuids = set(corrupt_dfs[0]['uuid'])
    print(
        'Searching for common examples among methods '
        '(i.e., some might not have returned answers for each example or be done.'
    )
    for idx in range(1, len(corrupt_dfs)):
        shared_uuids = shared_uuids.intersection(set(corrupt_dfs[idx]['uuid']))

    print(f'Common examples among methods -> {len(shared_uuids)}')
    corrupt_df = pd.concat(corrupt_dfs)
    print(f'Combined {len(corrupt_df)} corruptions for {len(corrupt_df.uuid.unique())} examples')
    print('Filtering for shared UUIDs among methods...')
    corrupt_df = corrupt_df[corrupt_df['uuid'].isin(shared_uuids)]
    print(f'Left with {len(corrupt_df)}')
    corrupt_df = corrupt_df.drop_duplicates(
        subset=['prediction'], keep='first'
    ).sort_values(by='uuid').reset_index(drop=True)
    print(
        f'After removing duplicates, combined {len(corrupt_df)} corruptions for '
        f'{len(corrupt_df.uuid.unique())} examples ({len(corrupt_df)/len(corrupt_df.uuid.unique())})'
    )

    out_fn = os.path.join(args.data_dir, args.dataset, 'corruptions.csv')
    print(f'Saving {len(corrupt_df)} corruptions to {out_fn}')
    corrupt_df.to_csv(out_fn, index=False)

    uuid_sample = set(np.random.choice(corrupt_df['uuid'].unique().tolist(), size=(250, ), replace=False).tolist())
    mini_df = corrupt_df[corrupt_df['uuid'].isin(uuid_sample)]
    mini_fn = os.path.join(args.data_dir, args.dataset, 'corruptions_mini.csv')
    print(f'Saving {len(mini_df)} corruptions to {mini_fn}')
    mini_df.to_csv(mini_fn, index=False)
