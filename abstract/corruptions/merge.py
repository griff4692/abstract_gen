import os

import argparse
import pandas as pd
import numpy as np
import ujson

np.random.seed(1992)  # For reproducibility


def load_corruption(fn, method):
    print(f'Loading {method} corruptions from {fn}')
    df = pd.read_csv(fn)
    df['method'] = method
    print(f'Loaded {len(df)} {method} corruptions from {fn}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Combine All Corruptions into a single DataFrame')
    
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--diverse_experiments', default='primera,long_t5')
    parser.add_argument('-diverse_decoding', default=False, action='store_true')
    parser.add_argument('-paraphrase', default=False, action='store_true')
    parser.add_argument('-mask_and_fill', default=False, action='store_true')
    parser.add_argument('-entity_swap', default=False, action='store_true')
    parser.add_argument('--max_examples', default=10000, type=int)
    parser.add_argument('--min_count', default=5, type=int)  # At least an average of 3 examples for each method

    args = parser.parse_args()
    
    in_fn = os.path.join(args.data_dir, 'abstract', 'processed_docs.json')
    with open(in_fn, 'r') as fd:
        dataset = ujson.load(fd)
    
    validation_uuids = set([x['uuid'] for x in dataset if x['split'] == 'validation'])
    test_uuids = set([x['uuid'] for x in dataset if x['split'] == 'test'])
    train_uuids = set([x['uuid'] for x in dataset if x['split'] == 'train'])

    span_dir = os.path.join(args.data_dir, 'spans')
    os.makedirs(span_dir, exist_ok=True)

    mask_and_fill_fn = os.path.join(args.data_dir, 'abstract', 'mask_and_fill', 'span_fills.csv')
    entity_swap_fn = os.path.join(args.data_dir, 'abstract', 'entity_number_swaps.csv')

    corrupt_dfs = []

    num_methods = 0
    if args.mask_and_fill:
        corrupt_dfs.append(load_corruption(mask_and_fill_fn, 'mask_and_fill'))
        num_methods += 1

    if args.diverse_decoding:
        for de in args.diverse_experiments.split(','):
            diverse_decoding_fn = os.path.join(args.data_dir, 'weights', de, 'results', 'diverse_decoding', 'train_predictions.csv')
            train_corruption = load_corruption(diverse_decoding_fn, 'diverse_decoding_' + de)
            diverse_decoding_fn = os.path.join(args.data_dir, 'weights', de, 'results', 'diverse_decoding_validation', 'validation_predictions.csv')
            validation_corruption = load_corruption(diverse_decoding_fn, 'diverse_decoding_' + de)
            full_corruption = pd.concat([train_corruption, validation_corruption])
            corrupt_dfs.append(full_corruption)
            num_methods += 1

    if args.entity_swap:
        corrupt_dfs.append(load_corruption(entity_swap_fn, 'entity_swap'))
        num_methods += 1

    if args.paraphrase:
        raise Exception('Not implemented Yet!')

    shared_uuids = set(corrupt_dfs[0]['uuid'])
    print('Searching for common examples among methods (i.e., some might not have returned answers for each example or be done.')
    for idx in range(1, len(corrupt_dfs)):
        shared_uuids = shared_uuids.intersection(set(corrupt_dfs[idx]['uuid']))

    print(f'Common examples among methods -> {len(shared_uuids)}')
    corrupt_df = pd.concat(corrupt_dfs)
    print(f'Combined {len(corrupt_df)} corruptions for {len(corrupt_df.uuid.unique())} examples')
    print('Filtering for shared UUIDs among methods...')
    corrupt_df = corrupt_df[corrupt_df['uuid'].isin(shared_uuids)]
    print(f'Left with {len(corrupt_df)}')
    corrupt_df = corrupt_df.drop_duplicates(subset=['prediction']).sort_values(by='uuid').reset_index(drop=True)
    print(f'After removing duplicates, combined {len(corrupt_df)} corruptions for {len(corrupt_df.uuid.unique())} examples ({len(corrupt_df)/len(corrupt_df.uuid.unique())})')
    uuid_counts = corrupt_df['uuid'].value_counts().to_dict()

    uuids = list(uuid_counts.keys())
    adj_min_count = args.min_count * num_methods
    valid_uuids = [uuid for uuid in uuids if uuid_counts[uuid] >= adj_min_count]

    print(f'{len(valid_uuids)}/{len(uuids)} examples have enough corruptions ({adj_min_count})')
    valid_train_uuids = [uuid for uuid in valid_uuids if uuid in train_uuids]
    valid_validation_uuids = set([uuid for uuid in valid_uuids if uuid in validation_uuids])
    n = len(valid_train_uuids)
    sampled_train_uuids = valid_train_uuids

    if n > args.max_examples:
        print(f'Sampling {args.max_examples} UUIds')
        sampled_idxs = list(sorted(list(np.random.choice(np.arange(n), size=(args.max_examples,), replace=False))))
        sampled_train_uuids = [valid_uuids[i] for i in sampled_idxs]

    # Keep all the validation corruptions.  Keep the down-sampled train ones. NO contrastive learning for evaluation (test set).
    print(f'Using {len(sampled_train_uuids)} from train set, {len(valid_validation_uuids)} from validation.')
    keep_uuids = valid_validation_uuids.union(sampled_train_uuids)
    corrupt_df = corrupt_df[corrupt_df['uuid'].isin(set(keep_uuids))]
    out_fn = os.path.join(args.data_dir, 'abstract', f'corruptions_{args.max_examples}.csv')
    print(f'Saving {len(corrupt_df)} corruptions to {out_fn}')
    corrupt_df.to_csv(out_fn, index=False)
