from random import shuffle
import regex as re
import argparse
import os
from tqdm import tqdm
from p_tqdm import p_uimap
import pandas as pd
from collections import defaultdict
import numpy as np
import ujson
import itertools

from quantulum3 import parser as number_parser


def linearize_sections(sections):
    out_str = ''
    for section in sections:
        if section['header'] is not None:
            out_str += section['header'] + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', section['body']) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    return out_str.strip()


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def swap_numbers(text, target_swap_rate, abstract_numbers, units2numbers):
    corrupted = text
    valid_abstract_numbers = [x for x in abstract_numbers if len(units2numbers[x.unit.name]) > 0 and x.surface in text]
    n = len(valid_abstract_numbers)
    num_swaps = min(n, round(target_swap_rate * n))
    swap_idxs = list(np.random.choice(np.arange(n), size=(num_swaps, ), replace=False))
    numbers_to_swap = [valid_abstract_numbers[i] for i in swap_idxs]
    numbers_to_swap = list(sorted(numbers_to_swap, key=lambda x: -x.span[0]))

    for num in numbers_to_swap:
        cands = units2numbers[num.unit.name]
        shuffle(cands)
        cand = cands[0]
        s, e = min(len(corrupted), num.span[0]), min(len(corrupted), num.span[1])
        corrupted = corrupted[:s] +  cand + corrupted[e:]
    return corrupted, len(numbers_to_swap)


def swap(abstract, target_swap_rate, target_ents, cat2ents):
    corrupted = abstract
    target_to_swap = min(len(target_ents), round(target_swap_rate * len(target_ents)))
    remaining_ents = [x for x in target_ents 
        if type(x['category']) == str and x['category'] in cat2ents
        and type(x['text']) == str and x['text'].strip() in abstract
    ]
    num_swapped = 0
    if len(remaining_ents) == 0:
        # print('No entities corruptable')
        return corrupted, num_swapped
    for _ in range(target_to_swap):
        idx = int(np.random.choice(len(remaining_ents), size=(1, ))[0])
        ent = remaining_ents[idx]
        text_to_remove = ent['text']
        candidates = cat2ents[ent['category']]
        text_to_add = list(np.random.choice(candidates['text'], p=candidates['p'], size=(1, )))[0]
        corrupted = corrupted.replace(text_to_remove.strip(), text_to_add.strip(), 1)
        num_swapped += 1
        remaining_ents = [remaining_ents[i] for i in range(len(remaining_ents)) if i != idx]
        if len(remaining_ents) == 0:
            break
    return corrupted, num_swapped


def perform_swaps(record, out_dir, swap_rates, cat2ents=None):
    outputs = []
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    entity_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    try:
        entities = pd.read_csv(entity_fn)
    except pd.errors.EmptyDataError:
        print(f'Empty DataFrame -> {entity_fn}. Skipping')
        return []
    target_entities = entities[entities['source'] == 'target'].to_dict('records')
    if cat2ents is None:
        source_entities = entities[entities['source'] == 'input'].to_dict('records')
        cat2ents_raw = defaultdict(set)
        for ent in source_entities:
            cat2ents_raw[ent['category']].add(ent['text'])
        cat2ents = {{'text': list(v), 'p': [1/len(v) for _ in range(v)]} for k, v in cat2ents_raw.items()}

    output_meta = {
        'uuid': record['uuid'],
        'input': record['input'],
    }

    linearized_sections = linearize_sections(record['sections'])
    # Too long for number parser (very very slow at large lengths)
    linearized_sections_trunc = linearized_sections[:min(len(linearized_sections), 25000)]
    source_numbers = number_parser.parse(linearized_sections_trunc)

    units2numbers = defaultdict(list)
    for num in source_numbers:
        units2numbers[num.unit.name].append(num.surface)

    for cand_idx in range(args.samples_per_bucket):
        for sample_idx, swap_rate in enumerate(swap_rates):
            row = output_meta.copy()
            corrupted, ents_swapped = swap(record['input'], swap_rate, target_entities, cat2ents)
            abstract_numbers = number_parser.parse(corrupted)
            try:
                corrupted, numbers_swapped = swap_numbers(corrupted, swap_rate, abstract_numbers, units2numbers)
            except:
                numbers_swapped = 0
                print(f'Failed to swap numbers for {uuid}')
            row['sample_idx'] = sample_idx
            row['ents_swapped'] = ents_swapped
            row['swap_rate'] = swap_rate
            row['numbers_swapped'] = numbers_swapped
            row['prediction'] = corrupted
            outputs.append(row)
    return outputs


def is_complete(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    return os.path.exists(out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))
    parser.add_argument('--num_cpus', default=1, type=int)
    parser.add_argument('--swap_rates', default='0.5,1.0')
    parser.add_argument('--samples_per_bucket', default=10, type=int)
    parser.add_argument('-update_existing', default=False, action='store_true')
    parser.add_argument('--dataset', default='chemistry', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--target_errors', default='intrinsic', choices=['intrinsic', 'extrinsic'])

    args = parser.parse_args()
    args.update_existing = True

    args.swap_rates = list(map(float, args.swap_rates.split(',')))

    out_dir = os.path.join(args.data_dir, 'entity')
    os.makedirs(out_dir, exist_ok=True)

    data_fn = os.path.join(args.data_dir, 'processed_docs.json')
    print(f'Loading dataset from {data_fn}')

    with open(data_fn, 'r') as fd:
        data = ujson.load(fd)
        prev_n = len(data)
        print('Filtering for examples with entities...')
        data = list(filter(lambda x: is_complete(x, out_dir), data))
        n = len(data)
        out_fn = os.path.join(args.data_dir, 'entity_number_swaps.csv')
        print(f'Processing {n}/{prev_n} complete records')

        existing_df = None
        if args.update_existing:
            print(f'Loading partial entity swap dataframe from {out_fn}')
            existing_df = pd.read_csv(out_fn)
            done_uuids = set(existing_df['uuid'])
            print('Removing already swapped examples...')
            data = [x for x in data if x['uuid'] not in done_uuids]

        cat2ents = None
        if args.target_errors == 'extrinsic':
            in_fn = os.path.join(args.data_dir, args.dataset, 'entity_inventory.json')
            print(f'Dumping inventory to {out_fn}')
            with open(out_fn, 'r') as fd:
                cat2ents = ujson.load(fd)

        if args.num_cpus > 1:
            outputs = list(itertools.chain(*list(p_uimap(
                lambda record: perform_swaps(record, out_dir, args.swap_rates, cat2ents=cat2ents), data,
                num_cpus=args.num_cpus
            ))))
        else:
            outputs = list(itertools.chain(*list(tqdm(map(
                lambda record: perform_swaps(record, out_dir, args.swap_rates, cat2ents=cat2ents), data),
                total=n
            ))))

        outputs = pd.DataFrame(outputs)
        if existing_df is not None:
            outputs = pd.concat([existing_df, outputs])
        print(f'Saving {len(outputs)} outputs to {out_fn}')
        outputs.to_csv(out_fn, index=False)
