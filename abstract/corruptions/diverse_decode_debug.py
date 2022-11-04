import argparse
import os
from glob import glob
from collections import defaultdict

from datasets import load_from_disk
import nltk
from p_tqdm import p_uimap
from tqdm import tqdm
import regex as re
import pandas as pd

nltk.download('punkt')
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer,
)

dirname = os.path.dirname(__file__)
DATA_DIR = os.path.expanduser('~/data_tmp')
T5_MODEL = 'google/long-t5-tglobal-base'
PRIMERA_MODEL = 'allenai/PRIMERA'


def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    # rougeLSum expects newline after each sentence
    return ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]


def doit(r):
    decoded_labels = tokenizer.decode(r['label'], skip_special_tokens=True)
    reference_no_space = re.sub(r'[^A-z]+', '', decoded_labels)
    return (reference_no_space, r['uuid'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Fixing UUID mapping issue for diverse decoding.'
    )

    parser.add_argument('--hf_model', default='primera', choices=['primera', 't5'])
    parser.add_argument('--experiment', default='primera_ft_pubmed')  # WandB name
    parser.add_argument('--splits', default='validation,train')
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    args = parser.parse_args()

    weight_dir = os.path.join(DATA_DIR, 'weights')
    experiment_dir = os.path.join(weight_dir, args.experiment)

    # Either PRIMERA (LED) or T5
    is_t5 = args.hf_model.lower() == 't5'
    args.hf_path = T5_MODEL if is_t5 else PRIMERA_MODEL
    tokenizer_constructor = T5Tokenizer if is_t5 else AutoTokenizer
    data_prefix = 't5' if is_t5 else 'primera'
    data_path = os.path.join(DATA_DIR, 'abstract', f'{data_prefix}_splits')
    tokenizer_dir = os.path.join(experiment_dir, 'tokenizer')
    print(f'Loading config from {args.hf_path}')
    config = AutoConfig.from_pretrained(args.hf_path)
    print(f'Loading tokenizer from {tokenizer_dir}')

    tokenizer = tokenizer_constructor.from_pretrained(tokenizer_dir)
    data_fn = os.path.join(DATA_DIR, args.dataset, f'{args.hf_model}_splits')
    dataset = load_from_disk(data_fn)

    contrast_uuids = set()
    if args.dataset != 'clinical':
        contrast_fn = os.path.join(DATA_DIR, args.dataset, 'contrast_uuids.csv')
        contrast_uuids = set(pd.read_csv(contrast_fn)['uuid'])

    for split in args.splits.split(','):
        fix_fn = os.path.join(experiment_dir, 'results', f'diverse_decoding_{split}', f'{split}_predictions_with_metrics.csv')
        fix_fns = glob(os.path.join(experiment_dir, 'results', f'diverse_decoding_{split}', f'{split}_predictions*'))
        fix_str = ','.join(fix_fns)
        print(f'Loading in {fix_str}')
        dfs = [pd.read_csv(fn) for fn in fix_fns]
        # print('All the same rows')

        split_data = dataset[split]
        uuids = split_data['uuid']
        labels = split_data['labels']

        ref2uuid = defaultdict(list)

        examples = []
        for uuid, label in tqdm(zip(uuids, labels), total=len(labels)):
            if split == 'train' and args.dataset != 'clinical' and uuid not in contrast_uuids:
                continue
            examples.append({'uuid': uuid, 'label': label})

        outputs = list(p_uimap(lambda ex: doit(ex), examples))
        # outputs = list(tqdm(map(lambda ex: doit(ex), examples)))
        for a, b in outputs:
            ref2uuid[a].append(b)

        duplicates = 0
        invalid_uuids = set()
        for k, v in ref2uuid.items():
            if len(v) > 1:
                duplicates += len(v)
                print(k)
                for ouch in v:
                    invalid_uuids.add(ouch)
        print(duplicates)
        print(invalid_uuids)

        invalid_fn = os.path.join(DATA_DIR, args.dataset, 'uuids_with_duplicate_references.csv')
        print(f'Saving invalid UUIDs to {invalid_fn}')
        invalid_df = pd.DataFrame({'uuid': list(sorted(list(invalid_uuids)))})
        invalid_df.to_csv(invalid_fn, index=False)

        for fn, df in zip(fix_fns, dfs):
            print(f'Processing {fn}...')
            num_changed = 0
            num_same = 0
            fixed_uuids = []
            for record in tqdm(df.to_dict('records'), total=len(df)):
                target_no_space = re.sub(r'[^A-z]+', '', record['target'])
                if target_no_space not in ref2uuid:
                    uuid = record['uuid']
                    target = record['target']
                    print(f'Could not locate below target in original data for {uuid}. Exiting')
                    print(target)
                    exit(0)
                should_uuid = ref2uuid[target_no_space]
                if len(should_uuid) > 1:
                    print('Duplicate record. Setting UUID to None')
                    new_uuid = None
                else:
                    assert len(should_uuid) == 1
                    new_uuid = should_uuid[0]
                actual_uuid = record['uuid']
                fixed_uuids.append(new_uuid)
                if actual_uuid != new_uuid:
                    num_changed += 1
                else:
                    num_same += 1
            print(f'Changed {num_changed}. Kept {num_same}.')
            df['uuid_fixed'] = fixed_uuids
            print(f'Saving back to {fn}')
            df.to_csv(fn, index=False)
