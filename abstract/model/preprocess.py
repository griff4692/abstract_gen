from collections import defaultdict
import ujson
import os
import regex as re

import argparse
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer, T5Tokenizer
from tqdm import tqdm


T5_MODEL = 'google/long-t5-tglobal-base'
PRIMERA_MODEL = 'allenai/PRIMERA'
DATA_DIR = os.path.expanduser('~/data_tmp')


def linearize_sections(sections):
    out_str = ''
    for section in sections:
        if section['header'] is not None:
            out_str += section['header'] + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', section['body']) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    return out_str.strip()


def process(example):
    return {
        'uuid': example['uuid'],
        'fp': example['fp'],
        'fn': example['fn'],
        'input': linearize_sections(example['sections']),
        'target': example['abstract'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenize with T5 and split into train-validation-test')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_input_length', default=10000, type=int)
    parser.add_argument('--max_target_length', default=1024, type=int)
    parser.add_argument('--model', default='t5', choices=['t5', 'primera'])
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()

    if args.model == 't5':
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
    elif args.model == 'primera':
        args.max_input_length = min(4096, args.max_input_length)
        tokenizer = AutoTokenizer.from_pretrained(PRIMERA_MODEL)
    else:
        raise Exception(f'Unrecognized model -> {args.model}')
    
    print(f'Input length max is {args.max_input_length}')
    out_dir = os.path.join(DATA_DIR, 'abstract', f'{args.model}_splits')
    print(f'Output directory is {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    data_fn = os.path.join(DATA_DIR, 'abstract', 'processed_docs.json')
    print(f'Opening processed data from {data_fn}')
    with open(data_fn, 'r') as fd:
        dataset = ujson.load(fd)

        splits = defaultdict(list)
        for example in tqdm(dataset):
            splits[example['split']].append(process(example))

        train_dataset = Dataset.from_dict(pd.DataFrame(splits['train']), split='train')
        validation_dataset = Dataset.from_dict(pd.DataFrame(splits['validation']), split='validation')
        test_dataset = Dataset.from_dict(pd.DataFrame(splits['test']), split='test')

        print(f'{len(train_dataset)} / {len(validation_dataset)} / {len(test_dataset)} Train / Val / Test splits')

        dataset = DatasetDict(
            {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}
        )

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['target']
        model_inputs = tokenizer(inputs, add_special_tokens=True, max_length=args.max_input_length, padding=False, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, add_special_tokens=True, max_length=args.max_target_length, padding=False, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=['input', 'target'],
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )
    if not args.debug:
        print(f'Saving to {out_dir}')
        dataset.save_to_disk(out_dir)
