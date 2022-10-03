import itertools
import regex as re
import os
from time import sleep
CWD = os.path.dirname(__file__)

import openai
import argparse
import pandas as pd
import ujson
from tqdm import tqdm
import numpy as np
from random import choice

openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)
assert len(openai.api_key) > 0

from abstract.preprocess.preprocess import data_loader


def build_prompt(abstract, annotated_abstracts):
    prompt = 'Paraphrase this abstract.\n\n'
    for orig, human in annotated_abstracts:
        prompt += orig.strip() + '=>' + choice(human).strip() + '\n\n'
    prompt += abstract + '=>'
    return prompt


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def is_incomplete(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    return not os.path.exists(out_fn)


def paraphrase_with_gpt(args, record, annotated_abstracts):
    few_shot_examples = list(itertools.combinations(list(range(len(annotated_abstracts))), args.few_shot_n))
    sampled_example_set = [annotated_abstracts[i] for i in few_shot_examples[choice(range(len(few_shot_examples)))]]
    prompt = build_prompt(record['abstract'], sampled_example_set)

    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=prompt,
        temperature=0.7,
        max_tokens=args.max_tokens,
        top_p=1,
        n=args.num_candidates,
        frequency_penalty=0,
        presence_penalty=0
    )

    return [x['text'] for x in response['choices']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--few_shot_n', default=1, type=int)
    parser.add_argument('--num_candidates', default=5, type=int)
    parser.add_argument('--max_tokens', default=512, type=int)

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.dataset, 'paraphrase', 'gpt')
    os.makedirs(out_dir, exist_ok=True)

    dataset = data_loader(args.dataset, contrast_subsample=True)
    annotations_fn = os.path.join(CWD, 'annotations.txt')
    with open(annotations_fn, 'r') as fd:
        paraphrase_annotations = fd.readlines()

    paraphrase_annotation_tuples = []
    for idx in range(len(paraphrase_annotations)):
        if paraphrase_annotations[idx].startswith('Abstract:'):
            paraphrase_annotation_tuples.append([paraphrase_annotations[idx].replace('Abstract:', ''), []])
        else:
            assert 'Paraphrase:' in paraphrase_annotations[idx]
            paraphrase_annotation_tuples[-1][1].append(paraphrase_annotations[idx].replace('Paraphrase:', ''))

    for split, data in dataset.items():
        prev_n = len(data)
        if not args.overwrite:
            print('Filtering out already done examples...')
            data = list(filter(lambda x: is_incomplete(x, out_dir), data))

        for record in tqdm(data):
            uuid = record['uuid']
            uuid_clean = clean_uuid(uuid)
            out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
            try:
                paraphrases = paraphrase_with_gpt(args, record, paraphrase_annotation_tuples)
            except openai.error.RateLimitError:
                print('Rate limit exceeded. Sleeping for a minute and re-trying.')
                sleep(60)
                paraphrases = paraphrase_with_gpt(args, record, paraphrase_annotation_tuples)
            except openai.error.InvalidRequestError as e:
                print(e)
                print('Skipping for now.')
                continue

            output_df = pd.DataFrame([
                {
                    'uuid': record['uuid'], 'split': split, 'abstract': record['abstract'],
                    'prediction': p, 'paraphrase_idx': i
                } for i, p in enumerate(paraphrases)
            ])
            output_df.to_csv(out_fn, index=False)
