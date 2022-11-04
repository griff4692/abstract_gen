import os
from glob import glob

import argparse
import torch
import ujson
from transformers import RobertaTokenizerFast
from tqdm import tqdm

from abstract.eval.distinguish.discriminator import Discriminator

HF_TRANSFORMER = os.path.expanduser('~/RoBERTa-base-PM-M3-Voc-distill-align-hf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Abstracts (real and synthetic corruptions)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='clinical', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--experiment', default='the_{}_distinguisher')
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-overwrite', action='store_true', default=False)

    args = parser.parse_args()

    weight_dir = os.path.join(args.data_dir, args.dataset, 'weights', 'distinguish')
    full_weight_dir = os.path.join(
        weight_dir, args.experiment.format(args.dataset), 'distinguish', '*', 'checkpoints', '*.ckpt'
    )
    pattern = glob(full_weight_dir)
    if len(pattern) == 0:
        print(f'No weights found matching {pattern}')
        exit(1)
    elif len(pattern) > 1:
        print('Multiple weights.')
        print(pattern)
        exit(1)

    pretrained_fn = pattern[0]
    print(f'Loading in weights -> {pretrained_fn}')
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path=HF_TRANSFORMER)
    model = Discriminator.load_from_checkpoint(pretrained_fn, tokenizer=tokenizer).eval().to(args.device)

    corrupt_dir = os.path.join(args.data_dir, args.dataset, 'corruptions')
    print(f'Reading in data to annotate with distinguish labels from {corrupt_dir}')
    pattern = os.path.join(corrupt_dir, '*', '*.json')
    fns = glob(pattern)
    print(f'Found {len(fns)} files matching {pattern}')
    for fn in tqdm(fns):
        with open(fn, 'r') as fd:
            records = ujson.load(fd)

        if 'turing_score' in records[0] and not args.overwrite:
            print(f'Already Done! Skipping {fn}...')

        predictions = [x['prediction'] for x in records]
        inputs = tokenizer(
            predictions,
            padding='longest',
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        )

        n = len(predictions)
        batch_starts = list(range(0, n, args.batch_size))

        logits = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + args.batch_size, n)
            batch_inputs = {k: v[batch_start:batch_end].to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_logits = model.generate_logits(batch_inputs).detach().cpu().numpy().tolist()
                logits += batch_logits
        logits = [float(x) for x in logits]

        for logit, record in zip(logits, records):
            record['turing_score'] = logit

        with open(fn, 'w') as fd:
            ujson.dump(records, fd)
