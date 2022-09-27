#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import argparse
import itertools
import os
from glob import glob

from datasets import load_from_disk, load_metric
from tqdm import tqdm
import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5Tokenizer,
    LongT5ForConditionalGeneration,
    LEDForConditionalGeneration,
)
import torch
from torch.utils.data import DataLoader


from abstract.preprocess.preprocess import data_loader


# from abstract.model.utils import add_global_attention_mask
def add_global_attention_mask(batch):
    global_attention_mask = torch.zeros_like(batch['input_ids']).to(batch['input_ids'].device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = global_attention_mask


dirname = os.path.dirname(__file__)
DATA_DIR = os.path.expanduser('~/data_tmp')
T5_MODEL = 'google/long-t5-tglobal-base'
PRIMERA_MODEL = 'allenai/PRIMERA'


def compute_rouge(metric, reference, prediction, rouge_types=None):
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2']
    result = metric.compute(references=[reference], predictions=[prediction], use_stemmer=True, rouge_types=rouge_types)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result


def merge_chunks(args, output_dir):
    out_fn = os.path.join(output_dir, f'{args.split}_predictions.csv')
    chunk_pattern = os.path.join(output_dir, 'predictions_*.csv')
    print(f'Searching for files matching {chunk_pattern}...')
    chunk_fns = list(glob(chunk_pattern))
    print(f'Found {len(chunk_fns)} matching files')
    merged = []
    for fn in tqdm(chunk_fns):
        try:
            merged.append(pd.read_csv(fn))
        except:
            print(f'Could not parse file {fn}')
    merged = pd.concat(merged)
    print(f'Saving {len(merged)} outputs to {out_fn}')
    merged.sort_values(by='uuid').reset_index(drop=True).to_csv(out_fn, index=False)


def generate(args, experiment_dir, output_dir, verbose=True):
    # Either PRIMERA (LED) or T5
    is_t5 = args.hf_model.lower() == 't5'
    args.hf_path = T5_MODEL if is_t5 else PRIMERA_MODEL
    model_constructor = LongT5ForConditionalGeneration if is_t5 else LEDForConditionalGeneration
    tokenizer_constructor = T5Tokenizer if is_t5 else AutoTokenizer
    args.max_source_length = 16384 if is_t5 else 4096
    args.max_target_length = 1024
    data_prefix = 't5' if is_t5 else 'primera'
    data_path = os.path.join(DATA_DIR, 'abstract', f'{data_prefix}_splits')

    if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
        singularity_out = os.environ['AMLT_OUTPUT_DIR']
        print(f'Running on singularity. Saving results to {singularity_out} instead of {output_dir}')
        output_dir = os.environ['AMLT_OUTPUT_DIR']
        os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = os.path.join(experiment_dir, 'best_ckpt')
    tokenizer_dir = os.path.join(experiment_dir, 'tokenizer')

    print(f'Loading config from {args.hf_path}')
    config = AutoConfig.from_pretrained(args.hf_path)
    print(f'Loading tokenizer from {tokenizer_dir}')

    tokenizer = tokenizer_constructor.from_pretrained(tokenizer_dir)

    config.vocab_size = len(tokenizer)
    print(f'Loading model from {ckpt_dir}')

    model = model_constructor.from_pretrained(ckpt_dir, from_tf=False, config=config).to(args.device).eval()
    model.resize_token_embeddings(len(tokenizer))
    if args.hf_model == 'primera':
        model = model.half()

    print(f'Loading custom dataset from {data_path}')
    train_dataset = data_loader(args.dataset, contrast_subsample=True)[args.split]

    if len(train_dataset) > args.max_examples:
        np.random.seed(1992)
        train_idxs = np.arange(len(train_dataset))
        np.random.shuffle(train_idxs)
        print(f'Subsampling {args.max_examples}/len(train_dataset)')
        idx_to_keep = train_idxs[:args.max_examples]
        train_subset = train_dataset.select(idx_to_keep)
    else:
        train_subset = train_dataset

    chunk_suffix = ''
    if args.chunk_idx is not None:
        assert args.chunk_idx < args.num_chunks
        data_idxs = np.arange(len(train_subset))
        chunk_idxs = np.array_split(data_idxs, args.num_chunks)[args.chunk_idx]
        train_subset = train_subset.select(chunk_idxs)
        chunk_suffix = '_' + str(args.chunk_idx)

    dataset_cols = list(train_subset.features.keys())
    important_cols = [x for x in dataset_cols if x not in {'input_ids', 'attention_mask', 'labels'}]

    # Data collator
    if is_t5:
        pad_multiple = None
    else:
        pad_multiple = max(model.config.attention_window)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=pad_multiple,
    )

    dataloader = DataLoader(
        train_subset.remove_columns(important_cols),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=8
    )

    # Metric
    metric = load_metric('rouge')

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        # rougeLSum expects newline after each sentence
        return ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]

    print('Starting to evaluate run...')
    gen_kwargs = {
        'no_repeat_ngram_size': 3, 'early_stopping': True, 'max_length': 1024,
        # Triggers Diverse Beam Search
        'num_beam_groups': args.num_candidates,
        'num_beams': args.num_candidates,
        'num_return_sequences': args.num_candidates,
        'diversity_penalty': 1.0,  # what to subtract from logits for duplicated words
    }

    data_offset = 0
    num_complete = 0
    uuids = train_subset['uuid']
    n = len(uuids)
    if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
        singularity_out = os.environ['AMLT_OUTPUT_DIR']
        print(f'Running on singularity. Saving results to {singularity_out} instead of {output_dir}')
        output_dir = os.environ['AMLT_OUTPUT_DIR']
        os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch_suffix = '_' + str(batch_idx)
        out_fn = os.path.join(output_dir, f'predictions{chunk_suffix}{batch_suffix}.csv')
        batch_outputs = []

        if os.path.exists(out_fn):
            print(f'Already done {out_fn} Skipping.')
            continue

        if args.hf_model == 'primera':
            add_global_attention_mask(batch)
            gen_kwargs['global_attention_mask'] = batch['global_attention_mask'].to(args.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_tokens = model.generate(
                batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                **gen_kwargs,
            ).cpu().numpy()

            labels = batch['labels'].numpy()
            batch_size = len(batch['input_ids'])
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_labels_ext = []
            for label in decoded_labels:
                decoded_labels_ext += [label] * args.num_candidates

            prepared_preds = postprocess_text(decoded_preds)
            references = postprocess_text(decoded_labels_ext)

            data_idxs = list(itertools.chain(*[[idx] * args.num_candidates for idx in range(data_offset, data_offset + batch_size)]))
            uuids_ext = [uuids[idx] for idx in data_idxs]
            data_offset += batch_size
            sample_idxs = []
            for _ in range(batch_size):
                sample_idxs += list(range(args.num_candidates))
                num_complete += 1                    
                if num_complete % 100 == 0:
                    print(f'Completed {num_complete} / {n}')

            if verbose:
                print(f'Completed {num_complete} / {n}')

            assert len(decoded_preds) == len(decoded_labels_ext) == len(prepared_preds) == len(references) == len(uuids_ext)
            for clean_prediction, clean_label, prediction, reference, uuid, sample_idx in zip(
                decoded_preds, decoded_labels_ext, prepared_preds, references, uuids_ext, sample_idxs
                ):
                output_row = {'prediction': clean_prediction, 'abstract': clean_label, 'uuid': uuid, 'sample_idx': sample_idx}
                rouge_obj = compute_rouge(metric, reference=reference, prediction=prediction)
                output_row.update(rouge_obj)
                batch_outputs.append(output_row)

            batch_outputs = pd.DataFrame(batch_outputs)
            if verbose:
                print(f'Saving {len(batch_outputs)} outputs to {out_fn}')
            batch_outputs.to_csv(out_fn, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Over-generating diverse candidates for summarization models to support contrastive learning.')

    parser.add_argument('--hf_model', default='primera', choices=['primera', 't5'])
    parser.add_argument('--experiment', default='primera')  # WandB name
    parser.add_argument('--num_candidates', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--chunk_idx', default=0, type=int)
    parser.add_argument('--num_chunks', default=5, type=int)
    parser.add_argument('--max_examples', default=200000, type=int)
    parser.add_argument('--split', default='train')

    parser.add_argument('--mode', default='generate', choices=['merge_chunks', 'generate'])

    args = parser.parse_args()

    weight_dir = os.path.join(DATA_DIR, 'weights')
    experiment_dir = os.path.join(weight_dir, args.experiment)
    output_dir = os.path.join(experiment_dir, 'results', 'diverse_decoding')
    if args.split == 'validation':
        output_dir = os.path.join(experiment_dir, 'results', 'diverse_decoding_validation')
        args.num_chunks = 1
        args.chunk_idx = 0  # Only 1000 examples
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving all outputs to {output_dir}')

    if args.mode == 'merge_chunks':
        merge_chunks(args, output_dir)
    else:
        assert args.mode == 'generate'
        generate(args, experiment_dir, output_dir)
