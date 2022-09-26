import ujson
import os
from glob import glob

import argparse
from datasets import load_metric
import pandas as pd
import nltk
import numpy as np
# import multiprocessing
from collections import defaultdict
from tqdm import tqdm
import itertools

from abstract.eval.bertscore import BertScoreWrapper
from abstract.eval.bartscore import LikelihoodWrapper
from abstract.eval.extractive_fragments import parse_extractive_fragments
from abstract.preprocess.preprocess import linearize_sections
from abstract.corruptions.diverse_decoding import compute_rouge
from abstract.eval.fact_checker import FactChecker


import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

METRICS = ['rouge', 'extractive_fragments', 'bert_score', 'bart_score', 'fact_score']


def df_to_table(df):
    metric_cols = list(sorted([
        'num_prediction_tokens', 'coverage', 'density', 'compression', 'rouge1', 'rouge2', 'rougeL',
        'bs_src_recall', 'bs_src_precision', 'bs_src_f1', 'bs_ref_recall', 'bs_ref_precision', 'bs_ref_f1',
        'bart_score', 'fact_score'
    ]))
    print('Paste into Excel and ensure columns line up')
    print(','.join(metric_cols))
    output_str = []
    for col in metric_cols:
        val = str(round(df[col].dropna().mean(), 4))
        output_str.append(val)
    print(','.join(output_str))


def source_sent_alignment(candidates, comparison):
    compare_set = set(list(map(lambda x: x.lower(), comparison.split(' '))))
    priority = []
    for cand in candidates:
        cand_toks = set(list(map(lambda x: x.lower(), cand.split(' '))))
        overlap = len(cand_toks.intersection(compare_set)) / max(1, len(cand_toks))
        priority.append(-overlap)
    order = np.argsort(priority)
    return list(order)


def remove_eos_bos_from_str(text):
    return text.replace('<s>', ' ').replace('</s>', ' ')


def tokenize(text, lower=True):
    tokens = nltk.word_tokenize(text)
    if lower:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def prepare(record, orig_data, uuid_cache=None, include_tokens=True, include_source=True):
    uuid = record['uuid']
    
    prediction = remove_eos_bos_from_str(record['prediction'])
    prediction_sents = nltk.sent_tokenize(prediction)
    prediction_tokens = None
    if include_tokens:
        try:
            prediction_tokens = tokenize(prediction)
        except:
            print(prediction)
            print(type(prediction), prediction is None)
            print('Error tokenizing prediction.')
            prediction_tokens = prediction.split(' ')

    if uuid_cache is not None and uuid in uuid_cache:
        outputs = uuid_cache.get(uuid).copy()
    else:
        reference = remove_eos_bos_from_str(orig_data['abstract'])
        reference_sents = nltk.sent_tokenize(reference)
        outputs = {
            'reference': reference,
            'reference_sents': reference_sents,
        }
        if include_source:
            source = remove_eos_bos_from_str(linearize_sections(orig_data['sections']))
            source_tokens = tokenize(source) if include_tokens else None
            source_sents = nltk.sent_tokenize(source)
            source_pre = {
                'source': source,
                'source_tokens': source_tokens,
                'source_sents': source_sents,
                'source_sent_alignment': source_sent_alignment(source_sents, prediction),
            }
            outputs.update(source_pre)
    processed = {
        'uuid': uuid,
        'temp_id': record['temp_id'],
        'prediction': prediction,
        'prediction_sents': prediction_sents,
        'prediction_tokens': prediction_tokens,
        'num_prediction_tokens': None if prediction_tokens is None else len(prediction_tokens)
    }

    for k, v in processed.items():
        outputs[k] = v
    return outputs


def _compute_extractive_frags(records, queue=None):
    outputs = []
    for record in tqdm(records, total=len(records), desc='Extractive Fragments'):
        frag_obj = parse_extractive_fragments(record['source_tokens'], record['prediction_tokens'], remove_stop=False)
        frag_obj.pop('fragments')
        row = {'temp_id': record['temp_id'], 'num_prediction_tokens': len(record['prediction_tokens'])}
        row.update(frag_obj)
        outputs.append(row)
    if queue is None:
        return outputs
    queue.put(outputs)
    print('Exiting extractive fragments...')
    exit(0)


def _compute_rouge(records, queue=None):
    rouge_metric = load_metric('rouge')
    outputs = []
    for record in tqdm(records, total=len(records), desc='ROUGE Wrapper'):
        row = {'temp_id': record['temp_id']}
        row.update(compute_rouge(rouge_metric, record['reference'], record['prediction'], rouge_types=['rouge1', 'rouge2']))
        outputs.append(row)
    if queue is None:
        return outputs
    queue.put(outputs)
    print('Exiting ROUGE...')
    exit(0)


def run_in_parallel(records, bartscore_path, uuid2data):
    metric_outputs = []

    print('Preprocessing inputs and outputs...')
    uuid_cache = {}
    eval_inputs = []
    for record in tqdm(records, total=len(records)):
        row = prepare(record, uuid2data[record['uuid']], uuid_cache, include_tokens=True)
        if row['uuid'] not in uuid_cache:
            uuid_cache[row['uuid']] = row
        eval_inputs.append(row)
    print('Done preprocessing...')

    for obj in eval_inputs:
        metric_outputs.append({'temp_id': obj['temp_id'], 'num_prediction_tokens': obj['num_prediction_tokens']})
    
    print('Initializing BERTScore')
    bert_scorer = BertScoreWrapper()
    print('Initializing BartScore')
    bart_scorer = LikelihoodWrapper(hf_model='t5', model_path=bartscore_path)
    print('Initializing FactChecker')
    fact_checker = FactChecker()

    q = mp.Queue()

    processes = [
        mp.Process(target=fact_checker.compute_batch, args=(eval_inputs, q)),
        mp.Process(target=bart_scorer.compute_batch, args=(eval_inputs, q)),
        mp.Process(target=bert_scorer.compute_batch, args=(eval_inputs, q)),
        mp.Process(target=_compute_rouge, args=(eval_inputs, q)),
        mp.Process(target=_compute_extractive_frags, args=(eval_inputs, q)),
    ]

    for prc in processes:
        prc.start()
    for prc in processes:
        prc.join()

    metric_outputs += list(itertools.chain(*[q.get() for _ in range(len(processes))]))
    bart_scorer.cleanup()
    fact_checker.cleanup()

    print('Merging metrics')
    metric_outputs_by_id = defaultdict(dict)
    for metric_output in metric_outputs:
        metric_outputs_by_id[metric_output.pop('temp_id')].update(metric_output)

    for record in records:
        temp_id = record.pop('temp_id')
        for k, v in metric_outputs_by_id[temp_id].items():
            record[k] = v

    return records


def run_single_metric(records, bartscore_path, uuid2data, metric='rouge'):
    metric_outputs = []

    print('Preprocessing inputs and outputs...')
    uuid_cache = {}
    eval_inputs = []
    for record in tqdm(records, total=len(records)):
        row = prepare(record, uuid2data[record['uuid']], uuid_cache, include_tokens=metric=='extractive_fragments', include_source=metric != 'rouge')
        if row['uuid'] not in uuid_cache:
            uuid_cache[row['uuid']] = row
        eval_inputs.append(row)
    print('Done preprocessing...')
    
    if metric == 'bert_score':
        print('Initializing BERTScore')
        bert_scorer = BertScoreWrapper()
        metric_outputs = bert_scorer.compute_batch(eval_inputs)
    elif metric == 'bart_score':
        print('Initializing BartScore')
        bart_scorer = LikelihoodWrapper(hf_model='t5', model_path=bartscore_path)
        metric_outputs = bart_scorer.compute_batch(eval_inputs)
        bart_scorer.cleanup()
    elif metric == 'extractive_fragments':
        metric_outputs = _compute_extractive_frags(eval_inputs)
    elif metric == 'rouge':
        metric_outputs = _compute_rouge(eval_inputs)
    elif metric == 'fact_score':
        print('Initializing FactChecker')
        fact_checker = FactChecker()
        metric_outputs = fact_checker.compute_batch(eval_inputs)
        fact_checker.cleanup()
    else:
        raise Exception(f'Unrecognized metric: {metric}')

    print('Merging metrics')
    metric_outputs_by_id = defaultdict(dict)
    for metric_output in metric_outputs:
        metric_outputs_by_id[metric_output.pop('temp_id')].update(metric_output)

    for record in records:
        temp_id = record.pop('temp_id')
        for k, v in metric_outputs_by_id[temp_id].items():
            record[k] = v

    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Evaluate Abstracts (real and synthetic corruptions)')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--fp', default='weights/primera_final/results/predictions.csv')
    parser.add_argument('--num_chunks', default=3, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('--mode', default='evaluate', choices=['evaluate', 'merge_chunks', 'merge_metrics', 'to_table'])
    parser.add_argument('-erase_after_merge', default=False, action='store_true')
    parser.add_argument('--metric', default=None, choices=METRICS)

    args = parser.parse_args()

    bartscore_path = os.path.join(args.data_dir, 'weights', 'long_t5', 'best_ckpt')
    prediction_fn = os.path.join(args.data_dir, args.fp)

    if args.mode == 'to_table':
        df = pd.read_csv(prediction_fn)
        df_to_table(df)
        exit(0)

    if args.mode == 'merge_chunks':
        in_pattern = prediction_fn.replace('.csv', '') + '_with_metrics_\d_\d.csv'
        fns = list(glob(in_pattern))

        dfs = pd.concat([
            pd.read_csv(fn) for fn in fns
        ])
        out_fn = prediction_fn.replace('.csv', '') + '_with_metrics.csv'
        print(f'Saving merged to {out_fn}')
        dfs.to_csv(out_fn, index=False)
        if args.erase_after_merge:
            for fn in fns:
                os.remove(fn)
        exit(0)
    if args.mode == 'merge_metrics':
        in_pattern = prediction_fn.replace('.csv', '') + '_with_{}.csv'
        dfs = []
        fns = []
        for metric in METRICS:
            fn = in_pattern.format(metric)
            if os.path.exists(fn):
                print(f'Loading {fn}')
                fns.append(fn)
            else:
                print(f'{fn} does not exist.')
        
        dfs = [pd.read_csv(fn) for fn in fns]
        lens = [len(x) for x in dfs]
        assert len(set(lens)) == 1

        merged = dfs[0]
        cols = merged.columns
        merged_predictions = merged['prediction'].tolist()
        for idx in range(1, len(dfs)):
            new_df = dfs[idx]
            new_cols = list(new_df.columns)
            new_cols = [col for col in new_cols if col not in list(merged.columns)]
            new_predictions = new_df['prediction'].tolist()
            assert all([a == b for a, b in zip(merged_predictions, new_predictions)])
            new_col_str = ', '.join(new_cols)
            print(f'Adding {new_col_str} from {fns[idx]}')
            for new_col in new_cols:
                merged[new_col] = new_df[new_col]
        out_fn = prediction_fn.replace('.csv', '') + '_with_metrics.csv'
        print(f'Saving merged to {out_fn}')
        merged.to_csv(out_fn, index=False)
        if args.erase_after_merge:
            for fn in fns:
                print(f'Erasing {fn}')
                os.remove(fn)
        
        df_to_table(merged)
        exit(0)

    in_fn = os.path.join(args.data_dir, 'abstract', 'processed_docs.json')
    print(f'Loading in original data from {in_fn}')
    with open(in_fn, 'r') as fd:
        data = ujson.load(fd)
    
    uuid2data = {}
    for record in data:
        uuid2data[record['uuid']] = record

    print(f'Loading in predictions from {prediction_fn}')
    predict_df = pd.read_csv(prediction_fn).sort_values(by='uuid')
    predict_df.dropna(subset=['prediction', 'uuid'], inplace=True)

    if args.chunk_idx is None:
        chunk_df = predict_df
        chunk_suffix = ''
    else:
        chunk_df = np.array_split(predict_df, args.num_chunks)[args.chunk_idx]
        chunk_suffix = '_' + str(args.chunk_idx) + '_' + str(args.num_chunks)

    chunk_df = chunk_df.assign(temp_id=list(range(len(chunk_df))))
    records = chunk_df.to_dict('records')

    if args.metric is None:
        augmented_records = run_in_parallel(records, bartscore_path=bartscore_path, uuid2data=uuid2data)
    else:
        augmented_records = run_single_metric(records, bartscore_path=bartscore_path, uuid2data=uuid2data, metric=args.metric)
    print('Statistics returned. Storing them in a dataframe with original columns.')
    augmented_df = pd.DataFrame(augmented_records)
    n = len(augmented_df)

    metric_suffix = 'metrics' if args.metric is None else args.metric
    out_fn = prediction_fn.replace('.csv', '') + f'_with_{metric_suffix}{chunk_suffix}.csv'
    if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
        out_dir = os.environ['AMLT_OUTPUT_DIR']
        os.makedirs(out_dir, exist_ok=True)
        out_fn = os.path.join(out_dir, out_fn.split('/')[-1])
    print(f'Saving {n} to {out_fn}')
    augmented_df.to_csv(out_fn, index=False)
