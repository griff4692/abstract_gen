import os
import regex as re

import numpy as np
import argparse
from p_tqdm import p_uimap
import pandas as pd
import ujson
from nltk import word_tokenize

from abstract.eval.extractive_fragments import parse_extractive_fragments

np.random.seed(1922)


def linearize_sections(sections):
    out_str = ''
    for section in sections:
        if section['header'] is not None:
            out_str += section['header'] + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', section['body']) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    return out_str.strip()


def parse(example):
    source_toks = word_tokenize(linearize_sections(example['sections']))
    target_toks = word_tokenize(example['abstract'])
    row = {
        'fp': example['fp'],
        'source_toks': len(source_toks),
        'target_toks': len(target_toks),
    }

    row.update(parse_extractive_fragments(source_toks, target_toks=target_toks, remove_stop=True))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process PDFs')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))

    args = parser.parse_args()

    in_fn = os.path.join(args.data_dir, 'processed_docs.json')
    print(f'Reading in data from {in_fn}')
    with open(in_fn, 'r') as fd:
        data = ujson.load(fd)

    stats = list(p_uimap(parse, data))
    stats = pd.DataFrame(stats)

    print(stats['source_toks'].mean())
    print(stats['target_toks'].mean())
    print(stats['compression'].mean())
    print(stats['coverage'].mean())
    print(stats['density'].mean())

    stats_fn = os.path.join(args.data_dir, 'stats.csv')
    print(f'Saving stats to {stats_fn}')
    stats.to_csv(stats_fn, index=False)
