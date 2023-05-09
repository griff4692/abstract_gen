import os

import numpy as np
np.random.seed(1992)
import pandas as pd
from abstract.preprocess.preprocess import data_loader

DIR = os.path.expanduser(os.path.join('~', 'data_tmp', 'human'))


if __name__ == '__main__':
    num_to_annotate = 50
    experiments = [
        'ft', 'relevance', 'faithful'
    ]

    annotators = ['Bichlien'] * 13 + ['Jake'] * 13 + ['Yuan'] * 12 + ['Kali'] * 12

    chem_test = data_loader('chemistry')['test']
    uuid2inputs = dict(zip(chem_test['uuid'], chem_test['input']))

    fns = list(map(lambda exp: os.path.join(DIR, f'{exp}.csv'), experiments))
    dfs = list(map(pd.read_csv, fns))

    ex2id2pred = {}

    valid_uuids = set()
    for i, (exp, df) in enumerate(zip(experiments, dfs)):
        valid_records = {
            record['uuid']: record['prediction'] for record in df.to_dict('records')
            if record['prediction'].endswith('.')
        }
        vr = set(list(valid_records.keys()))
        valid_uuids = vr if len(valid_uuids) == 0 else valid_uuids.intersection(vr)
        ex2id2pred[exp] = valid_records

    valid_uuids = np.sort(list(valid_uuids))
    uuids_sample = np.random.choice(valid_uuids, size=(num_to_annotate, ), replace=False)

    outputs = []
    oracle_info = []
    for i, uuid in enumerate(uuids_sample):
        if i > 0 and annotators[i - 1] != annotators[i]:
            output_str = '\n\n'.join(outputs)
            with open(f'{annotators[i - 1]}_annotations.txt', 'w') as fd:
                fd.write('\n\n'.join(outputs))
            outputs = []

        exp_order = experiments.copy()
        np.random.shuffle(exp_order)
        oracle_info.append({
            'uuid': uuid,
            'annotation_idx': i,
            'annotator': annotators[i],
            'summary_1': exp_order[0],
            'summary_2': exp_order[1],
            'summary_3': exp_order[2],
        })
        outputs += ['Title ID:' + uuid, 'START OF PAPER', uuid2inputs[uuid], 'END OF PAPER', 'START OF ABSTRACTS']
        for sum_idx, exp in enumerate(exp_order):
            prediction = ex2id2pred[exp][uuid]
            outputs.append(f'ABSTRACT {sum_idx + 1}\n{prediction}')
            outputs.append(f'INTRINSIC ERRORS ABSTRACT {sum_idx + 1}\n<Copy Paste 1 Error Per Line From Abstract>')
            outputs.append(f'EXTRINSIC ERRORS ABSTRACT {sum_idx + 1}\n<Copy Paste 1 Error Per Line From Abstract>')

        outputs.append('END OF ABSTRACTS')
        outputs.append('Relevance Ranking:')
        outputs.append('<Insert Rank Here>')
        outputs.append('-' * 100)

    output_str = '\n\n'.join(outputs)
    with open(f'{annotators[-1]}_annotations.txt', 'w') as fd:
        fd.write('\n\n'.join(outputs))

    oracle_fn = 'oracle_info.csv'
    oracle_df = pd.DataFrame(oracle_info)
    oracle_df.to_csv(oracle_fn, index=False)
