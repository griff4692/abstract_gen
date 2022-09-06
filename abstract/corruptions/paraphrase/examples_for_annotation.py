import os
import ujson

import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Generate Paraphrases of Abstracts')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))
    parser.add_argument('--mode', default='annotations')

    args = parser.parse_args()
    
    in_fn = os.path.join(args.data_dir, 'processed_docs.json')
    para_dir = os.path.join(args.data_dir, 'paraphrase')
    os.makedirs(para_dir, exist_ok=True)
    out_fn = os.path.join(para_dir, 'paraphrase_annotations.txt')

    with open(in_fn, 'r') as fd:
        data = ujson.load(fd)

    validation = [x for x in data if x['split'] == 'validation']
    n = len(validation)
    idxs = np.arange(n)
    np.random.seed(1992)
    np.random.shuffle(idxs)

    abstracts = [validation[idx]['abstract'] for idx in idxs[:10]]

    out_str = ''
    for abstract in abstracts:
        out_str += 'Abstract:\n'
        out_str += abstract
        out_str += '\n\n'
        out_str += 'Paraphrase:\n'
        out_str += '\n\n'

    print(f'Saving paraphrase annotation data to {out_fn}')
    with open(out_fn, 'w') as fd:
        fd.write(out_str)
