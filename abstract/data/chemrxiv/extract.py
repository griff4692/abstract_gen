import pandas as pd
from abstract.chemrxiv.chemrxiv_wrapper import *
import json
import os


KEEP_COLS = [
    'id',
    'doi',
    'abstract',
    'title',
]

DATA_DIR = os.path.expanduser('~/data_tmp')


def process_item(item):
    output = {
        'url': item['asset']['original']['url'],
        'categories': '|'.join([x['name'] for x in item['categories']]),
        'keywords': '|'.join(item['keywords']),
        'subject': item['subject']['name'],
    }

    for col in KEEP_COLS:
        output[col] = item[col]
    
    return output


if __name__ == '__main__':
    out_dir = os.path.join(DATA_DIR, 'abstract', 'chemrxiv')
    meta_fn = os.path.join(out_dir, 'meta.csv')
    urls_fn = os.path.join(out_dir, 'urls.txt')
    os.makedirs(out_dir, exist_ok=True)
    offset = 0
    data = []

    while True:
        response = chemrxiv_api(
            skip=offset,
            limit=50,
        )

        content = json.loads(response.content)
        items = content['itemHits']
        items = list(map(lambda x: x['item'], items))
        offset += len(items)

        if len(items) == 0:
            break

        for item in items:
            example = process_item(item)
            data.append(example)

        if len(data) % 1000 == 0:
            print(f'Downloaded {len(data)} Chemrxiv paper metadata...')
    data_df = pd.DataFrame(data)
    print(f'Saving {len(data_df)} to {meta_fn}')
    data_df.to_csv(meta_fn, index=False)

    urls = data_df['url'].tolist()
    print(f'Saving URLS to {urls_fn}')
    with open(urls_fn, 'w') as fd:
        fd.write('\n'.join(urls))
