import os
import regex as re
import ujson

from bs4 import BeautifulSoup
import argparse
import pandas as pd
from p_tqdm import p_uimap

from abstract.data.parse_xml import get_sections


def parse(record):
    data = record['xml']

    parser = 'lxml-xml' if record['pmc_source'] == 'oa' else 'lxml'

    # Passing the stored data inside the beautifulsoup parser 
    bs_data = BeautifulSoup(data, parser)
    try:
        abstract = bs_data.find('abstract')
        body = bs_data.find('body')
        journal = bs_data.find('journal-title').text
        title = bs_data.find('article-title').text
    except:
        print('Could not parse XML')
        return None

    if body is None or abstract is None:
        return None

    article = bs_data.find('article')
    if article is not None:
        body = article

    sections = get_sections(body)
    
    if sections is None:
        print('Could not parse sections')
        return None

    non_xml_data = {k: v for k, v in record.items() if k != 'xml'}
    fp = record['Article File']

    row = {
        'fp': fp,
        'fn': fp.split('/')[1],
        'journal': journal,
        'title': title,
        'abstract': re.sub(r'\s+', ' ', abstract.text.replace('\\n', ' ')).strip(),
        'sections': sections
    }

    row.update(non_xml_data)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process PDFs')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))

    args = parser.parse_args()

    oa_in_fn = os.path.join(args.data_dir, 'pmc', 'pmc_oa_xml.csv')
    author_in_fn = os.path.join(args.data_dir, 'pmc_author', 'pmc_author_xml.csv')
    out_fn = os.path.join(args.data_dir, 'pmc', 'processed_docs.json')

    print(f'Reading in unprocessed XML records from {oa_in_fn}')
    oa_records = pd.read_csv(oa_in_fn)
    oa_records['pmc_source'] = ['oa'] * len(oa_records)
    print(f'Loaded {len(oa_records)} Open Access PMC articles')

    print(f'Reading in unprocessed XML records from {author_in_fn}')
    author_records = pd.read_csv(author_in_fn)
    author_records['pmc_source'] = 'author_manuscript'
    print(f'Loaded {len(author_records)} Author Manuscript PMC articles')

    records = author_records.to_dict('records') + oa_records.to_dict('records')

    print('Starting to parse XML')
    # processed = list(filter(None, list(map(parse, records))))
    processed = list(filter(None, list(p_uimap(parse, records))))

    print(f'Saving {len(processed)} files (with an extracted abstract and body) to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(processed, fd)
