import os
import string

import numpy as np
from bs4 import BeautifulSoup
from glob import glob
import argparse
from p_tqdm import p_uimap
import regex as re
import pandas as pd


np.random.seed(1922)


def get_section_name(tag):
    if tag.parent.name == 'ack':
        return 'Acknowledgments'
    
    for child in tag.parent:
        if child.text == tag.text:
            break
        if child.name in {'title', 'head'}:
            return child.text.strip()
    return None


def get_uuid(record, idx=None):
    title = record['title'].replace(string.punctuation, ' ').strip()
    title_prefix = re.sub(r'\s+', '_', title).lower()
    title_prefix = title_prefix[:min(100, len(title_prefix))]
    if idx is None:
        return title_prefix
    return str(idx) + '_' + title_prefix


def is_supplemental(section_name):
    if section_name is None:
        return False
    x = section_name.lower()
    if 'suppl' in x:
        return True
    if 'support' in x:
        return True
    if 'additional' in x:
        return True
    if 'methods' in x:
        return True
    return False


def get_sections(body):
    paragraphs = body.find_all('p')

    # Get rid of paragraphs which have nested paragraphs inside
    paragraph_dedup = [p for p in paragraphs if p.find('p') is None]

    section_names = list(map(get_section_name, paragraph_dedup))
    sup_idxs = [idx for idx, section_name in enumerate(section_names) if is_supplemental(section_name)]

    section_names_trunc = [section_names[i] for i in sup_idxs]
    paragraph_trunc = [paragraph_dedup[i].text.strip() for i in sup_idxs]

    n = len(section_names_trunc)
    if n == 0:
        return None
    return paragraph_trunc


def parse(fn):
    # Reading the data inside the xml file to a variable under the name  data
    with open(fn, 'r') as f:
        data = f.read()

    # Passing the stored data inside the beautifulsoup parser 
    bs_data = BeautifulSoup(data, 'lxml-xml')
    abstract = bs_data.find('abstract')
    body = bs_data.find('body')
    title = bs_data.find('title')

    title_text = ''
    if title is None:
        print(f'Could not locate title for {fn}')
    else:
        title_text = title.text.strip()

    if body is None or abstract is None:
        return None

    sections = get_sections(body)

    if sections is None:
        return None

    suffix = fn.split('/')[-1].split('.')[0]
    return {
        'fp': fn,
        'fn': suffix,
        'title': title_text,
        'abstract': re.sub(r'\s+', ' ', abstract.text.replace('\\n', ' ')).strip(),
        'sections': '<sec>'.join(sections)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process PDFs')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))

    args = parser.parse_args()

    xml_patterns = os.path.join(args.data_dir, '*', 'pdf_out', '*.xml')
    out_fn = os.path.join(args.data_dir, 'processed_docs.json')

    fns = list(glob(xml_patterns))
    print(f'Found {len(fns)} files')

    # processed = list(filter(None, list(map(parse, fns))))
    processed = list(filter(None, list(p_uimap(parse, fns))))

    sup_df = pd.DataFrame(processed)
    out_fn = os.path.join(args.data_dir, 'supplemental_materials.csv')
    print(f'Saving {len(sup_df)} paragraphs to {out_fn}')
    sup_df.to_csv(out_fn, index=False)
