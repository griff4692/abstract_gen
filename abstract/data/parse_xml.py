import os
import string

import numpy as np
from bs4 import BeautifulSoup
from glob import glob
import argparse
from p_tqdm import p_uimap
import regex as re
import pandas as pd
import ujson
from tqdm import tqdm


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


def get_sections(body):
    paragraphs = body.find_all('p')

    # Get rid of paragraphs which have nested paragraphs inside
    paragraph_dedup = [p for p in paragraphs if p.find('p') is None]
    abstract_idx = -1
    for idx in range(len(paragraph_dedup)):
        if paragraph_dedup[idx].parent.name == 'abstract' or paragraph_dedup[idx].parent.parent.name == 'abstract':
            abstract_idx = idx

    if abstract_idx == -1:
        print('No abstract paragraph')
    paragraph_post_abstract = paragraph_dedup[abstract_idx + 1:]

    if len(paragraph_post_abstract) == 0:
        return None

    section_names = list(map(get_section_name, paragraph_post_abstract))
    ack_idxs = [idx for idx, section_name in enumerate(section_names) if section_name is not None and 'acknowledgment' in section_name.lower()]

    if len(ack_idxs) == 0:
        end_idx = len(section_names)
        section_names_trunc, paragraph_trunc = section_names, paragraph_post_abstract
    else:
        end_idx = ack_idxs[0]
        if end_idx == 0:
            print(f'Starts with Acknowledgments. Probably an error')
            return None
        section_names_trunc, paragraph_trunc = section_names[:end_idx], paragraph_post_abstract[:end_idx]
    
    sections = []
    curr_section = 'first section'
    curr_body = ''
    for idx, (section, paragraph) in enumerate(zip(section_names_trunc, paragraph_trunc)):
        if curr_section != section and len(curr_body) > 0:
            sections.append({
                'header': curr_section,
                'body': curr_body,
            })

            curr_body = ''
        curr_section = section
        curr_body += f'<p>{paragraph.text.strip()}</p>'
    if len(curr_body) > 0:
        sections.append({
            'header': curr_section,
            'body': curr_body,
        })
    return sections


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
        print('Could not parse sections')
        return None

    suffix = fn.split('/')[-1].split('.')[0]
    return {
        'fp': fn,
        'fn': suffix,
        'title': title_text,
        'abstract': re.sub(r'\s+', ' ', abstract.text.replace('\\n', ' ')).strip(),
        'sections': sections
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process PDFs')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))
    parser.add_argument('--min_target_toks', default=50, type=int)
    parser.add_argument('--min_source_toks', default=500, type=int)

    args = parser.parse_args()

    xml_patterns = os.path.join(args.data_dir, '*', 'pdf_out', '*.xml')
    out_fn = os.path.join(args.data_dir, 'processed_docs.json')

    fns = list(glob(xml_patterns))
    print(f'Found {len(fns)} files')

    processed = list(filter(None, list(p_uimap(parse, fns))))
    # processed = list(filter(None, list(map(parse, fns))))

    # Look for self pre-processed articles
    preprocessed_pattern = os.path.join(args.data_dir, '*', 'processed_docs.json')
    preprocessed_fns = list(glob(preprocessed_pattern))
    print(f'Found {len(preprocessed_fns)} already pre-processed files to add.  Adding them now...')
    for fn in preprocessed_fns:
        with open(fn, 'r') as fd:
            pre = ujson.load(fd)
            print(f'Added {len(pre)} examples from {fn}')
            processed.extend(pre)

    seen_uuids = set()
    seen_abstracts = set()
    unique_processed = []
    duplicates = 0
    for record in tqdm(processed):
        record['uuid'] = get_uuid(record)
        if record['uuid'] in seen_uuids or record['abstract'] in seen_abstracts:
            duplicates += 1
            continue
        unique_processed.append(record)
        seen_uuids.add(record['uuid'])
        seen_abstracts.add(record['abstract'])
    
    np.random.shuffle(unique_processed)
    print(f'Found {duplicates} duplicates (some might be null abstracts). {len(unique_processed)} unique papers.')

    # Compression ratio < 1 (source shorter than target)
    valid_processed = []
    stats = []
    for record in tqdm(unique_processed):
        source_toks = sum([len(x['body'].split(' ')) for x in record['sections']])
        target_toks = len(record['abstract'].split(' '))
        record['source_toks'] = source_toks
        record['target_toks'] = target_toks
        record['compression'] = source_toks / max(1, target_toks)

        if record['compression'] > 1 and target_toks >= args.min_target_toks and source_toks >= args.min_source_toks:
            valid_processed.append(record)
            stats.append({
                'fp': record['fp'],
                'title': record['title'],
                'source_toks': record['source_toks'],
                'target_toks': record['target_toks'],
                'compression': record['compression'],
            })
    print(f'{len(valid_processed)}/{len(unique_processed)} have a source length > target length and not too short abstract length.')

    splits = ['validation'] * 1000 + ['test'] * 2000 + ['train'] * (len(valid_processed) - 3000)
    for split, example in zip(splits, valid_processed):
        example['split'] = split
    print(f'Saving {len(valid_processed)} files (with an extracted abstract and body) to {out_fn}')
    with open(out_fn, 'w') as fd:
        ujson.dump(valid_processed, fd)

    stats = pd.DataFrame(stats)

    print(stats['source_toks'].mean())
    print(stats['target_toks'].mean())
    print(stats['compression'].mean())

    stat_fn = os.path.join(args.data_dir, 'stats.csv')
    stats.to_csv(stat_fn, index=False)
