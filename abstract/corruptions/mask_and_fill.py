from p_tqdm import p_uimap
import ujson
import itertools
from glob import glob

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import stanza
import regex as re
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

# from abstract.corruptions.entity.bern_entities import clean_uuid
def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


HF_MODEL = 'razent/SciFive-base-Pubmed_PMC'
# https://www.dampfkraft.com/penn-treebank-tags.html
KEEP_TAGS = ['NP', 'PP', 'VP']


class MaskFiller:
    def __init__(self, device='cuda:0', num_beams=4) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL).to(device).eval()
        self.device = device
        self.mask_pattern = r'<extra_id_\d+>'
        self.max_length = 1024
        self.num_beams = num_beams
    
    def fill(self, texts, target_length):
        encoding = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        kwargs = {'num_beams': self.num_beams, 'min_length': min(self.max_length, target_length + 3), 'max_length': self.max_length}
        with torch.no_grad():
            preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, **kwargs)
        decoded = self.tokenizer.batch_decode(preds)
        batch_filled = []
        for text, pred in zip(texts, decoded):
            snippets = re.split(self.mask_pattern, pred.replace('<pad>', ''))
            assert len(snippets[0]) == 0
            snippets = snippets[1:]
            unmasked_text = text
            for snippet in snippets:
                mask_location = re.search(r'<extra_id_\d+>', unmasked_text)
                if mask_location is not None:
                    unmasked_text = unmasked_text[:mask_location.start()] + snippet + unmasked_text[mask_location.end():]
            unmasked_text = re.sub(r'\s+', ' ', unmasked_text).strip()
            if re.search(self.mask_pattern, unmasked_text) is not None:
                print('Invalid generation. Could not fully unmask.')
                batch_filled.append(None)
            else:
                batch_filled.append(unmasked_text)
        return batch_filled

    def cleanup(self):
        self.model.cpu()


def extract(tree, target_level, level):
    if target_level != level:
        return []
    if tree.label in KEEP_TAGS:
        return [tree.leaf_labels()]


def get_spans(text, stanza_nlp):
    abstract = stanza_nlp(text)
    all_spans = []

    def add_span(tree):
        if tree.label in KEEP_TAGS:
            all_spans.append({'label': tree.label, 'tokens': tree.leaf_labels()})

    for sentence in abstract.sentences:
        tree = sentence.constituency
        try:
            tree.visit_preorder(internal=add_span, preterminal=add_span, leaf=add_span)
        except ValueError as e:
            print(f'Caught following error from Stanza: {e}')

    valid_spans = []
    for span in all_spans:
        pattern = r'\s*'.join([re.escape(token) for token in span['tokens']])
        match = re.search(pattern, text)

        if match is not None:
            valid_spans.append(span)
    
    for span in valid_spans:
        span['tokens'] = ' '.join(span['tokens'])

    return pd.DataFrame(valid_spans)


def sample_mask(abstract, span_df, target_mask_rate, max_masks=20):
    masked_abstract = abstract
    abstract_toks = len(abstract.split(' '))
    tokens_to_mask = int(round(target_mask_rate * abstract_toks))
    span_df.dropna(inplace=True)
    n = len(span_df)
    priority = np.arange(n)
    np.random.shuffle(priority)
    tokens_masked = 0
    mask_ct = 0
    placeholder_mask = '<mask>'
    for idx in priority:
        span_row = span_df.iloc[idx]
        pattern = r'\s*'.join([re.escape(token) for token in span_row['tokens'].split(' ')])
        num_tokens_in_span = len(span_row['tokens'].split(' '))
        matches = list(re.finditer(pattern, masked_abstract))
        np.random.shuffle(matches)
        if len(matches) > 0:
            match = matches[0]
            tokens_masked += num_tokens_in_span
            mask_ct += 1

            masked_abstract = masked_abstract[:match.start()] + placeholder_mask + masked_abstract[match.end():]
            if tokens_masked > tokens_to_mask or mask_ct >= max_masks:
                break

    for mask_idx in range(mask_ct):
        t5_template = f'<extra_id_{mask_idx}>'
        masked_abstract = masked_abstract.replace(placeholder_mask, t5_template, 1)  # only the first instance

    return masked_abstract, tokens_masked, mask_ct


def build_masks(record, mask_rates):
    outputs = []
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    span_fn = os.path.join(span_dir, f'{uuid_clean}.csv')
    if not os.path.exists(span_fn):
        print(f'No spans for {uuid}.')
        return None
    span_df = pd.read_csv(span_fn)
    for mask_rate in mask_rates:
        for sample_idx in range(args.samples_per_bucket):
            masked, removed_tokens, num_masks = sample_mask(record['abstract'], span_df, target_mask_rate=mask_rate)
            outputs.append({
                'uuid': uuid,
                'target_mask_rate': mask_rate,
                'sample_idx': sample_idx,
                'abstract': record['abstract'],
                'masked_input': masked,
                'removed_tokens': removed_tokens,
                'num_masks': num_masks,
            })
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to Extract, Mask, and Fill Syntactic Spans from References')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp/abstract'))
    parser.add_argument('--mode', default='mask_spans', choices=['extract_spans', 'mask_spans', 'fill_spans', 'merge_chunks'])
    # Extract Span Arguments
    parser.add_argument('-overwrite', default=False, action='store_true')
    # Mask Span Arguments
    parser.add_argument('--mask_rates', default='0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--samples_per_bucket', default=1, type=int)
    # Fill Span Arguments
    parser.add_argument('--batch_size', default=32, type=int)  # Will use cuda:0 by default
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)

    args = parser.parse_args()
    
    in_fn = os.path.join(args.data_dir, 'processed_docs.json')
    span_dir = os.path.join(args.data_dir, 'mask_and_fill', 'spans')
    os.makedirs(span_dir, exist_ok=True)

    records = None
    if args.mode in {'extract_spans', 'mask_spans', 'all'}:
        print(f'Loading dataset from {in_fn}')
        with open(in_fn, 'r') as fd:
            records = ujson.load(fd)

    if args.mode in {'extract_spans', 'all'}:
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

        seen_fns = set()
        duped_uuids = set()
        for record in tqdm(records):
            uuid = record['uuid']
            uuid_clean = clean_uuid(uuid)
            out_fn = os.path.join(span_dir, f'{uuid_clean}.csv')
            if out_fn in seen_fns:
                print(f'{uuid} is duplicated.')
                print(f'{uuid} --> {uuid_clean}.')
                duped_uuids.add(uuid_clean)
            seen_fns.add(out_fn)
            if os.path.exists(out_fn) and not args.overwrite:
                print(f'Skipping {uuid}')
                continue
            span_df = get_spans(record['abstract'], stanza_nlp)
            print(f'Saving {len(span_df)} constituent spans to {out_fn}')
            span_df.to_csv(out_fn, index=False)
        duped_uuids = list(duped_uuids)
        debug_fn = '/home/t-gadams/duped_uuids.txt'
        print(f'Writing {len(duped_uuids)} Duplicated UUIDS to {debug_fn}')
        with open(debug_fn, 'w') as fd:
            fd.write('\n'.join(duped_uuids))
    if args.mode in {'mask_spans', 'all'}:
        mask_rates = list(map(float, args.mask_rates.split(',')))
        outputs = list(p_uimap(lambda record: build_masks(record, mask_rates), records))
        # outputs = list(map(lambda record: build_masks(record, mask_rates), records))
        outputs = [x for x in outputs if x is not None]
        outputs = list(itertools.chain(*outputs))
        outputs = pd.DataFrame(outputs)
        out_fn = os.path.join(args.data_dir, 'mask_and_fill', 'span_masks.csv')
        print(f'Saving {len(outputs)} masked inputs to {out_fn}')
        outputs.to_csv(out_fn, index=False)

        for mask_rate in mask_rates:
            mr = outputs[outputs['target_mask_rate'] == mask_rate]
            removed_tokens = mr['removed_tokens'].dropna().mean()
            num_masks = mr['num_masks'].dropna().mean()
            print(f'Mask Rate {mask_rate}: Remove Tokens={removed_tokens}, Number of Masks={num_masks}')
    if args.mode in {'fill_spans', 'all'}:
        mask_filler = MaskFiller(num_beams=args.num_beams)
        in_fn = os.path.join(args.data_dir, 'mask_and_fill', 'span_masks.csv')
        print(f'Reading in masked abstracts from {in_fn}')
        df = pd.read_csv(in_fn)
        df['target_length'] = df['removed_tokens'] + df['num_masks']
        df.sort_values(by='target_length', inplace=True)
        prev_n = len(df)
        df = df[df['removed_tokens'] >= 1]
        n = len(df)
        empty_ct = prev_n - n
        print(f'{empty_ct} abstracts have no masks. Filtering them out.')
        if args.chunk_idx is None:
            chunk_df = df
            chunk_suffix = ''
        else:
            chunk_df = np.array_split(df, args.num_chunks)[args.chunk_idx]
            chunk_suffix = '_' + str(args.chunk_idx)

        records = chunk_df.to_dict('records')
        n = len(records)

        augmented_records = []
        batch_starts = list(range(0, n, args.batch_size))
        print('Starting to Fill...')
        for s in tqdm(batch_starts, desc=f'Filling in {n} masked abstracts'):
            e = min(n, s + args.batch_size)
            batch = records[s:e]
            batch_inputs = [x['masked_input'] for x in batch]
            target_length = batch[0]['target_length']
            batch_preds = mask_filler.fill(batch_inputs, target_length)

            for pred, row in zip(batch_preds, batch):
                row['prediction'] = pred
                augmented_records.append(row)
            print(f'Saved {len(augmented_records)} records so far.')
        augmented_df = pd.DataFrame(augmented_records)
        augmented_df.dropna(subset=['prediction'], inplace=True)
        augmented_df['num_abstract_tokens'] = augmented_df['abstract'].apply(lambda x: len(x.split(' ')))
        augmented_df['num_prediction_tokens'] = augmented_df['prediction'].apply(lambda x: len(x.split(' ')))

        print('Mean abstract tokens: ', augmented_df['num_abstract_tokens'].mean())
        print('Mean prediction tokens: ', augmented_df['num_prediction_tokens'].mean())
        print('Mean removed tokens: ', augmented_df['removed_tokens'].mean())
        print('Mean masks: ', augmented_df['num_masks'].mean())

        if 'AMLT_OUTPUT_DIR' in os.environ and os.environ['AMLT_OUTPUT_DIR'] is not None:
            out_dir = os.environ['AMLT_OUTPUT_DIR']
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.join(args.data_dir, 'mask_and_fill')
        out_fn = os.path.join(out_dir, f'span_fills{chunk_suffix}.csv')
        print(f'Saving {len(augmented_df)} filled in examples to {out_fn}')
        augmented_df.to_csv(out_fn, index=False)
        mask_filler.cleanup()
    if args.mode in {'merge_chunks', 'all'}:
        chunk_fns = list(glob(os.path.join(args.data_dir, 'mask_and_fill', f'span_fills_*.csv')))
        chunk_fns = [x for x in chunk_fns if any(chr.isdigit() for chr in x)]

        output_df = []
        for fn in tqdm(chunk_fns, desc='Loading disjoint dataset chunks before merging into single dataframe...'):
            chunk_df = pd.read_csv(fn)
            print(f'Adding {len(chunk_df)} examples from {fn}')
            output_df.append(chunk_df)

        output_df = pd.concat(output_df)
        out_fn = os.path.join(args.data_dir, 'mask_and_fill', 'span_fills.csv')
        print(f'Saving {output_df} outputs to {out_fn}')
        # Ensure no duplicates
        uuid = output_df['uuid'] + output_df['target_mask_rate'].astype(str) + output_df['sample_idx'].astype(str)
        print(len(uuid), len(set(uuid)))
        output_df.to_csv(out_fn, index=False)
