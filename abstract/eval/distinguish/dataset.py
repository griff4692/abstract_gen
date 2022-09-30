from collections import defaultdict
import os
import itertools

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


DATA_DIR = os.path.expanduser('~/data_tmp')


class DistinguishCollate:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert self.max_length <= tokenizer.model_max_length
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        flat_summaries = list(itertools.chain(*[x['candidates'] for x in batch_list]))
        inputs = self.tokenizer(
            flat_summaries,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        methods = [x['methods'] for x in batch_list]
        return {'model_inputs': inputs, 'methods': methods}


class DistinguishDataModule(pl.LightningDataModule):
    def __init__(self, args, data_fn, tokenizer):
        super().__init__()

        print(f'Reading in dataset from {data_fn}')
        raw_data = pd.read_csv(data_fn)

        uuid_fn = os.path.join(DATA_DIR, args.dataset, 'contrast_uuids.csv')
        uuid_df = pd.read_csv(uuid_fn)
        self.split2uuid = defaultdict(set)
        for record in uuid_df.to_dict('records'):
            self.split2uuid[record['split']].add(record['uuid'])

        print(f'Grouping raw data ({len(raw_data)} rows) by UUID')
        self.uuid2records = dict(tuple(raw_data.groupby('uuid')))

        self.tokenizer = tokenizer
        self.debug = args.debug
        self.tokenizer = tokenizer
        self.num_workers = 0 if self.debug else 8
        self.batch_size = args.batch_size
        self.max_candidates = args.max_candidates

    def train_dataloader(self):
        examples = []
        for uuid in self.split2uuid['train']:
            if uuid in self.uuid2records and len(self.uuid2records[uuid]) >= self.max_candidates:
                examples.append({'records': self.uuid2records[uuid].to_dict('records'), 'uuid': uuid})

        train_split = DistinguishDataset(examples, 'train', max_candidates=self.max_candidates)
        collate_fn = DistinguishCollate(self.tokenizer)
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(train_split, **kwargs)

    def val_dataloader(self):
        examples = []
        for uuid in self.split2uuid['validation']:
            if uuid in self.uuid2records and len(self.uuid2records[uuid]) >= self.max_candidates:
                examples.append({'records': self.uuid2records[uuid].to_dict('records'), 'uuid': uuid})
        validation_split = DistinguishDataset(examples, 'validation', max_candidates=self.max_candidates)
        collate_fn = DistinguishCollate(self.tokenizer)
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(validation_split, **kwargs)


class DistinguishDataset(Dataset):
    def __init__(self, examples, split, max_candidates=5):
        super(DistinguishDataset, self).__init__()
        self.examples = examples
        self.split = split
        self.max_candidates = max_candidates

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        records = example['records']
        reference = [x for x in records if x['method'] == 'reference'][0]['prediction']
        corruptions = [x for x in records if x['method'] != 'reference']
        num_c = len(corruptions)
        rand_idx = list(sorted(np.random.choice(
            list(range(num_c)), size=(self.max_candidates,), replace=False
        ).tolist()))
        sampled_corruptions = [corruptions[i] for i in rand_idx]
        candidates = [reference] + [x['prediction'] for x in sampled_corruptions]
        return {'candidates': candidates, 'methods': ['reference'] + [x['method'] for x in sampled_corruptions]}
