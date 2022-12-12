import os
from glob import glob

corrupt_dir = os.path.expanduser('~/data_tmp/clinical/corruptions')
pattern = os.path.join(corrupt_dir, '*', '*.json')
fns = glob(pattern)
fns = [fn for fn in fns if 'invalid' not in fn]
from tqdm import tqdm
import ujson
invalid = 0
valid = 0
for fn in tqdm(fns):
    with open(fn, 'r') as fd:
        obj = ujson.load(fd)
        if 'primera_bertscore' not in obj[0]:
            print(fn)
            invalid += 1
            print(valid, invalid)
        else:
            valid += 1

print(valid, invalid)
