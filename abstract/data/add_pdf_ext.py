import os
from glob import glob
from tqdm import tqdm


DATA_DIR = os.path.expanduser('~/data_tmp')


pdf_dir = os.path.join(DATA_DIR, 'abstract', '*', 'pdf')
pattern = os.path.join(pdf_dir, '*')
fns = list(glob(pattern))

changed = 0
for fn in tqdm(fns, desc='Adding .pdf to files'):
    if fn.endswith('.pdf'):
        continue
    changed += 1
    os.rename(fn, fn + '.pdf')

print(f'Added {changed} PDF extensions...')
