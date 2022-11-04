
import os
from glob import glob
from tqdm import tqdm

AMLT_DIR = 'direct-mole'
TMP_DIR = 'span_fills_tmp'

DATA_DIR = os.path.expanduser('~/data_tmp')
FROM_PATTERN = os.path.expanduser(f'~/amlt/{AMLT_DIR}/*/*.csv')
DEST_DIR = os.path.expanduser(f'~/amlt/{AMLT_DIR}/{TMP_DIR}')

os.makedirs(DEST_DIR, exist_ok=True)

matching = list(glob(FROM_PATTERN))

print(len(matching))

for fn in tqdm(matching):
    to_fn = os.path.join(DEST_DIR, fn.split('/')[-1])
    os.rename(fn, to_fn)

print(DEST_DIR)
