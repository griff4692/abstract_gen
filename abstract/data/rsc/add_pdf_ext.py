import os
from glob import glob
from tqdm import tqdm


DATA_DIR = os.path.expanduser('~/data_tmp')


rsc_pdf_dir = os.path.join(DATA_DIR, 'abstract', 'rsc', 'pdf')

pattern = os.path.join(rsc_pdf_dir, '*')

fns = list(glob(pattern))

for fn in tqdm(fns, desc='Adding .pdf to RSC files'):
    os.rename(fn, fn + '.pdf')
