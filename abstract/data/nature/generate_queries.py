import os
from tqdm import tqdm


DATA_DIR = os.path.expanduser('~/data_tmp')
NATURE_DIR = os.path.join(DATA_DIR, 'abstract', 'nature_coms')
pdf_dir = os.path.join(NATURE_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

pages = range(1, 31)
urls = []
for page in tqdm(pages):
    page_url = f'https://www.nature.com/commschem/research-articles?page={page}'
    urls.append(page_url)

search_fn = os.path.join(NATURE_DIR, 'search_urls.txt')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
