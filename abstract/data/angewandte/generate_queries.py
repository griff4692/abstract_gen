QUERY = 'https://onlinelibrary.wiley.com/action/doSearch?ConceptID=15941&PubType=journal&SeriesKey=15213773&sortBy=Earliest&startPage={}&pageSize=803'


import os
from tqdm import tqdm


DATA_DIR = os.path.expanduser('~/data_tmp')
ANG_DIR = os.path.join(DATA_DIR, 'abstract', 'angewandte')
pdf_dir = os.path.join(ANG_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

pages = range(4)
urls = []
for page in tqdm(pages):
    page_url = QUERY.format(page)
    urls.append(page_url)

search_fn = os.path.join(ANG_DIR, 'search_urls.txt')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
