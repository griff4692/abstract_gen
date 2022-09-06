import os
from tqdm import tqdm


QUERY = 'https://www.beilstein-journals.org/bjoc/advancedSearch?at=Full+Research+Paper&pf=01+Jan+{}&pt=31+Dec+{}&page={}'
YEAR_RANGE = list(range(2005, 2023))

DATA_DIR = os.path.expanduser('~/data_tmp')
BEILSTEIN_DIR = os.path.join(DATA_DIR, 'abstract', 'beilstein')
pdf_dir = os.path.join(BEILSTEIN_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

urls = []
for year in tqdm(YEAR_RANGE):
    for page in range(8):
        page_url = QUERY.format(year, year, page)
        urls.append(page_url)

search_fn = os.path.join(BEILSTEIN_DIR, 'search_urls.txt')
print(f'Saving {len(urls)} search page result URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
