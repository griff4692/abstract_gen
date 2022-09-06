import os
from tqdm import tqdm


DATA_DIR = os.path.expanduser('~/data_tmp')
SR_DIR = os.path.join(DATA_DIR, 'abstract', 'scientific_reports')
pdf_dir = os.path.join(SR_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

pages = range(1, 164)
urls = []
for page in tqdm(pages):
    page_url = f'https://www.nature.com/subjects/chemistry/srep?searchType=journalSearch&page={page}'
    urls.append(page_url)

search_fn = os.path.expanduser(os.path.join(SR_DIR, 'search_urls.txt'))
print(f'Saving {len(urls)} URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
