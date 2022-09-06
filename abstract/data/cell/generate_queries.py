import os


QUERY = 'https://www.cell.com/action/doSearch?journalCode=chempr&SeriesKey=chempr&rel=nofollow&ContentItemType=fla&pageSize=100&startPage={}'


DATA_DIR = os.path.expanduser('~/data_tmp')
CELL_DIR = os.path.join(DATA_DIR, 'abstract', 'cell')
pdf_dir = os.path.join(CELL_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

urls = []
for page in range(8):
    page_url = QUERY.format(page)
    urls.append(page_url)

search_fn = os.path.join(CELL_DIR, 'search_urls.txt')
print(f'Saving {len(urls)} search page result URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
