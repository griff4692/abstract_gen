import os


QUERY = 'https://www.journal.csj.jp/topic/csj-oa?SeriesKey=bcsj&pageSize=100&ContentItemType=research-article&AfterYear={}&BeforeYear={}&startPage={}'
YEAR_RANGE = list(range(1926, 2023))

DATA_DIR = os.path.expanduser('~/data_tmp')
CSJ_DIR = os.path.join(DATA_DIR, 'abstract', 'csj')
pdf_dir = os.path.join(CSJ_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

urls = []
for year in YEAR_RANGE:
    for page in range(5):
        page_url = QUERY.format(year, year + 1, page)
        urls.append(page_url)

search_fn = os.path.join(CSJ_DIR, 'search_urls.txt')
print(f'Saving {len(urls)} search page result URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
