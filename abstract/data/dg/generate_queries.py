import os

QUERY = 'https://www.degruyter.com/search?query=*&startItem=0&sortBy=relevance&documentVisibility=open&subjectFacet=CH%7ECH-06%7ECH-15%7ECH-11%7ELF%7ECH-12%7ELF-03%7ECH-26&documentTypeFacet=article&languageFacet=en&pageSize=100&startItem={}'


DATA_DIR = os.path.expanduser('~/data_tmp')
DG_DIR = os.path.join(DATA_DIR, 'abstract', 'dg')
pdf_dir = os.path.join(DG_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

urls = []
for start in range(0, 45600, 100):
    page_url = QUERY.format(start)
    urls.append(page_url)

search_fn = os.path.join(DG_DIR, 'search_urls.txt')
print(f'Saving {len(urls)} search page result URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
