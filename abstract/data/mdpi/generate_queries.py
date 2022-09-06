import os


QUERY = 'https://www.mdpi.com/search?sort=pubdate&page_count=200&year_from=1996&year_to=2022&article_type=research-article&journal=molecules&view=compact&page_no={}'

DATA_DIR = os.path.expanduser('~/data_tmp')
MDPI_DIR = os.path.join(DATA_DIR, 'abstract', 'mdpi')
pdf_dir = os.path.join(MDPI_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)

urls = []
for page in range(1, 153):
    page_url = QUERY.format(page)
    urls.append(page_url)

search_fn = os.path.join(MDPI_DIR, 'search_urls.txt')
print(f'Saving {len(urls)} search page result URLS to {search_fn}')
with open(search_fn, 'w') as fd:
    fd.write('\n'.join(urls))
