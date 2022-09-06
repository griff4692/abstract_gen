import os
from glob import glob
import regex as re


DATA_DIR = os.path.expanduser('~/data_tmp')
CSJ_DIR = os.path.join(DATA_DIR, 'abstract', 'csj')
in_dir = os.path.join(CSJ_DIR, 'search_results')
HREF_REGEX = 'href=\"(\S+)\"'

pattern = in_dir + '/*'
url_template = 'https://www.journal.csj.jp{}'

fns = list(glob(pattern))

urls = []
full_text_ct = 0
for fn in fns:
    with open(fn, 'r') as fd:
        html = fd.read()
        hrefs = list(filter(lambda x: 'doi/pdf' in x, re.findall(HREF_REGEX, html)))
        urls.extend([url_template.format(href) for href in hrefs])
        full_text_ct += len(re.findall(r'Full Text', html))

print(full_text_ct)
urls = list(set(urls))
url_fn = os.path.join(CSJ_DIR, 'urls.txt')
print(f'Saving {len(urls)} urls to {url_fn}')
with open(url_fn, 'w') as fd:
    fd.write('\n'.join(urls))
