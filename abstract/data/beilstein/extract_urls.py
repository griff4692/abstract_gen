import os
from glob import glob
import regex as re


DATA_DIR = os.path.expanduser('~/data_tmp')
BEILSTEIN_DIR = os.path.join(DATA_DIR, 'abstract', 'beilstein')
in_dir = os.path.join(BEILSTEIN_DIR, 'search_results')
HREF_REGEX = 'href=\"(\S+)\"'

pattern = in_dir + '/*'
url_template = 'https://www.beilstein-journals.org{}'

fns = list(glob(pattern))

urls = []
for fn in fns:
    with open(fn, 'r') as fd:
        html = fd.read()
        hrefs = list(filter(lambda x: 'pdf' in x, re.findall(HREF_REGEX, html)))
        urls.extend([url_template.format(href) for href in hrefs])

url_fn = os.path.join(BEILSTEIN_DIR, 'urls.txt')
print(f'Saving {len(urls)} urls to {url_fn}')
with open(url_fn, 'w') as fd:
    fd.write('\n'.join(urls))
