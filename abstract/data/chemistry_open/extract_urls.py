import os
import regex as re


HREF_REGEX = 'href=\"/doi/([^ ]+)\"'
DATA_DIR = os.path.expanduser('~/data_tmp')
OPEN_DIR = os.path.join(DATA_DIR, 'abstract', 'chemistry_open')
pdf_dir = os.path.join(OPEN_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)


if __name__ == '__main__':
    full = open('./search_results/page1.html', 'r').read()
    urls = list(set(list(filter(lambda x: 'open' in x, re.findall(HREF_REGEX, full)))))    
    full_urls = ['https://chemistry-europe.onlinelibrary.wiley.com/doi/pdfdirect/' + url + '?download=true' for url in urls]

    url_fn = os.path.join(OPEN_DIR, 'urls.txt')
    print(f'Saving {len(urls)} urls to {url_fn}')
    with open(url_fn, 'w') as fd:
        fd.write('\n'.join(full_urls))
