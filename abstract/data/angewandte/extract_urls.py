import os
import regex as re


HREF_REGEX = 'href=\"/doi/([^ ]+)\"'
DATA_DIR = os.path.expanduser('~/data_tmp')
ANG_DIR = os.path.join(DATA_DIR, 'abstract', 'angewandte')
pdf_dir = os.path.join(ANG_DIR, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)


if __name__ == '__main__':
    p1 = open('search_results/page1.html', 'r').read()
    p2 = open('search_results/page2.html', 'r').read()
    p3 = open('search_results/page3.html', 'r').read()
    p4 = open('search_results/page4.html', 'r').read()

    full = p1 + p2 + p3 + p4

    urls = list(set(list(filter(lambda x: 'anie' in x, re.findall(HREF_REGEX, full)))))
    
    full_urls = ['https://onlinelibrary.wiley.com/doi/pdfdirect/' + url + '?download=true' for url in urls]

    url_fn = os.path.join(ANG_DIR, 'urls.txt')
    print(f'Saving {len(urls)} urls to {url_fn}')
    with open(url_fn, 'w') as fd:
        fd.write('\n'.join(full_urls))
