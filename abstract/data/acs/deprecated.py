import os

from tqdm import tqdm


OUT_DIR = '~/data_tmp/abstract/acs/pdf/'
URL_FN = '~/data_tmp/abstract/acs/urls.txt'
os.makedirs(OUT_DIR, exist_ok=True)


with open(URL_FN, 'r') as fd:
    urls = fd.readlines()
    urls = [x.strip() for x in urls if len(x.strip()) > 0]
    cmds = []
    for url in tqdm(urls):
        out_fn = OUT_DIR + url.split('/')[-1] + '.pdf'
        cmd = f'curl "{url.strip()}" -o {out_fn}'
        cmds.append(cmd)

    with open('/root/lca/abstract/acs/fetch_pdfs.sh', 'w') as fd:
        fd.write('\n'.join(cmds))
