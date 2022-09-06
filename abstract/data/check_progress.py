import os
import pandas as pd
from tqdm import tqdm

main = '/data/abstract/'
tmp = '~/data_tmp/abstract/'

main_dirs = os.listdir(main)
tmp_dirs = os.listdir(tmp)

all_dirs = list(set(main_dirs).union(set(tmp_dirs)))
all_dirs = [x for x in all_dirs if '.zip' not in x and '.json' not in x]

def get_counts(prefix_dir, sub_dir):
    path = os.path.join(prefix_dir, sub_dir)
    if not os.path.exists(path):
        return 0
    else:
        return len(os.listdir(path))

report = []
for source_dir in tqdm(all_dirs):
    row = {'source': source_dir}
    for sub_dir in ['pdf', 'pdf_out']:
        # row['data' + '-' + sub_dir] = get_counts(os.path.join(main, source_dir), sub_dir)
        row['data_tmp' + '-' + sub_dir] = get_counts(os.path.join(tmp, source_dir), sub_dir)
    report.append(row)

report = pd.DataFrame(report)
print(report.head(len(report)))

for col in report.columns:
    if col == 'source':
        continue
    print(col, '-> ', report[col].dropna().sum())
