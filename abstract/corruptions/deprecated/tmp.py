import pandas as pd


fns = [
    # '/home/ga2530/data_tmp/weights/primera_ft_clinical/results/diverse_decoding_{}/{}_predictions.csv',
    # '/home/ga2530/data_tmp/weights/long_t5_ft_clinical/results/diverse_decoding_{}/{}_predictions.csv',
    # '/home/ga2530/data_tmp/weights/long_t5_ft_chemistry/results/diverse_decoding_{}/{}_predictions.csv',
    # '/home/ga2530/data_tmp/weights/primera_ft_chemistry/results/diverse_decoding_{}/{}_predictions.csv',
    '/home/ga2530/data_tmp/weights/long_t5_ft_pubmed/results/diverse_decoding_{}/{}_predictions.csv',
    '/home/ga2530/data_tmp/weights/primera_ft_pubmed/results/diverse_decoding_{}/{}_predictions.csv',
]

# long_t5_ft_chemistry, train, (20) (128)
# primera_ft_pubmed, validation (20) (4)
# long_t5_ft_pubmed, train, (20) (17,288)
# primera_ft_clinical, train, (30) (15,384 > 10)
# long_t5_ft_clinical, train (30) (13,676 > 10)

for fn in fns:
    model = fn.split('/')[5]
    for split in ['train', 'validation']:
        act_fn = fn.format(split, split)
        df = pd.read_csv(act_fn)
        uuid = df['uuid_fixed'].value_counts()
        min_ct, max_ct = uuid.min(), uuid.max()
        num_greater = len([x for x in uuid.tolist() if x > 10])
        print(f'{model} {split}, {len(uuid)}, {num_greater}, {min_ct}, {max_ct}')
