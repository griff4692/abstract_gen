import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('/home/t-gadams/data_tmp/abstract/corruptions_25000_with_metrics.csv')

    single_example = dict(tuple(df.groupby('uuid')))

    # Click short selections
    for uuid, sub_df in single_example.items():
        abstract = sub_df['abstract'].tolist()[0]
        if len(abstract.split(' ')) <= 64:
            print(sub_df['abstract'].tolist())
            for record in sub_df.to_dict('records'):
                print(record['method'])
                print(record['prediction'])
            print('\n\n')
