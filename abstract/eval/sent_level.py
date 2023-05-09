import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('~/data_tmp/weights/primera_ft_chemistry/results/test_predictions_with_metrics.csv')
    df2 = pd.read_csv('~/data_tmp/weights/primera_ft_pubmed/results/predictions_with_metrics.csv')
    df3 = pd.read_csv('~/data_tmp/weights/primera_ft_clinical/results/predictions_with_metrics.csv')

    factscores = df['fact_score_str'].tolist() + df2['fact_score_str'].tolist() + df3['fact_score_str'].tolist()

    by_len = [
        [] for _ in range(20)
    ]

    for fstring in factscores:
        probs = fstring.split(',')
        for sent_idx, prob in enumerate(probs):
            by_len[min(len(by_len) - 1, sent_idx)].append(float(prob))

    import numpy as np
    for i in range(len(by_len)):
        v = np.mean(by_len[i])
        print(i + 1, str(round(v, 2)))
