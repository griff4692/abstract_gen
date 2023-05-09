import numpy as np
from collections import defaultdict


if __name__ == '__main__':
    with open('rankings.txt', 'r') as fd:
        lines = fd.readlines()
        cts = defaultdict(list)
        for line in lines:
            order = line.split(',')
            for rank, k in enumerate(order):
                cts[k.strip()].append(rank + 1)

    for k, v in cts.items():
        print(k, np.mean(v))
