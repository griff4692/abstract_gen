import os
from glob import glob

if __name__ == '__main__':
    fns = list(glob(os.path.expanduser('~/data_tmp/weights/*/*/pytorch_model/*.pt')))

    just_optim = True
    for fn in fns:
        if just_optim and 'optim' not in fn:
            continue

        print(f'Removing {fn}')
        os.remove(fn)