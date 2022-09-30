import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import subprocess


import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import torch
from transformers import RobertaTokenizerFast


from abstract.eval.distinguish.dataset import DistinguishDataModule
from abstract.eval.distinguish.discriminator import Discriminator, HF_TRANSFORMER


DATA_DIR = os.path.expanduser('~/data_tmp')


def get_free_gpus():
    try:
        gpu_stats = subprocess.check_output(
            ['nvidia-smi', '--format=csv,noheader', '--query-gpu=memory.used'], encoding='UTF-8')
        used = list(filter(lambda x: len(x) > 0, gpu_stats.split('\n')))
        return [idx for idx, x in enumerate(used) if int(x.strip().rstrip(' [MiB]')) <= 500]
    except:
        return []


def set_same_seed(seed):
    # Set same random seed for each run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set random, numpy and torch seeds to {seed}')


def run(args):
    if args.gpu_device is not None:
        gpus = [args.gpu_device]
    else:
        gpus = get_free_gpus() if torch.cuda.is_available() and not args.cpu else None
        assert gpus is None or len(gpus) > 0
        if gpus is not None and args.debug:
            gpus = [gpus[0]]
        if gpus is not None and len(gpus) > args.max_gpus:
            gpus = gpus[:args.max_gpus]
        if gpus is not None:
            gpu_str = ','.join([str(x) for x in gpus])
            print(f'Using GPUS --> {gpu_str}...')

    args.num_gpus = None if gpus is None else len(gpus)
    print('Num GPUs --> {}'.format(args.num_gpus))
    precision = 16 if args.num_gpus is not None else 32
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path=HF_TRANSFORMER)

    debug_str = '_mini' if args.debug else ''
    data_fn = os.path.join(DATA_DIR, args.dataset, f'corruptions{debug_str}.csv')
    model = Discriminator(args, tokenizer=tokenizer)
    datamodule = DistinguishDataModule(args, data_fn, tokenizer=tokenizer)

    args.weight_dir = os.path.join(DATA_DIR, args.dataset, 'weights', 'distinguish')
    experiment_dir = os.path.join(args.weight_dir, args.experiment)
    print(f'Experiment Dir -> {experiment_dir}')
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable

    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='distinguish',
        entity='griffinadams',
    )

    primary_eval_metric = 'validation/accuracy'
    primary_metric_mode = 'max'  # Higher is better ('min' for val_loss)
    checkpoint_callback = ModelCheckpoint(
        monitor=primary_eval_metric,
        save_top_k=1,
        save_last=False,
        mode=primary_metric_mode
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]
    plugins = DDPPlugin(find_unused_parameters=False) if args.num_gpus is not None and args.num_gpus > 1 else None
    trainer = pl.Trainer.from_argparse_args(
        args,
        # resume_from_checkpoint=args.restore_path,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        accelerator=None if args.num_gpus is None or args.num_gpus == 1 else 'ddp',
        gpus=gpus,
        default_root_dir=experiment_dir,
        gradient_clip_val=0.1,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=1.0 if args.debug else 0.25,
        num_sanity_val_steps=0 if args.debug else 2,
        log_every_n_steps=25,
        max_steps=args.max_steps,
        plugins=plugins,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Distinguisher.')
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('--num_dataloaders', default=8, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_candidates', type=int, default=5)

    args = parser.parse_args()

    # Set same random seed for each run
    set_same_seed(args.seed)
    run(args)
