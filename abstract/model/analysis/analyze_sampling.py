import os
import pandas as pd
import ujson
from tqdm import tqdm
from glob import glob
import argparse
from collections import defaultdict
import numpy as np
from collections import Counter
from abstract.eval.diversity import diversity_score
from p_tqdm import p_uimap

from transformers import DataCollatorForContrastSeq2Seq


def cross_diversity(arr_a, arr_b):
    cross_divs = []
    for a in range(len(arr_a)):
        max_div = max([diversity_score([arr_a[a], arr_b[b]]) for b in range(len(arr_b))])
        cross_divs.append(max_div)
    return float(np.mean(cross_divs))


def record(args, fn):
    stats_by_method = defaultdict(lambda: defaultdict(list))
    with open(fn, 'r') as fd:
        cset = ujson.load(fd)
        cset_filt = []
        seen = set()
        for cs in cset:
            cs['prediction'] = cs['prediction'].strip()
            if cs['prediction'] in seen:
                continue
            else:
                seen.add(cs['prediction'])
                cset_filt.append(cs)

        for strategy in strategies:
            collator.contrast_sample_strategy = strategy
            if args.metric == 'relevance':
                try:
                    subset = collator.select_mixed_methods(cset_filt, args.max_num_rank)
                except Exception as e:
                    print(e)
                    continue
                subset_obj = [x for x in cset_filt if x['prediction'] in subset]

                likelihoods = []
                for x in subset_obj:
                    if 'primera_bertscore' in x:
                        likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        likelihoods.append(x['primera_bartscore'])
                if len(likelihoods) > 0:
                    stats_by_method[strategy]['likelihood'].append(float(np.mean(likelihoods)))

                avg_beam = np.mean([x['sample_idx'] + 1 for x in subset_obj])
                length = np.mean([len(s.split(' ')) for s in subset])
                stats_by_method[strategy]['beam'].append(avg_beam)
                stats_by_method[strategy]['length'].append(length)
                methods = Counter([x['method'] for x in subset_obj])
                for k, v in methods.items():
                    key = 'primera' if 'primera' in k else 'long_t5'
                    stats_by_method[strategy][key].append(v / len(subset_obj))
                avg_relevance = float(np.mean([
                    score_candidate_fn(x, relevance_metrics) for x in subset_obj
                ]))
                rels = [score_candidate_fn(x, relevance_metrics) for x in subset_obj]
                rels = np.sort(rels)
                gaps = []
                for i in range(1, len(rels)):
                    gaps.append(abs(rels[i] - rels[i - 1]))
                avg_gap = np.mean(gaps)
                avg_faithful = float(np.mean([
                    score_candidate_fn(x, faith_metrics) for x in subset_obj
                ]))
                avg_density = float(np.mean([x['density'] for x in subset_obj]))
                avg_coverage = float(np.mean([x['coverage'] for x in subset_obj]))
                stats_by_method[strategy]['diversity'].append(diversity_score(subset))
                stats_by_method[strategy]['density'].append(avg_density)
                stats_by_method[strategy]['coverage'].append(avg_coverage)
                stats_by_method[strategy]['relevance'].append(avg_relevance)
                stats_by_method[strategy]['faithful'].append(avg_faithful)
                stats_by_method[strategy]['metric_gap'].append(avg_gap)
            else:
                subset = collator.select_hard_set(cset_filt)

                pos_obj = [x for x in cset_filt if x['prediction'] in subset[:2]]
                neg_obj = [x for x in cset_filt if x['prediction'] in subset[2:]]

                neg_abs = [x['prediction'] for x in neg_obj]
                pos_abs = [x['prediction'] for x in pos_obj]
                neg_div = diversity_score(neg_abs)
                pos_div = diversity_score(pos_abs)

                stats_by_method[strategy]['diversity_positive'].append(neg_div)
                stats_by_method[strategy]['diversity_negative'].append(pos_div)
                stats_by_method[strategy]['diversity_cross'].append(cross_diversity(pos_abs, neg_abs))
                neg_methods = Counter([x['method'] for x in neg_obj])
                uses_reference = 0
                for x in pos_obj:
                    if x['method'] == 'reference':
                        uses_reference = 1
                        break
                stats_by_method[strategy]['includes_reference'].append(uses_reference)

                for k, v in neg_methods.items():
                    stats_by_method[strategy][k].append(v / len(neg_obj))

                pos_likelihoods = []
                for x in pos_obj:
                    if 'primera_bertscore' in x:
                        pos_likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        pos_likelihoods.append(x['primera_bartscore'])

                neg_likelihoods = []
                for x in neg_obj:
                    if 'primera_bertscore' in x:
                        neg_likelihoods.append(x['primera_bertscore'])
                    elif 'primera_bartscore' in x:
                        neg_likelihoods.append(x['primera_bartscore'])

                if len(neg_likelihoods) > 0:
                    ap = float(np.mean(pos_likelihoods))
                    an = float(np.mean(neg_likelihoods))
                    stats_by_method[strategy]['likelihood_positive'].append(ap)
                    stats_by_method[strategy]['likelihood_positive'].append(an)
                    stats_by_method[strategy]['likelihood_gap'].append(ap - an)

                neg_density = float(np.mean([x['density'] for x in neg_obj]))
                neg_coverage = float(np.mean([x['coverage'] for x in neg_obj]))
                stats_by_method[strategy]['density_negative'].append(neg_density)
                stats_by_method[strategy]['coverage_negative'].append(neg_coverage)

                pos_density = float(np.mean([x['density'] for x in pos_obj]))
                pos_coverage = float(np.mean([x['coverage'] for x in pos_obj]))

                stats_by_method[strategy]['density_positive'].append(pos_density)
                stats_by_method[strategy]['coverage_positive'].append(pos_coverage)

                stats_by_method[strategy]['density_gap'].append(pos_density - neg_density)
                stats_by_method[strategy]['coverage_gap'].append(pos_coverage - neg_coverage)
    return stats_by_method


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to analyze different sampling strategies for calibration')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='chemistry')
    parser.add_argument('--metric', default='faithful')
    parser.add_argument('--max_num_rank', default=4, type=int)
    parser.add_argument('--max_examples', default=100000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    from transformers import AutoTokenizer
    dummy = AutoTokenizer.from_pretrained('sshleifer/bart-tiny-random')

    metric_norm_fn = os.path.join(args.data_dir, f'{args.dataset}_metric_bounds.json')
    with open(metric_norm_fn, 'r') as fd:
        stats = ujson.load(fd)

    faith_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
    relevance_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']
    if args.metric == 'faithful':
        strategies = [
            'random', 'max_margin', 'min_margin', 'avg_margin', 'max_diversity', 'min_diversity',
            'easy', 'hard',
        ]
        default_metrics = faith_metrics.copy()
    elif args.metric == 'relevance':
        strategies = [
            'random', 'max_margin', 'min_margin', 'max_diversity', 'min_diversity', 'top_beam', 'bottom_beam',
            'wide_beam', 'min_metric', 'max_metric', 'max_gap', 'min_gap',
            'max_surprise', 'min_surprise',
        ]
        default_metrics = relevance_metrics.copy()
    else:
        raise Exception('Unrecognized metric')

    def score_candidate_fn(row, contrast_metrics=default_metrics):
        norm_vals = []
        for metric in contrast_metrics:
            stat = stats[metric]
            norm_vals.append((row[metric] - stat['mean']) / stat['std'])
        return sum(norm_vals) / len(norm_vals)

    collator = DataCollatorForContrastSeq2Seq(
        tokenizer=dummy,
        max_num_rank=args.max_num_rank,
        max_num_positive=2,
        max_num_negative=2,
        score_candidate_fn=score_candidate_fn,
        metric_mode='max',
        positive_methods='all',
        mixed_methods='all',
        negative_methods='none' if args.metric == 'relevance' else 'all',
        reference_status='remove' if args.metric == 'relevance' else 'positive',
        use_mixed_methods=args.metric == 'relevance'
    )

    pattern = os.path.join(args.data_dir, args.dataset, 'corruptions', 'train', '*.json')
    print(f'Looking for files matching {pattern}')
    fns = list(glob(pattern))

    # poss = []
    # negs = []
    # refs = []
    #
    # poo = []
    # pee = []
    # pong = []
    #
    # blah = []
    # bloo = []
    # blee = []
    # for fn in fns[:100]:
    #     with open(fn, 'r') as fd:
    #         obj = ujson.load(fd)
    #         para = [x for x in obj if x['method'] == 'paraphrase']
    #         neg = [x for x in obj if x['sign'] == 'negative']
    #         ref = [x for x in obj if x['method'] == 'reference']
    #
    #         for x in para:
    #             poss.append(x['fact_score'])
    #             poo.append(x['bart_score'])
    #             blah.append(x['bs_src_precision'])
    #
    #         for x in ref:
    #             refs.append(x['fact_score'])
    #             pong.append(x['bart_score'])
    #             blee.append(x['bs_src_precision'])
    #
    #         for x in neg:
    #             negs.append(x['fact_score'])
    #             pee.append(x['bart_score'])
    #             bloo.append(x['bs_src_precision'])
    #
    # print(np.mean(negs), np.mean(pee), np.mean(bloo))
    # print(np.mean(poss), np.mean(poo), np.mean(blah))
    # print(np.mean(refs), np.mean(pong), np.mean(blee))
    # raise

    n = len(fns)
    if n > args.max_examples:
        fns = list(np.random.choice(fns, size=(args.max_examples, ), replace=False))
    all_stats_by_method = defaultdict(lambda: defaultdict(list))
    if args.debug:
        single_stats_by_method = list(tqdm(map(lambda fn: record(args, fn), fns), total=len(fns)))
    else:
        single_stats_by_method = list(p_uimap(lambda fn: record(args, fn), fns))


    for stats_by_method in single_stats_by_method:
        for strategy, obj in stats_by_method.items():
            for k, v in obj.items():
                all_stats_by_method[strategy][k] += v

    out_df = []
    for strategy, obj in all_stats_by_method.items():
        strat_row = {'strategy': strategy}
        for k, v in obj.items():
            v_valid = [z for z in v if not np.isnan(z)]
            strat_row[k] = float(np.mean(v_valid))
        out_df.append(strat_row)
    out_df = pd.DataFrame(out_df)
    out_fn = os.path.join(args.data_dir, f'{args.dataset}_{args.metric}_strategy_covariates.csv')
    print(f'Saving to {out_fn}...')
    out_df.to_csv(out_fn, index=False)
