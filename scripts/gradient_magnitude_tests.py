import numpy as np
from pathlib import Path
import os
import re
import h5py
import pandas as pd
import logging
import pickle

from tqdm import tqdm
import time
from hyppo.ksample._utils import k_sample_transform
from sklearn.metrics import pairwise_distances
from itertools import combinations
from collections import defaultdict
import argparse

import sys
sys.path.append("../")
from src.tools import get_files, get_latents

from scipy.stats import wilcoxon, mannwhitneyu

################ DEFINITIONS #########################

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
logpath = Path.home() / 'meditation' / 'logs'

lookup = {'Experts All':[0,1,2],
          'Novices All':[3,4,5],
          'Experts Resting':[0],
          'Experts Open Monitoring':[1],
          'Experts Compassion':[2],
          'Novices Resting':[3],
          'Novices Open Monitoring':[4],
          'Novices Compassion':[5],
          'Experts Meditating':[1,2],
          'Novices Meditating':[4,5],
          'Resting':[0,3],
          'Compassion':[2,5],
          'Open Monitoring':[1,4],
          'Meditating':[1,2,4,5]
}
TEST_LIST = []
# ## Intra (within) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: restricted, within subject
    ('Experts Resting', 'Experts Compassion', 'within'),
    ('Experts Resting', 'Experts Open Monitoring', 'within'),
    ('Experts Open Monitoring', 'Experts Compassion', 'within'),
    # ('Experts Resting', 'Experts Meditating', 'within'),
    ('Novices Resting', 'Novices Compassion', 'within'),
    ('Novices Resting', 'Novices Open Monitoring', 'within'),
    ('Novices Open Monitoring', 'Novices Compassion', 'within'),
    # ('Novices Resting', 'Novices Meditating', 'within')
]
# ## Inter (between) Trait, Intra (within) State
TEST_LIST += [
    # Permutation: full
    ('Experts Resting', 'Novices Resting', 'full'),
    ('Experts Compassion', 'Novices Compassion', 'full'),
    ('Experts Open Monitoring', 'Novices Open Monitoring', 'full'),
]
# Permutation: restricted, across subject
# TEST_LIST += [
#     ('Experts Meditating', 'Novices Meditating', 'across'),
#     ('Experts All', 'Novices All', 'across'),
# ]
## Inter (between) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: free
    ('Experts Resting', 'Novices Compassion', 'full'),
    ('Experts Resting', 'Novices Open Monitoring', 'full'),
    ('Experts Compassion', 'Novices Resting', 'full'),
    ('Experts Compassion', 'Novices Open Monitoring', 'full'),
    ('Experts Open Monitoring', 'Novices Resting', 'full'),
    ('Experts Open Monitoring', 'Novices Compassion', 'full'),
]
# # Intra State (need to figure out these permutations)
# TEST_LIST += [
#     # Permutation: restricted, permute state
#     ('Resting', 'Compassion', 'within'),
#     ('Resting', 'Open Monitoring', 'within'),
#     ('Compassion', 'Open Monitoring', 'within'),
#     # Permutation: restricted, permute state (preserve # labels)
#     ('Resting', 'Meditating', 'within')
# ]
TEST_LIST = [((a,b),c) for a,b,c in TEST_LIST]

################ FUNCTIONS ###################

def test_groups(
    test,
    group_names,#g1, g2,
    groups,
    labels,
    subjs,
    gradients,
):
    if len(group_names) == 2:
        name = f'{group_names[0]} vs. {group_names[1]}'
    else:
        name = f"{len(group_names)}-sample ({[lookup[g] for g in group_names]})"

    results_dict = {}

    subjs1, subjs2 = [
        np.concatenate([np.asarray(subjs[i]) for i in lookup[g]])
        for g in group_names]

    sort1_idx = np.argsort(subjs1)
    sort2_idx = np.argsort(subjs2)
    if len(subjs1) == len(subjs2) and \
            np.all(subjs1[sort1_idx] == subjs2[sort2_idx]):
        print(f'{name} wilcoxon (paired)')
    else:
        print(f'{name} mann-whitney U (unpaired)')

    sample1, sample2 = [np.vstack([np.linalg.norm(groups[i], axis=1) for i in lookup[g]]) for g in group_names]

    for grads in gradients:
        s1g = sample1[:, grads]
        s2g = sample2[:, grads]
        if len(subjs1) == len(subjs2) and \
            np.all(subjs1[sort1_idx] == subjs2[sort2_idx]):
            stat, pval = wilcoxon(s1g[sort1_idx], s2g[sort2_idx])
        else:
            stat, pval = mannwhitneyu(s1g, s2g)

    return name, pval


def get_k_sample_group(k_sample):
    if k_sample == '6':
        return [(
            (
                'Experts Resting',
                'Experts Open Monitoring',
                'Experts Compassion',
                'Novices Resting',
                'Novices Open Monitoring',
                'Novices Compassion',
            ),
            'NA'
        )]
    elif k_sample == '3N':
        return [(
            (
                'Novices Resting',
                'Novices Open Monitoring',
                'Novices Compassion',
            ),
            'NA'
        )]
    if k_sample == '3E':
        return [(
            (
                'Experts Resting',
                'Experts Open Monitoring',
                'Experts Compassion',
            ),
            'NA'
        )]
    else:
        raise ValueError(f'Undefined k_sample group label {k_sample}')

def main(
    SOURCE,
    TEST,
    LABEL,
    k_sample,
    exclude_ids,
    data,
    start_grad,
):
    ## Create Log File
    logging.basicConfig(filename=logpath / 'logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.INFO
                        )
    logging.info(f'NEW RUN: test magnitudes, k_sample={k_sample}, start_grad {start_grad}')

    source_dir = Path(SOURCE)
    flag = '_' + data if data != 'dmap2' else '_emb'
    ftype = 'h5' if data != 'dmap2' else 'npy'
    print(f'Loading data from directory: {source_dir}')
    logging.info(f'Loading data from directory: {source_dir}')
    groups, labels, subjs = get_latents(
        source_dir, n_components=3, flag=flag, ids=True, ftype=ftype,
        subjects_exclude=exclude_ids, start_grad=start_grad, source=data)

    # check proper exclusion
    if exclude_ids is not None and len(set(exclude_ids).intersection(np.concatenate(subjs))) > 0:
        raise RuntimeError(f'Subject ID not excluded: {set(exclude_ids).intersection(np.concatenate(subjs))}')

    ## Gradients
    gradients = [
        (0), (1), (2),
    ]
    data_dict = {}
    if k_sample is None:
        save_dir = Path('../data/')
        test_list = np.asarray(TEST_LIST)
        save_name = "magnitudes_2-sample"
    # else:
    #     save_dir = Path('../data/')
    #     test_list = get_k_sample_group(k_sample)
    #     save_name = f"{k_sample}-kw_magnitudes"
    save_dir = save_dir / f'{TEST}_{data}_{LABEL}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir / f'{save_name}.csv', "w") as f:
        f.write(",".join(['Comparison'] + [f'\"Gradients {grads}\"' for grads in gradients]) + '\n')

    for (group_names, permute_structure) in test_list:
        t0 = time.time()
        name, pval = test_groups(
            TEST,
            group_names,
            groups=groups,
            labels=labels,
            subjs=subjs,
            gradients=gradients,
        )
        logging.info(f'Test {name} done in {time.time()-t0}')
        with open(save_dir / f'{save_name}.csv', "a") as f:
            f.write(",".join([f'\"{name}\"'] + [str(pval) for grads in gradients]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="", type=str, required=True)
    parser.add_argument("--test", help="", type=str, required=True)
    parser.add_argument("--label", help="", type=str, default=None)
    parser.add_argument("--k-sample", help="Options {6: 6-sample}", type=str, default=None)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str)
    parser.add_argument("-d", "--data", help="list servers, storage, or both (default: %(default)s)", choices=['gcca', 'dmap', 'mase', 'svd', 'mase_dmap', 'joint', 'grouppca', 'dmap2'], default="gcca")
    parser.add_argument("--start-grad", help="Starting index of the first 3 gradients", type=int, default=0)
    args = parser.parse_args()

    main(
        SOURCE = args.source,
        TEST = args.test,
        LABEL = args.label,
        k_sample = args.k_sample,
        exclude_ids = args.exclude_ids,
        data = args.data,
        start_grad = args.start_grad,
    )
