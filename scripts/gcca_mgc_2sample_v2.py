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
from hyppo.independence import Dcorr
from hyppo.ksample._utils import k_sample_transform
from scipy.stats import multiscale_graphcorr
from sklearn.metrics import pairwise_distances
from itertools import combinations
from collections import defaultdict

import sys
sys.path.append("../")
from src.tools.utils import get_files, get_latents

################ DEFINITIONS #########################

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
tag = '_min_rank-ZG3'#'_max_rank-ZG2' 
gccadir = datadir / f'gcca_05-26-10:39{tag}'#f'gcca_05-17-18:27{tag}' # 
decimate_dir = datadir / 'decimate'
logpath = Path('../logs')

groups, labels, subjs = get_latents(gccadir, flag="_gcca", ids=True)

## Params
n_permutations = 10000
fast = False

## Test
TEST = 'MGC'

# [['e', 'restingstate'],
#  ['e', 'openmonitoring'],
#  ['e', 'compassion'],
#  ['n', 'restingstate'],
#  ['n', 'openmonitoring'],
#  ['n', 'compassion']]
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

## Intra (within) Trait, Inter (between) State
test_list = [
    ('Experts Resting', 'Experts Compassion'),
    ('Experts Resting', 'Experts Open Monitoring'),
    ('Experts Open Monitoring', 'Experts Compassion'),
    ('Experts Resting', 'Experts Meditating'),
    ('Novices Resting', 'Novices Compassion'),
    ('Novices Resting', 'Novices Open Monitoring'),
    ('Novices Open Monitoring', 'Novices Compassion'),
    ('Novices Resting', 'Novices Meditating')
]
## Inter (between) Trait, Intra (within) State
test_list += [
    ('Experts Resting', 'Novices Resting'),
    ('Experts Compassion', 'Novices Compassion'),
    ('Experts Open Monitoring', 'Novices Open Monitoring'),
    ('Experts Meditating', 'Novices Meditating'),
    ('Experts All', 'Novices All'),
]
## Inter (between) Trait, Inter (between) State
test_list += [
    ('Experts Resting', 'Novices Compassion'),
    ('Experts Resting', 'Novices Open Monitoring'),
    ('Experts Compassion', 'Novices Resting'),
    ('Experts Compassion', 'Novices Open Monitoring'),
    ('Experts Open Monitoring', 'Novices Resting'),
    ('Experts Open Monitoring', 'Novices Compassion'),
    ('Experts Resting', 'Novices Meditating'),
    ('Experts Meditating', 'Novices Resting'),
]
## Intra State
test_list += [
    ('Resting', 'Compassion'),
    ('Resting', 'Open Monitoring'),
    ('Compassion', 'Open Monitoring'),
    ('Resting', 'Meditating')
]
## Gradients
gradients = [
    (0), (1), (2),
    (0,1), (1,2), (2,0),
    (0,1,2)
]

################ FUNCTIONS ###################

def discrim_test(X, Y, compute_distance=True, y_groups=None):
    if TEST == 'MGC':
        if compute_distance:
            stat, pvalue, mgc_dict = multiscale_graphcorr(
                X,
                Y,
                workers=-1,
                reps=n_permutations,
                random_state=0,
                y_groups=y_groups,
            )
        else:
            stat, pvalue, mgc_dict = multiscale_graphcorr(
                X,
                Y,
                workers=-1,
                reps=n_permutations,
                random_state=0,
                compute_distance=None,
                y_groups=y_groups,
            )
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
            "null_dist": mgc_dict["null_dist"],
            "opt_scale": mgc_dict["opt_scale"],
        }
    elif TEST == 'DCORR':
        stat, pval = Dcorr().test(
            X, Y,
            reps=n_permutations,
            workers=-1,
            auto=fast)
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
        }

    return stat_dict

def gcca_pvals(g1, g2):
    name = f'{g1} vs. {g2}'
    results_dict = {}
    
    g1_labels = lookup[g1]
    g2_labels = lookup[g2]

    subj_list = np.concatenate(
        [np.asarray(subjs[i]) for i in g1_labels]
        + [np.asarray(subjs[i]) for i in g2_labels],
    )

    for grads in gradients:

        X, Y = k_sample_transform(
            [np.vstack([np.asarray(groups[i]) for i in g1_labels])]
            + [np.vstack([np.asarray(groups[i]) for i in g2_labels])]
        )
        X = X[:, :, grads].reshape(X.shape[0], -1)

        X_dists = pairwise_distances(X, metric="euclidean")
        Y_dists = pairwise_distances(Y, metric="sqeuclidean")

        Y_group_adj = pairwise_distances(subj_list[:, None], metric=lambda x, y: x != y)

        stat_dict = discrim_test(
            X_dists, Y_dists, compute_distance=False, y_groups=Y_group_adj
        )
        results_dict[grads] = stat_dict

    return(name, results_dict)

def main():
    ## Create Log File
    logging.basicConfig(filename=logpath / 'mgc_logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG
                        )
    logging.info(f'NEW RUN: {TEST} 2sample, {n_permutations} permutations, fast={fast}')

    data_dict = {}
    for (g1,g2) in test_list:
        t0 = time.time()
        name, stat_dict = gcca_pvals(g1,g2)
        data_dict[name] = stat_dict
        logging.info(f'Test {g1} vs. {g2} done in {time.time()-t0}')

    df = pd.DataFrame(columns=['Comparison'] + [f'Gradients {g}' for g in gradients])
    df['Comparison'] = data_dict.keys()
    for grads in gradients:
        df[f'Gradients {grads}'] = [val_dict[grads]['pvalue'] for val_dict in data_dict.values()]

    save_dir = Path('../data/2sample_tests/')
    logging.info(f'Saving to {save_dir}')

    df.to_csv(save_dir / f'{TEST}_pvalues_{n_permutations}{tag}.csv', index=False)
    with open(save_dir / f"{TEST}_results_dict_{n_permutations}{tag}.pkl", "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()