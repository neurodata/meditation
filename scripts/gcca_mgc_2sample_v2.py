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


# if ONLY_FULL_SUBJECTS:
#     subj_indices = defaultdict(list)
#     for i,subj in enumerate(np.hstack(subjs)):
#         subj_indices[subj].append(i)
#     max_len = len(max(subj_indices.values(), key=lambda x: len(x)))
#     valid_subjs = np.unique([
#         subj for subj,group in subj_indices.items() if len(group) == max_len
#     ])

# else:
#     valid_subjs = np.unique(np.hstack(subjs))

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
test_list = []
# ## Intra (within) Trait, Inter (between) State
test_list += [
    # Permutation: restricted, within subject
    # ('Experts Resting', 'Experts Compassion', 'within'),
    # ('Experts Resting', 'Experts Open Monitoring', 'within'),
    # ('Experts Open Monitoring', 'Experts Compassion', 'within'),
    # ('Experts Resting', 'Experts Meditating', 'within'),
    # ('Novices Resting', 'Novices Compassion', 'within'),
    # ('Novices Resting', 'Novices Open Monitoring', 'within'),
    ('Novices Open Monitoring', 'Novices Compassion', 'within'),
    ('Novices Resting', 'Novices Meditating', 'within')
]
# ## Inter (between) Trait, Intra (within) State
test_list += [
    # Permutation: full
    # ('Experts Resting', 'Novices Resting', 'full'),
    # ('Experts Compassion', 'Novices Compassion', 'full'),
    ('Experts Open Monitoring', 'Novices Open Monitoring', 'full'),
]
# Permutation: restricted, across subject
test_list += [
    # ('Experts Meditating', 'Novices Meditating', 'across'),
    ('Experts All', 'Novices All', 'across'),
]
## Inter (between) Trait, Inter (between) State
test_list += [
    # Permutation: free
    # ('Experts Resting', 'Novices Compassion', 'full'),
    # ('Experts Resting', 'Novices Open Monitoring', 'full'),
    # ('Experts Compassion', 'Novices Resting', 'full'),
    # ('Experts Compassion', 'Novices Open Monitoring', 'full'),
    # ('Experts Open Monitoring', 'Novices Resting', 'full'),
    ('Experts Open Monitoring', 'Novices Compassion', 'full'),
    # Permutation: restricted, permute state (preserve # labels)
    # # ('Experts Resting', 'Novices Meditating', 'across'),
    # # ('Experts Meditating', 'Novices Resting', 'across'),
]
# # Intra State (need to figure out these permutations)
test_list += [
    # Permutation: restricted, permute state
    # ('Resting', 'Compassion', 'within'),
    # ('Resting', 'Open Monitoring', 'within'),
    ('Compassion', 'Open Monitoring', 'within'),
    # Permutation: restricted, permute state (preserve # labels)
    ('Resting', 'Meditating', 'within')
]


################ FUNCTIONS ###################

def discrim_test(
    TEST,
    X, Y,
    fast,
    compute_distance=None,
    n_permutations=10000,
    permute_groups=None, 
    permute_structure=None
):
    if TEST == 'MGC':
        if compute_distance:
            stat, pvalue, mgc_dict = multiscale_graphcorr(
                X,
                Y,
                workers=-1,
                reps=n_permutations,
                random_state=0,
                permute_groups=permute_groups,
                permute_structure=permute_structure,
                #global_corr='mgc_restricted',
            )
        else:
            stat, pvalue, mgc_dict = multiscale_graphcorr(
                X,
                Y,
                workers=-1,
                reps=n_permutations,
                random_state=0,
                compute_distance=None,
                permute_groups=permute_groups,
                permute_structure=permute_structure,
                #global_corr='mgc_restricted',
            )
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
            "null_dist": mgc_dict["null_dist"],
            "opt_scale": mgc_dict["opt_scale"],
        }
    elif TEST == 'DCORR':
        stat, pvalue = Dcorr().test(
            X, Y,
            reps=n_permutations,
            workers=-1,
            auto=fast,
            permute_groups=permute_groups,
            permute_structure=permute_structure,
        )
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
        }

    return stat_dict

def gcca_pvals(
    test,
    g1, g2,
    groups,
    labels,
    subjs,
    n_permutations,
    gradients,
    fast,
    permute_structure=None
):
    name = f'{g1} vs. {g2}'
    results_dict = {}
    
    g1_labels = lookup[g1]
    g2_labels = lookup[g2]

    subj_list = np.concatenate(
        [np.asarray(subjs[i]) for i in g1_labels] +
        [np.asarray(subjs[i]) for i in g2_labels]
    )

    print(name)
    for grads in gradients:
        X, Y = k_sample_transform(
            [np.vstack([np.asarray(groups[i]) for i in g1_labels])]
            + [np.vstack([np.asarray(groups[i]) for i in g2_labels])]
        )
        X = X[:, :, grads].reshape(X.shape[0], -1)
        permute_groups = subj_list
  
        #X_dists = pairwise_distances(X, metric="euclidean")
        #Y_dists = pairwise_distances(Y, metric="sqeuclidean")

        stat_dict = discrim_test(
            test,
            X, Y,
            fast=fast,
            compute_distance=True,
            n_permutations=n_permutations,
            permute_groups=permute_groups,
            permute_structure=permute_structure,

        )
        results_dict[grads] = stat_dict

    return(name, results_dict)

def simulate_data(subjs):
    subj2vec = dict()
    for subj in np.unique(np.concatenate(subjs)):
        subj2vec[subj] = np.random.normal(0,1,(18715,1))
    groups = []
    for subj_list in subjs:
        groups.append([np.random.normal(subj2vec[subj],0.1) for subj in subj_list])
    return groups

def main():
    # Params
    n_permutations = 10000
    fast = False

    ## Test
    TEST = 'DCORR'#'MGC'#
    LABEL = 'restricted_perm'

    ## Test data
    SIMULATED_TEST = True

    ## Create Log File
    logging.basicConfig(filename=logpath / 'mgc_logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG
                        )
    logging.info(f'NEW RUN: {TEST} 2sample, {n_permutations} permutations, fast={fast}, simulated={SIMULATED_TEST}')

    if SIMULATED_TEST:
        _, labels, subjs = get_latents(gccadir, flag="_gcca", ids=True)
        n_datasets = 500
        n_permutations = 100
        data_dict = defaultdict(list)
        for _ in range(n_datasets):
            groups = simulate_data(subjs)
            for (g1,g2,permute_structure) in test_list:
                t0 = time.time()
                name, stat_dict = gcca_pvals(
                    TEST,
                    g1,g2,
                    groups=groups,
                    labels=labels,
                    subjs=subjs,
                    fast=fast,
                    n_permutations=n_permutations,
                    gradients=[(0)],
                    permute_structure=permute_structure
                )
                data_dict[name].append(stat_dict[(0)]['pvalue'])

        save_dir = Path('../data/2sample_tests/simulations/')
        with open(save_dir / f"{TEST}_{LABEL}_SIMULATED_datasets={n_datasets}_dict_perm={n_permutations}{tag}.pkl", "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        groups, labels, subjs = get_latents(gccadir, flag="_gcca", ids=True)
        ## Gradients
        gradients = [
            (0), (1), (2),
            (0,1), (1,2), (2,0),
            (0,1,2)
        ]
        data_dict = {}
        for (g1,g2,permute_structure) in test_list:
            t0 = time.time()
            name, stat_dict = gcca_pvals(
                TEST,
                g1,g2,
                groups=groups,
                labels=labels,
                subjs=subjs,
                fast=fast,
                n_permutations=n_permutations,
                gradients=gradients,
                permute_structure=permute_structure
            )
            data_dict[name] = stat_dict
            logging.info(f'Test {g1} vs. {g2} done in {time.time()-t0}')

        df = pd.DataFrame(columns=['Comparison'] + [f'Gradients {g}' for g in gradients])
        df['Comparison'] = data_dict.keys()
        for grads in gradients:
            df[f'Gradients {grads}'] = [val_dict[grads]['pvalue'] for val_dict in data_dict.values()]

        save_dir = Path('../data/2sample_tests/')
        logging.info(f'Saving to {save_dir}')

        df.to_csv(save_dir / f'{TEST}_{LABEL}_pvalues_{n_permutations}{tag}.csv', index=False)
        with open(save_dir / f"{TEST}_{LABEL}_results_dict_{n_permutations}{tag}.pkl", "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()