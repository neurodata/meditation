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
import argparse

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
logpath = Path('../logs/')


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
TEST_LIST = []
# ## Intra (within) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: restricted, within subject
    ('Experts Resting', 'Experts Compassion', 'within'),
    ('Experts Resting', 'Experts Open Monitoring', 'within'),
    ('Experts Open Monitoring', 'Experts Compassion', 'within'),
    ('Experts Resting', 'Experts Meditating', 'within'),
    ('Novices Resting', 'Novices Compassion', 'within'),
    ('Novices Resting', 'Novices Open Monitoring', 'within'),
    ('Novices Open Monitoring', 'Novices Compassion', 'within'),
    ('Novices Resting', 'Novices Meditating', 'within')
]
# ## Inter (between) Trait, Intra (within) State
TEST_LIST += [
    # Permutation: full
    ('Experts Resting', 'Novices Resting', 'full'),
    ('Experts Compassion', 'Novices Compassion', 'full'),
    ('Experts Open Monitoring', 'Novices Open Monitoring', 'full'),
]
# Permutation: restricted, across subject
TEST_LIST += [
    ('Experts Meditating', 'Novices Meditating', 'across'),
    ('Experts All', 'Novices All', 'across'),
]
## Inter (between) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: free
    ('Experts Resting', 'Novices Compassion', 'full'),
    ('Experts Resting', 'Novices Open Monitoring', 'full'),
    ('Experts Compassion', 'Novices Resting', 'full'),
    ('Experts Compassion', 'Novices Open Monitoring', 'full'),
    ('Experts Open Monitoring', 'Novices Resting', 'full'),
    ('Experts Open Monitoring', 'Novices Compassion', 'full'),
    # Permutation: restricted, permute state (preserve # labels)
    # # ('Experts Resting', 'Novices Meditating', 'across'),
    # # ('Experts Meditating', 'Novices Resting', 'across'),
]
# # Intra State (need to figure out these permutations)
TEST_LIST += [
    # Permutation: restricted, permute state
    ('Resting', 'Compassion', 'within'),
    ('Resting', 'Open Monitoring', 'within'),
    ('Compassion', 'Open Monitoring', 'within'),
    # Permutation: restricted, permute state (preserve # labels)
    ('Resting', 'Meditating', 'within')
]
TEST_LIST = [((a,b),c) for a,b,c in TEST_LIST]
SIMULATE_IDX = [0, 3, 8, 11, 12, 13, 19, 22]

################ FUNCTIONS ###################

def perm2blocks(perm_labels, perm_structure):
    # if perm_structure == 'full':
    #     return None
    # elif perm_structure == 'within':
    #     _, labels = np.unique(perm_labels, return_inverse=True)
    #     labels = labels*-1 - 1
    #     return labels
    # elif perm_structure == 'across':
    #     _, labels = np.unique(perm_labels, return_inverse=True)
    #     return np.hstack((labels[:, np.newaxis], -1*np.arange(labels.shape[0])[:, np.newaxis]))
    # else:
    #     print(f'Invalid structure: {perm_structure}')

    _, labels = np.unique(perm_labels, return_inverse=True)
    return labels

def discrim_test(
    TEST,
    X, Y,
    fast,
    compute_distance=None,
    n_permutations=1000,
    permute_groups=None, 
    permute_structure=None,
    global_corr='mgc',
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
                global_corr=global_corr,#'mgc_restricted'
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
                global_corr=global_corr,
            )
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
            "null_dist": mgc_dict["null_dist"],
            "opt_scale": mgc_dict["opt_scale"],
        }
    elif TEST == 'DCORR':
        perm_blocks = perm2blocks(permute_groups, permute_structure)
        stat, pvalue = Dcorr().test(
            X, Y,
            reps=n_permutations,
            workers=-1,
            auto=fast,
            perm_blocks=perm_blocks,
        )
        stat_dict = {
            "pvalue": pvalue,
            "test_stat": stat,
        }

    return stat_dict

def gcca_pvals(
    test,
    group_names,#g1, g2,
    groups,
    labels,
    subjs,
    n_permutations,
    gradients,
    fast,
    permute_structure=None,
    global_corr="mgc",
):
    if len(group_names) == 2:
        name = f'{group_names[0]} vs. {group_names[1]}'
    else:
        name = f"{len(group_names)}-sample ({[lookup[g] for g in group_names]})"

    results_dict = {}

    subj_list = np.concatenate(
        [np.concatenate([np.asarray(subjs[i]) for i in lookup[g]]) for g in group_names]
    )

    print(name)
    for grads in gradients:
        X, Y = k_sample_transform(
            [np.vstack([np.asarray(groups[i]) for i in lookup[g]]) for g in group_names]
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
            global_corr=global_corr,
        )
        results_dict[grads] = stat_dict

    return(name, results_dict)

def simulate_data(subjs, d=18715):
    subj2vec = dict()
    for subj in np.unique(np.concatenate(subjs)):
        subj2vec[subj] = np.random.normal(0,1,(d,1))
    groups = []
    for subj_list in subjs:
        groups.append([np.random.normal(subj2vec[subj],0.1) for subj in subj_list])
    return groups

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
    TEST,
    LABEL,
    n_permutations,
    n_datasets,
    fast,
    SIMULATED_TEST,
    global_corr,
    d,
    k_sample,
):
    ## Create Log File
    logging.basicConfig(filename=logpath / 'logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.INFO
                        )
    logging.info(f'NEW RUN: {TEST}, {n_permutations} permutations, fast={fast}, simulated={SIMULATED_TEST}, k_sample={k_sample}')

    if SIMULATED_TEST:
        _, labels, subjs = get_latents(gccadir, flag="_gcca", ids=True)
        data_dict = defaultdict(list)
        if k_sample is None:
            save_dir = Path('../data/2sample_tests/simulations/')
            test_list = np.asarray(TEST_LIST)[SIMULATE_IDX]
        else:
            save_dir = Path('../data/ksample_tests/simulations/')
            test_list = get_k_sample_group(k_sample)
        for _ in range(n_datasets):
            groups = simulate_data(subjs, d=d)
            for (group_names,permute_structure) in test_list:
                t0 = time.time()
                name, stat_dict = gcca_pvals(
                    TEST,
                    group_names,
                    groups=groups,
                    labels=labels,
                    subjs=subjs,
                    fast=fast,
                    n_permutations=n_permutations,
                    gradients=[(0)],
                    permute_structure=permute_structure,
                    global_corr=global_corr,
                )
                data_dict[name].append(stat_dict[(0)]['pvalue'])
            
        with open(save_dir / f"{TEST}_{LABEL}_SIMULATED_datasets={n_datasets}_dict_perm={n_permutations}_dim={d}{tag}.pkl", "wb") as f:
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
        if k_sample is None:
            save_dir = Path('../data/2sample_tests/')
            test_list = np.asarray(TEST_LIST)
        else:
            save_dir = Path('../data/ksample_tests/')
            test_list = get_k_sample_group(k_sample)
        with open(save_dir / f'{TEST}_{LABEL}_pvalues_{n_permutations}{tag}.csv', "w") as f:
            f.write(",".join(['Comparison'] + [f'Gradients {grads}' for grads in gradients]) + '\n')
        for (group_names,permute_structure) in test_list:
            t0 = time.time()
            name, stat_dict = gcca_pvals(
                TEST,
                group_names,
                groups=groups,
                labels=labels,
                subjs=subjs,
                fast=fast,
                n_permutations=n_permutations,
                gradients=gradients,
                permute_structure=permute_structure,
                global_corr=global_corr,
            )
            data_dict[name] = stat_dict
            logging.info(f'Test {name} done in {time.time()-t0}')
            with open(save_dir / f'{TEST}_{LABEL}_pvalues_{n_permutations}{tag}.csv', "a") as f:
                f.write(",".join([name] + [str(stat_dict[grads]['pvalue']) for grads in gradients]) + '\n')

        logging.info(f'Saving to {save_dir}')
        with open(save_dir / f"{TEST}_{LABEL}_results_dict_{n_permutations}{tag}.pkl", "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="", type=str, required=True)
    parser.add_argument("--label", help="", type=str, default=None)
    parser.add_argument("--n-perms", help="", type=int, default=1000)
    parser.add_argument("--n-datasets", help="", type=int, default=100)
    parser.add_argument("--simulate", help="", action="store_true")
    parser.add_argument("--gcorr", help="", type=str, default="mgc")
    parser.add_argument("--sim-dim", help="", type=int, default=18715)
    parser.add_argument("--k-sample", help="Options {6: 6-sample}", type=str, default=None)
    args = parser.parse_args()
    
    fast = False

    main(
        TEST = args.test,
        LABEL = args.label,
        n_permutations = args.n_perms,
        n_datasets = args.n_datasets,
        fast = fast,
        SIMULATED_TEST = args.simulate,
        global_corr = args.gcorr,
        d = args.sim_dim,
        k_sample = args.k_sample,
    )
