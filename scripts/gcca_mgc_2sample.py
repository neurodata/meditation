import numpy as np
from pathlib import Path
import os
import re
import h5py
import pandas as pd
import logging

from tqdm import tqdm
import time
#from mgcpy.independence_tests.mgc import MGC
from hyppo.independence import Dcorr
from scipy.stats import multiscale_graphcorr
from itertools import combinations

import sys
sys.path.append("../")
from src.tools.utils import get_files, get_latents

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
tag = '_min_rank-ZG3'#'_max_rank-ZG2' 
gccadir = datadir / f'gcca_05-26-10:39{tag}'#f'gcca_05-17-18:27{tag}' # 
decimate_dir = datadir / 'decimate'
logpath = Path('../logs')

## Stuff
h5_key = 'latent'
tasks = ['restingstate', 'openmonitoring', 'compassion']
levels = ['e', 'n']

## Params
n_permutations = 10000
fast = False

## Test
TEST = 'DCORR'


## Get files
def get_files(path,
              level='(e|n)',
              subject='([0-9]{3})',
              task='(.+?)',
              filetype='h5',
              flag=''):
    files = []
    query = f'^{level}_sub-'
    query += f'{subject}_ses-1_'
    query += f'task-{task}{flag}.{filetype}'
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))
    
    return(files)

lookup = {'(e|n)':'All',
          '(.+?)':'All',
          'e':'Experts',
          'n':'Novices',
          'openmonitoring':'Open Monitoring',
          'compassion':'Compassion',
          'restingstate':'Resting',
          '(openmonitoring|compassion)': 'Meditating'}

## Get class files
def get_class(components,level,task,cls_num=1):
    paths = get_files(path=gccadir, level=level, task=task, flag='_gcca')

    latents = []
    labels = []
    for path,_ in tqdm(paths):
        h5f = h5py.File(gccadir / path,'r')
        if components is not None:
            latent = h5f[h5_key][:][:,components]
        else:
            latent = h5f[h5_key][:]
        h5f.close()
        latents.append(latent.reshape(1, -1))
        labels.append(f'{cls_num}')

    return(np.vstack(latents), labels)

def transform_matrices(x, y, is_y_categorical=False):
    if not is_y_categorical:
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    else:
        u = x
        v = preprocessing.LabelEncoder().fit_transform(y) + 1
    
    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]
    
    return(u, v)

def discrim_test(trait1,state1,trait2,state2,components):
    latents1, _ = get_class(components=components,cls_num=1,level=trait1,task=state1)
    latents2, _ = get_class(components=components,cls_num=2,level=trait2,task=state2)
    
    u,v = transform_matrices(latents1,latents2)
    
    if TEST == 'MGC':
        # mgc = MGC()
        # pval, _ = mgc.p_value(u,v, is_fast=fast, replication_factor=n_permutations)
        _, pval, _ = multiscale_graphcorr(
             u,v,
             workers=-1,
             reps=n_permutations,
             random_state=0
        )
    elif TEST == 'DCORR':
        _, pval = Dcorr().test(
            u, v,
            reps=n_permutations,
            workers=-1,
            auto=fast)
    
    name = f'{lookup[trait1]} {lookup[state1]} vs. {lookup[trait2]} {lookup[state2]}'
    
    return(pval, name)

def gcca_pvals(n_components):
    pvals = []
    names = []

    ## x vs. x
    states = ['restingstate', 'openmonitoring', 'compassion']
    traits = ['e', 'n', '(e|n)']

    for trait in traits:
        for state1,state2 in combinations(states,2):
            pval,name = discrim_test(trait,state1,trait,state2,n_components)
            pvals.append(pval)
            names.append(name)
            
        state1, state2 = '(openmonitoring|compassion)', 'restingstate'
        pval,name = discrim_test(trait,state1,trait,state2,n_components)
        pvals.append(pval)
        names.append(name)

    ## x vs. y
    trait1 = 'e'
    trait2 = 'n'

    ## Inter (between) states
    for state1,state2 in combinations(states,2):
        pval,name = discrim_test(trait1,state1,trait2,state2,n_components)
        pvals.append(pval)
        names.append(name)

        pval,name = discrim_test(trait1,state2,trait2,state1,n_components)
        pvals.append(pval)
        names.append(name)
        
    ## Intra (within) state
    for state in states + ['(.+?)']:
        pval,name = discrim_test(trait1,state,trait2,state,n_components)
        pvals.append(pval)
        names.append(name)
        
    #state1, state2 = '(openmonitoring|compassion)', 'restingstate'
    #pval,name = discrim_test(trait1,state1,trait2,state2,n_components)

    state1, state2 = 'restingstate', '(openmonitoring|compassion)'
    pval,name = discrim_test(trait1,state1,trait2,state2,n_components)
    pvals.append(pval)
    names.append(name)
    pval,name = discrim_test(trait1,state2,trait2,state1,n_components)
    pvals.append(pval)
    names.append(name)

    return(names,np.array(pvals))

def main():
    ## Create Log File
    logging.basicConfig(filename=logpath / 'mgc_logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG
                        )
    logging.info(f'NEW RUN: {TEST} 2sample, {n_permutations} permutations, fast={fast}')

    data_dict = {}
    components = [[0], [1], [2],
                [0,1], [1,2], [2,0],
                [0,1,2]]
    for cs in components:
        t0 = time.time()
        names,pvals = gcca_pvals(cs)
        data_dict[str(cs)] = (names,pvals)
        logging.info(f'Component(s) {cs} done in {time.time()-t0}')

    df=pd.DataFrame(columns=[])
    for key in sorted(data_dict.keys(), reverse=True):
        names,pvals = data_dict[key]
        if not fast:
            pvals[pvals < 1/n_permutations] = 1/n_permutations 
        logging.info(f'Names: {names}')
        logging.info(f'pvals: {pvals}')
        df['Samples Compared'] = names
        df[f'components={key}'] = [f'{x:.2g}' for x in pvals]

    if fast:
        save_path =  f'../data/{TEST}_gcca_pvals_{n_permutations}_FAST{tag}.csv'
    else:
        save_path =  f'../data/{TEST}_gcca_pvals_{n_permutations}{tag}.csv'
    logging.info(f'Saving to {save_path}')
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()