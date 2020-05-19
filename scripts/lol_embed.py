import numpy as np
from pathlib import Path
import os
import re
import h5py
import pandas as pd
import logging

from tqdm import tqdm
import time

from lol import QOQ
from lol import LOL

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
gccadir = datadir / 'gcca_raw-True_param-0.9'
logpath = Path('.')
savepath = logpath / 'lolp_state_gcca_raw-True_param-0.9_split-half_ranks-3:6.csv'
savepath_fit = logpath / 'lolp_state_gcca_raw-True_param-0.9-split-half_fit_ranks-3:6.csv'


## Stuff
h5_key = 'latent'

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

def get_latents(paths,components=3):
    latents = []
    for path,_ in tqdm(paths):
        h5f = h5py.File(gccadir / path,'r')
        if components is not None:
            latent = h5f[h5_key][:][:,2:2+components]
        else:
            latent = h5f[h5_key][:]
        h5f.close()
        latents.append(latent.reshape(1, -1))
    latents = np.array(latents)
    return(latents.reshape(latents.shape[0],-1))

def main():
    paths = get_files(path=gccadir)
    latents = get_latents(paths)
    labels_all = []
    for _,(trait,_,state) in paths:
        if trait == 'n':
            if 'restingstate' in state:
                labels_all.append(1)
            elif 'compassion' in state:
                labels_all.append(2)
            elif 'openmonitoring' in state:
                labels_all.append(3)
        if trait == 'e':
            if 'restingstate' in state:
                labels_all.append(4)
            elif 'compassion' in state:
                labels_all.append(5)
            elif 'openmonitoring' in state:
                labels_all.append(6)
    
    labels_trait = [int(s) > 3 for s in labels_all]
    labels_state = [int(s)%3 for s in labels_all]

    fit_idx = []
    transform_idx = []
    labels_all = np.array(labels_all)
    for i in range(1,7):
        temp_idx = np.where(labels_all==i)[0]
        idx_half = int(len(temp_idx) / 2)
        fit_idx.append(temp_idx[idx_half:])
        transform_idx.append(temp_idx[:idx_half])

    fit_idx = np.hstack(fit_idx)
    transform_idx = np.hstack(transform_idx)

    labels_shuffled = np.copy(labels_all)
    #np.random.shuffle(labels_shuffled)

    lmao = LOL(n_components=7, svd_solver='full')
    lmao.fit(latents[fit_idx,:], labels_all[fit_idx])
    proj = lmao.transform(latents[transform_idx,:])
    proj_fit = lmao.transform(latents[fit_idx,:])

    df = pd.DataFrame(data=proj)
    df.index=labels_all[transform_idx]
    df.to_csv(savepath, header=False)

    df = pd.DataFrame(data=proj_fit)
    df.index=labels_all[fit_idx]
    df.to_csv(savepath_fit, header=False)

if __name__ == '__main__':
    main()