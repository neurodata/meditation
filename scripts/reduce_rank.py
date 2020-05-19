import numpy as np
from pathlib import Path
import h5py
import pandas as pd
import re
from tqdm import tqdm
import os

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
savedir = datadir / 'gcca_raw-True_ZG-3_reduced'
gccadir = datadir / 'gcca_raw-True_ZG-3'
decimate_dir = datadir / 'decimate'
logpath = Path('.')

## Stuff
h5_key = 'latent'
tasks = ['restingstate', 'openmonitoring', 'compassion']
levels = ['e', 'n']
reduced_component_rank = 3

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

## Get GCCA vectors
def reduce_latents(paths,components=reduced_component_rank):
    for path,_ in tqdm(paths):
        h5f = h5py.File(gccadir / path,'r')
        if components is not None:
            latent = h5f[h5_key][:][:,:reduced_component_rank]
        else:
            latent = h5f[h5_key][:]
        h5f.close()
        h5f = h5py.File(savedir / path, 'w')
        h5f.create_dataset('latent', data=latent)
        h5f.close()

def main():
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    paths = get_files(path=gccadir)
    reduce_latents(paths)

if __name__ == '__main__':
    main()