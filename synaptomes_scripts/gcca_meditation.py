## Imports
import numpy as np
from multiview.embed.gcca import GCCA
import pandas as pd
from pathlib import Path
import logging
import os
import re
import sys
import h5py

## Define paths
datadir = Path('/mnt/ssd3/ronan/data')
rawdir = datadir / 'raw'
decimate_dir = datadir / 'decimate'
logpath = Path('.')

## Grab filenames
def get_files(path,
              level='(e|n)',
              subject='([0-9]{3})',
              task='(.+?)',
              ftype='csv',
              flag=''):
    files = []
    query = f'^{level}_sub-'
    query += f'{subject}_ses-1_'
    query += f'task-{task}{flag}\.{ftype}'
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))
    
    return(files)

## Read a csv or h5 file. Meta corresponds to h5_key
def read_file(path, ftype, h5_key=None):
    if ftype == 'csv':
        return(pd.read_csv(path, header = None).to_numpy())
    elif ftype == 'h5':
        h5f = h5py.File(path,'r')
        temp = h5f[h5_key][:]
        h5f.close()
        return(temp)

def main(n_components = 4, rank_tolerance=0.1, data_source=rawdir, 
        tag='', ftype='csv', h5_key=None, n_elbows=2):
    ## Create Log File
    logging.basicConfig(filename=logpath / 'logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG
                        )
    logging.info('NEW GCCA RUN')

    ## Get filenames for each task, novice vs. experienced
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']

    logging.info(f'Pulling data from {data_source}')

    paths = get_files(path=data_source, ftype=ftype, flag=tag)
    raw_data = []
    infos = []
    for path,info in paths:
        raw_data.append(read_file(data_source / path, ftype, h5_key=h5_key))
        infos.append(info)
    #print(paths)

    ## Create GCCA
    logging.info(f'Performing GCCA')
    gcca = GCCA(n_elbows=n_elbows)
    latents = gcca.fit_transform(raw_data)

    logging.info(f'Ranks are {gcca.ranks_}')
    logging.info(f'Min rank: {min(gcca.ranks_)}')

    ## Save latents
    logging.info(f'Saving reduce correlations to {gccadir}')
    for info,latent in zip(infos, latents):
        level,subj,task = info
        save_path = gccadir / f'{level}_sub-{subj}_ses-1_task-{task}_gcca-zg.h5'
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('latent', data=latent)
        h5f.close()

if __name__ == '__main__':
    RAW = True
    data_source=decimate_dir
    n_elbows=3
    if len(sys.argv) > 1:
        param = int(float(sys.argv[1]))
        gccadir = datadir / f'gcca_raw-{RAW}_param-{param}'
    else:
        param=None
        gccadir = datadir / f'gcca_raw-{RAW}_ZG-{n_elbows}'
    if not os.path.exists(gccadir): 
        os.makedirs(gccadir)
    tag='_decimate'; ftype='h5'; h5_key='decimated'
    if param:
        if RAW:
            main(rank_tolerance=param, n_elbows=n_elbows)
        else:
            main(rank_tolerance=param, data_source=data_source,
                tag=tag,ftype=ftype,h5_key=h5_key)
    else:
        if RAW:
            main(n_elbows=n_elbows)
        else:
            main(data_source=data_source,
                tag=tag,ftype=ftype,h5_key=h5_key)