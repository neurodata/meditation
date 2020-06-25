#-*- coding: utf-8 -*-

import numpy as np
import os
import re
import h5py

def decimate_ptr(X, nbins=1000):
    '''
    Places the flattened data into bins, equally spaced from 
    the minimum to the maximum. Data is reassigned the index of the bin
    and reshaped.
    
    Parameters
    ----------
    X : numpy.array, shape (n, m)
        data to decimate
    bins: int
        number of bins to decimate the data into
    '''
    
    size = X.shape
    X = X.flatten()

    bins = np.linspace(min(X), max(X), nbins+1)
    bins[-1] += 1 #To include the max element in the last bin
    X = np.digitize(X, bins)
    
    return(X.reshape(size))

def get_files(path,
              level='(e|n)',
              subject='([0-9]{3})',
              task='(.+?)',
              filetype='csv',
              flag=''):
    files = []
    query = f'^{level}_sub-'
    query += f'{subject}_ses-1_'
    query += f'task-{task}{flag}\.{filetype}'
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))
    
    return(files)

def get_latents(data_dir, n_components=-1, flag='_gcca',ids=False):
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']
    h5_key = 'latent'

    latents = []
    labels = []
    subj_ids = []

    for level in levels:
        for task in tasks:
            subgroup = []
            labels.append([level, task])
            paths = get_files(path=data_dir, level=level, task=task, flag=flag, filetype='h5')
            n_load = len(paths)
            subjs = []

            for path,subj in paths[:n_load]:
                h5f = h5py.File(data_dir / path,'r')
                latent = h5f[h5_key][:][:,:n_components]
                h5f.close()
            
                subgroup.append(latent)
                subjs.append(subj[0])
            latents.append(subgroup)
            subj_ids.append(subjs)
    if ids:
        return latents, labels, subj_ids
    else:
        return(latents, labels)