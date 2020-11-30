#-*- coding: utf-8 -*-

import numpy as np
import os
import re
import h5py
import pandas as pd
from collections import defaultdict

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
              flag='',
              source='gcca',
              subjects_exclude=None,
              ):
    """
    Loads files from a directory

    Returns
    -------
    list of tuples, each (path, groups)
        groups is a tuple depending on which the inputs

        gcca : (level, subject, task, flag, filetype)

        dmap :

    """
    files = []
    if isinstance(subjects_exclude, (int, str)):
        subjects_exclude = [subjects_exclude]
    if subjects_exclude is not None:
        subject = f'(?!(?:{"|".join([f"{id:0>3}" for id in subjects_exclude])})){subject}'
    if source == 'gcca' or source == 'dmap':
        query = f'^{level}_sub-'
        query += f'{subject}_ses-1_'
        query += f'task-{task}{flag}\.{filetype}'
    elif source == 'dmap_raw':
        query = f'^{level}_embedding_dense{flag}'
        query += f'\.sub-{subject}'
        query += f'\.{task}\.{filetype}'
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))
    
    return(files)

# Read a csv or h5 file. Meta corresponds to h5_key
def read_file(path, ftype, h5_key=None):
    if ftype == 'csv':
        return(pd.read_csv(path, header=None).to_numpy())
    elif ftype == 'h5':
        h5f = h5py.File(path, 'r')
        temp = h5f[h5_key][:]
        h5f.close()
        return(temp)

def get_latents(data_dir, n_components=None, flag='_gcca', ids=False, ftype='h5', source='gcca', subjects_exclude=None, as_groups=True, h5_key='latent'):
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']

    latents = []
    labels = []
    subj_ids = []

    for level in levels:
        for task in tasks:
            subgroup = []
            labels.append([level, task])
            paths = get_files(path=data_dir, level=level, task=task, flag=flag, filetype=ftype, source=source, subjects_exclude=subjects_exclude)
            n_load = len(paths)
            subjs = []

            for path,subj in paths[:n_load]:
                if ftype == 'h5':
                    h5f = h5py.File(data_dir / path,'r')
                    latent = h5f[h5_key][:][:,:n_components]
                    h5f.close()
                elif ftype == 'npy':
                    latent = np.load(data_dir / path, allow_pickle=True)[:, :n_components]
                else:
                    raise ValueError(f'Invalid ftype {ftype}')
                subgroup.append(latent)
                subjs.append(subj[0])
            latents.append(subgroup)
            subj_ids.append(subjs)
    if not as_groups:
        labels = np.vstack([[label]*len(latent) for label,latent in zip(labels,latents)])
        latents = np.vstack([np.asarray(l) for l in latents])
        if ids:
            subj_ids = np.hstack(subj_ids)
    if ids:
        return latents, labels, subj_ids
    else:
        return latents, labels

def get_h5(data_dir, flag='_gcca'):
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']

    data_dict = defaultdict(list)
    labels = []
    subj_ids = []

    for level in levels:
        for task in tasks:
            subgroup = []
            labels.append([level, task])
            paths = get_files(path=data_dir, level=level, task=task, flag=flag, filetype='h5')
            n_load = len(paths)
            #subjs = []

            for path,subj in paths[:n_load]:
                h5f = h5py.File(data_dir / path,'r')
                for key in h5f.keys():
                    data_dict[key].append(h5f[key][:])
                h5f.close()
                data_dict['state'].append(task)
                data_dict['trait'].append(level)
                data_dict['subject'].append(subj)
            
                #subjs.append(subj[0])
            #subj_ids.append(subjs)
    return data_dict