# %% Imports
import numpy as np
import argparse
from mvlearn.embed.gcca import GCCA
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import os
import re
import h5py
import sys; sys.path.append('../')
from src.tools.split_sample import split_sample
from multiviewica import multiviewica

# Grab filenames
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

# Read a csv or h5 file. Meta corresponds to h5_key


def read_file(path, ftype, h5_key=None):
    if ftype == 'csv':
        return(pd.read_csv(path, header=None).to_numpy())
    elif ftype == 'h5':
        h5f = h5py.File(path, 'r')
        temp = h5f[h5_key][:]
        h5f.close()
        return(temp)


def embed_all(method, data_dir, save_dir, logging, n_components=4, rank_tolerance=0.1,
         tag='', ftype='csv', h5_key=None, n_elbows=2, paths=None, max_rank=False):
    
    if method not in ['gcca', 'multiviewica']:
        print('`method` must be one of [gcca, multiviewica]')
        sys.exit(1)
    # Make directory to save to
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get filenames for each task, novice vs. experienced
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']

    logging.info(f'Pulling data from {data_dir}')

    if not paths:
        paths = get_files(path=data_dir, ftype=ftype, flag=tag)
    raw_data = []
    infos = []
    for path, info in paths:
        raw_data.append(read_file(data_dir / path, ftype, h5_key=h5_key))
        infos.append(info)

    # Create GCCA
    if method == 'gcca':
        logging.info(f'Performing GCCA')
        if max_rank:
            gcca = GCCA(n_elbows=n_elbows, max_rank=True)
        else:
            gcca = GCCA(n_elbows=n_elbows)
        result_dict = {'latents':gcca.fit_transform(raw_data)}
        logging.info(f'Ranks are {gcca.ranks_}')
        logging.info(f'Group embedding dimensions: {latents.shape}')
    elif method == 'multiviewica':
        logging.info(f'Performing multiviewICA')
        assert(np.asarray(raw_data[0]).shape[0] == 18715)
        W, S = multiviewica([np.asarray(x).T for x in raw_data], tol=1e-4, max_iter=10000, n_components=n_components)
        assert(len(W) == len(raw_data))
        result_dict = {'unmixing':W, 'source':[S for _ in range(len(W))]}

    # Save latents
    logging.info(f'Saving results to {save_dir}')
    for i, info in enumerate(infos):
        level, subj, task = info
        save_path = save_dir / f'{level}_sub-{subj}_ses-1_task-{task}_{method}.h5'
        h5f = h5py.File(save_path, 'w')
        for key in result_dict.keys():
            h5f.create_dataset(key, data=result_dict[key][i])
        h5f.close()

if __name__ == '__main__':
    # %% Define paths
    DATA_DIR = Path('/mnt/ssd3/ronan/data')
    RAW_DIR = DATA_DIR / 'raw'
    log_path = Path('../logs')

    parser = argparse.ArgumentParser()
    # Mandatory
    parser.add_argument('method')

    # Optional
    parser.add_argument('--data-dir', action='store', default=RAW_DIR)
    parser.add_argument('--elbos', action='store', type=int, default=2)
    parser.add_argument('--max-rank', action='store_true')
    parser.add_argument('--n-components', action='store', type=int, default=4)
    parser.add_argument(
        '--save-dir',
        action='store',
        default=DATA_DIR / f'gcca_{datetime.now().strftime("%m-%d-%H:%M")}'
        )
    parser.add_argument('--tag', action='store', default=False)
    parser.add_argument('--split-half', action='store_true')

    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    if args['tag']:
        save_dir = Path(str(save_dir) + f"_{args['tag']}")

    # Create Log File
    logging.basicConfig(filename=log_path / 'logging.log',
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG
                        )
    logging.info(f'NEW {args["method"]} RUN')
    logging.info('Args:')
    logging.info(args)
    print()
    embed_all(
        method = args['method'],
        data_dir = args['data_dir'],
        save_dir = save_dir,
        logging = logging,
        n_elbows = args['elbos'],
        max_rank = args['max_rank'],
        n_components = args['n_components'],
    )
