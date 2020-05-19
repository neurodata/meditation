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


def embed_all(data_dir, save_dir, logging, n_components=4, rank_tolerance=0.1,
         tag='', ftype='csv', h5_key=None, n_elbows=2, paths=None, max_rank=False):
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
    logging.info(f'Performing GCCA')
    if max_rank:
        gcca = GCCA(n_elbows=n_elbows, max_rank=True)
    else:
        gcca = GCCA(n_elbows=n_elbows)
    latents = gcca.fit_transform(raw_data)

    logging.info(f'Ranks are {gcca.ranks_}')
    logging.info(f'Group embedding dimensions: {latents.shape}')

    # Save latents
    logging.info(f'Saving reduce correlations to {save_dir}')
    for info, latent in zip(infos, latents):
        level, subj, task = info
        save_path = save_dir / f'{level}_sub-{subj}_ses-1_task-{task}_gcca.h5'
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('latent', data=latent)
        h5f.close()

if __name__ == '__main__':
    # %% Define paths
    DATA_DIR = Path('/mnt/ssd3/ronan/data')
    RAW_DIR = DATA_DIR / 'raw'
    log_path = Path('../logs')

    parser = argparse.ArgumentParser()
    # Mandatory

    # Optional
    parser.add_argument('--data-dir', action='store', default=RAW_DIR)
    parser.add_argument('--elbos', action='store', type=int, default=2)
    parser.add_argument('--max-rank', action='store_true')
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
    logging.info('NEW GCCA RUN')
    logging.info('Args:')
    logging.info(args)

    if args['split_half']:
        logging.info('SPLIT HALF')
        e_paths = get_files(path=args['data_dir'], level='(e)')
        n_paths_1, n_paths_2 = split_sample(path=args['data_dir'], level='(n)')
        logging.info('SPLIT HALF 1')
        embed_all(
            data_dir = args['data_dir'],
            save_dir = Path(save_dir) / f'split_half_1',
            logging = logging,
            n_elbows = args['elbos'],
            paths = e_paths + n_paths_1
        )

        logging.info('SPLIT HALF 2')
        embed_all(
            data_dir = args['data_dir'],
            save_dir = Path(save_dir) / f'split_half_2',
            logging = logging,
            n_elbows = args['elbos'],
            paths = e_paths + n_paths_2
        )

    else:
        embed_all(
            data_dir = args['data_dir'],
            save_dir = save_dir,
            logging = logging,
            n_elbows = args['elbos'],
            max_rank = args['max_rank']
        )
