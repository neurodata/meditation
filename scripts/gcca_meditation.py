# %% Imports
import numpy as np
import argparse
from mvlearn.embed.gcca import GCCA
from pathlib import Path
from datetime import datetime
import logging
import os
import re
import h5py
import sys; sys.path.append('../')
from src.tools.utils import get_files, read_file
from src.tools.split_sample import split_sample


def embed_all(
    data_dir,
    save_dir,
    logging,
    n_components=4,
    rank_tolerance=0.1,
    tag='',
    ftype='csv',
    h5_key=None,
    n_elbows=2,
    paths=None,
    max_rank=False,
    transpose=False,
    exclude_ids=None,
):
    # Make directory to save to
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # Get filenames for each task, novice vs. experienced
    tasks = ['restingstate', 'openmonitoring', 'compassion']
    levels = ['e', 'n']

    logging.info(f'Pulling data from {data_dir}')

    if not paths:
        paths = get_files(path=data_dir, filetype=ftype, flag=tag, subjects_exclude=exclude_ids)
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
    if transpose:
        latents = gcca.fit_transform([X.T for X in raw_data if len(X.T) == 300])
    else:
        latents = gcca.fit_transform(raw_data)

    logging.info(f'Ranks are {gcca.ranks_}')
    logging.info(f'Group embedding dimensions: {latents.shape}')

    # Save latents
    logging.info(f'Saving reduce correlations to {save_dir}')
    for info, latent, proj_mat, svals in zip(infos, latents, gcca.projection_mats_, gcca._Sall):
        level, subj, task = info
        save_path = save_dir / f'{level}_sub-{subj}_ses-1_task-{task}_gcca.h5'
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('latent', data=latent)
        h5f.create_dataset('projection', data=proj_mat)
        h5f.create_dataset('svals', data=svals)
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
    parser.add_argument('--max-rank', action='store_true', default=False)
    parser.add_argument(
        '--save-dir',
        action='store',
        default=DATA_DIR / f'gcca_{datetime.now().strftime("%m-%d-%H:%M")}'
        )
    parser.add_argument('--tag', action='store', default=False)
    parser.add_argument('--split-half', action='store_true')
    parser.add_argument('--transpose', action='store_true', default=False)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str)

    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    if args['tag']:
        save_dir = Path(str(save_dir) + f"_{args['tag']}")
    if args['transpose']:
        save_dir = Path(str(save_dir) + f"_transpose")

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
            max_rank = args['max_rank'],
            transpose = args['transpose'],
            exclude_ids = args['exclude_ids']
        )
