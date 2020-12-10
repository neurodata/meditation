import argparse
from pathlib import Path
# from mapalign import DiffusionMapEmbedding
from numba import jit
from graspy.embed import MultipleASE
from mvlearn.embed import GCCA
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds
import numpy as np
import os
import h5py
from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from mapalign import embed

import sys; sys.path.append('../')
from src.tools.utils import get_files, read_file
from tqdm import tqdm

def load_subj():
    """
    Loads a subject raw timeseries
    """
    return


#@jit(parallel=True)
def run_perc(data, thresh):
    perc_all = np.zeros(data.shape[0])
    for n,i in enumerate(data):
        data[n, i < np.percentile(i, thresh)] = 0.
    for n,i in enumerate(data):
        data[n, i < 0.] = 0.
    return data


def compute_affinity(K, step2=True):
    """
    Computes a subject affinity matrix, using diffusion map.
    """

    # Calculate Cov mat
    K[np.isnan(K)] = 0.0

    A_mA = K - K.mean(1)[:,None]
    ssA = (A_mA**2).sum(1)
    Asq = np.sqrt(np.dot(ssA[:,None],ssA[None]))
    Adot = A_mA.dot(A_mA.T)

    K = Adot/Asq
    del A_mA, ssA, Asq, Adot
    K = run_perc(K, 90)

    norm = (K * K).sum(0, keepdims=True) ** .5
    K = K.T @ K
    L_alpha = K / norm / norm.T
    
    if not step2:
        return L_alpha
    # Calculate diffusion map
    ndim = L_alpha.shape[0]
    alpha = 0.5

    if alpha > 0:
        # Step 2
        d = np.array(L_alpha.sum(axis=1)).flatten()
        d_alpha = np.power(d, -alpha)
        L_alpha = d_alpha[:, np.newaxis] * L_alpha 
        L_alpha = L_alpha * d_alpha[np.newaxis, :]

    # Step 3
    d_alpha = np.power(np.array(L_alpha.sum(axis=1)).flatten(), -1)

    L_alpha = d_alpha[:, np.newaxis] * L_alpha

    return L_alpha


def embed_all(
    data_dir,
    raw,
    method_name,
    n_components=None,
    exclude_ids=None,
):
    """
    Iteratively loads and partial_fits. Computes group embeddings.
    """
    paths = get_files(path=data_dir, filetype='csv', flag='', subjects_exclude=exclude_ids)

    # Init method, run
    if method_name == 'gcca':
        results_dict, infos = _embed_gcca(paths, raw)
    elif method_name == 'mase':
        assert raw == False
        results_dict, infos = _embed_mase(paths)
    elif method_name == 'svd':
        results_dict, infos = _embed_svd(paths, raw)
    elif method_name == 'dmap':
        results_dict, infos = _embed_dmap(paths)
    else:
        raise ValueError(f'Invalid method input: {method_name}')
    
    return results_dict, infos


def _embed_mase(method, paths):
    infos = []
    results_dict = defaultdict(list)
    method = MultipleASE()

    # X_mean = None
    # print(f'Mean: {datetime.now().strftime("%H:%M:%S")}')
    # for path, _ in tqdm(paths):
    #     X = read_file(data_dir / path, ftype='csv')
    #     X = compute_affinity(X)
    #     if X_mean is None:
    #         X_mean = X / len(paths)
    #     else:
    #         X_mean += X / len(paths)

    print(f'Fit: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        X = read_file(data_dir / path, ftype='csv')
        X = compute_affinity(X)# - X_mean
        method = method.partial_fit(X)
        results_dict['rank'].append(method.ranks_[-1])
        results_dict['eigenvalues'].append(method.Ds_[-1])

    assert len(method.Us_) > 1
    # Compute scores, iteratively
    print(f'Transform: {datetime.now().strftime("%H:%M:%S")}')
    for path, info in tqdm(paths):
        infos.append(info)
        X = read_file(data_dir / path, ftype='csv')
        X = compute_affinity(X)# - X_mean
        if method.latent_right_ is None:
            scores = X @ method.latent_left_
        else:
            scores = X @ method.latent_right_
        results_dict['latent'].append(scores)
        results_dict['latent_right'].append(method.latent_right_)
        results_dict['latent_left'].append(method.latent_left_)
    return results_dict, infos


def _embed_gcca(paths, raw):
    method = GCCA(n_elbows=3, center=raw, max_rank=True)
    infos = []
    results_dict = defaultdict(list)

    print(f'Fit: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        X = read_file(data_dir / path, ftype='csv')
        if not raw:
            X = compute_affinity(X)# - X_mean
        if i < len(paths) - 1:
            method = method.partial_fit([X], multiview_step=False)
        else:
            method = method.partial_fit([X], multiview_step=True)
        results_dict['rank'].append(method.ranks_[-1])
        results_dict['eigenvalues'].append(method._Sall[-1])

    assert len(method._Uall) > 1
    # Compute scores, iteratively
    print(f'Transform: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype='csv')
        if not raw:
            X = compute_affinity(X)
        scores = method.transform([X], view_idx=i)
        results_dict['latent'].append(scores)
    
    return results_dict, infos


def _embed_svd(paths, raw, n_components=5):
    infos = []
    results_dict = defaultdict(list)
    print(f'Fit transform: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype='csv')
        if not raw:
            X = compute_affinity(X)# - X_mean
            evals, evecs = eigsh(X, k=n_components + 1)
            sort_idx = np.argsort(evals)[::-1]
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]
            lambdas = evals[1:] / (1 - evals[1:])
            vectors = evecs[:, 1:]# / evecs[:, [0]]
            vectors = vectors @ np.diag(lambdas)
        else:
            X -= X.mean(0)
            u, s, _ = svds(X, k=n_components)
            sort_idx = np.argsort(s)[::-1]
            evecs = u[:, sort_idx]
            evals = s[sort_idx]
            vectors = evecs @ np.diag(evals)
        
        results_dict['rank'].append(n_components)
        results_dict['eigenvalues'].append(evals)
        results_dict['eigenvectors'].append(evecs)
        results_dict['latent'].append(vectors)
    
    return results_dict, infos


def _embed_dmap(paths):
    infos = []
    results_dict = defaultdict(list)

    print(f'Fit transform: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype='csv')
        X = compute_affinity(X, step2=False)
        emb, res = embed.compute_diffusion_map(X, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True, eigen_solver=eigsh, return_result=True)
        results_dict['rank'].append(res['n_components'])
        results_dict['eigenvalues'].append(res['lambdas'])
        results_dict['latent'].append(emb)
    
    return results_dict, infos


def save_embeddings(save_dir, results_dict, infos, tag, method):
    """
    Saves the group embeddings.
    """
    # Make directory to save to
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save
    assert len(infos) > 1
    for i, (level, subj, task) in enumerate(infos):
        if tag is not None:
            save_path = save_dir / f'{level}_sub-{subj}_ses-1_task-{task}_{method}.h5'
        else:
            save_path = save_dir / f'{level}_sub-{subj}_ses-1_task-{task}_{method}.h5'
        h5f = h5py.File(save_path, 'w')
        for key, val in results_dict.items():
            if val[i] is not None:
                h5f.create_dataset(key, data=val[i])
        h5f.close()

    return

def main(
    data_dir,
    raw,
    save_dir,
    method_name,
    n_components=None,
    exclude_ids=None,
    tag=None,
):
    """
    Iteratively loads subjects and partial fits. Saves group embeddings.
    """

    # Get embeddings
    results_dict, infos = embed_all(data_dir, raw, method_name, n_components, exclude_ids)

    # Save embeddings        
    save_embeddings(save_dir, results_dict, infos, tag, method_name)
    

if __name__ == '__main__':
    # Constants
    DATA_DIR = Path('/mnt/ssd3/ronan/data')
    RAW_DIR = DATA_DIR / 'raw'

    # Variable inputs
    parser = argparse.ArgumentParser()

    # Mandatory
    parser.add_argument('--method', type=str, help="list servers, storage, or both (default: %(default)s)", choices=['gcca', 'mase', 'svd', 'dmap'])
    # Optional
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('--source', action='store', default=RAW_DIR)
    parser.add_argument('--n-components', type=int, default=None)
    parser.add_argument('--save', action='store', default=None)
    parser.add_argument('--tag', action='store', type=str, default=None)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str, default=None)

    args = vars(parser.parse_args())

    method = args['method']
    raw = args['raw']
    data_dir = args['source']
    save_dir = args['save']
    n_components = args['n_components']
    exclude_ids = args['exclude_ids']
    tag = args['tag']
    if save_dir is None:
        if tag is None: 
            save_dir = DATA_DIR / f'{method}_{datetime.now().strftime("%m-%d")}'
        else:
            save_dir = DATA_DIR / f'{method}_{tag}_{datetime.now().strftime("%m-%d")}'
    print(f'Saving Dir: {save_dir}')
    # Run
    main(data_dir, raw, save_dir, method, n_components, exclude_ids, tag)

# python embed_dmap_affinity.py --method gcca --tag raw --exclude-ids 073 --raw