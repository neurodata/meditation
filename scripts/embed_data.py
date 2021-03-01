import argparse
from pathlib import Path
# from mapalign import DiffusionMapEmbedding
from numba import jit
from graspy.embed import MultipleASE
from mvlearn.embed import GCCA
from mvlearn.embed.utils import select_dimension
from mvlearn.decomposition import GroupPCA, GroupICA
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
from src.tools.preprocess import procrustes
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


def _processed_cov_mat(K):
    K = (K.T - np.nanmean(K, axis = 1)).T
    K = (K.T / np.nanstd(K, axis = 1)).T
    K[np.isnan(K)] = 0.0

    A_mA = K - K.mean(1)[:,None]
    ssA = (A_mA**2).sum(1)
    Asq = np.sqrt(np.dot(ssA[:,None],ssA[None]))
    Adot = A_mA.dot(A_mA.T)

    K = Adot/Asq
    del A_mA, ssA, Asq, Adot
    K = run_perc(K, 90)

    return K

def _cos_similarity(K, K2=None):
    if K2 is None:
        K2 = K
    norm = (K * K).sum(0, keepdims=True) ** .5
    norm2 = (K2 * K2).sum(0, keepdims=True) ** .5
    L_alpha = K.T @ K2
    L_alpha = L_alpha / norm.T / norm2
    return L_alpha

def compute_affinity(K, K2=None, step2=True):
    """
    Computes a subject affinity matrix, using diffusion map.
    """

    # Calculate Cov mat
    K = _processed_cov_mat(K)
    if K2 is None:
        K2 = K
    else:
        K2 = _processed_cov_mat(K2)

    L_alpha = _cos_similarity(K, K2)
    
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
    data,
    method_name,
    n_components=None,
    exclude_ids=None,
    ftype='csv',
):
    """
    Iteratively loads and partial_fits. Computes group embeddings.
    """
    paths = get_files(path=data_dir, filetype='csv', flag='', subjects_exclude=exclude_ids)
    
    # Init method, run
    if method_name == 'gcca':
        results_dict, infos = _embed_gcca(paths, data, ftype)
    elif method_name == 'mase':
        assert data != 'raw'
        results_dict, infos = _embed_mase(paths, ftype, data)
    elif method_name == 'svd':
        results_dict, infos = _embed_svd(paths, data, ftype)
    elif method_name == 'dmap':
        results_dict, infos = _embed_dmap(paths, ftype)
    elif method_name == 'joint':
        results_dict, infos = _joint_embedding(paths, ftype)
    elif method_name == 'grouppca':
        results_dict, infos = _embed_grouppca(paths, ftype)
    elif method_name == 'groupica':
        results_dict, infos = _embed_groupica(paths, ftype)
    else:
        raise ValueError(f'Invalid method input: {method_name}')
    
    return results_dict, infos


def _embed_mase_helper(path, ftype, info, data):
    X = read_file(path, ftype=ftype, info=info)
    if data == 'dmap':
        X = compute_affinity(X)# - X_mean
    elif data == 'aff':
        X = compute_affinity(X, step2=False)

    return X, info

def _embed_mase(paths, ftype, data):
    results_dict = defaultdict(list)
    mase = MultipleASE(scaled=False)
    print(f'Fit ({ftype}, {data}): {datetime.now().strftime("%H:%M:%S")}')
    # Xs = Parallel(n_jobs=45, verbose=10)(delayed(_embed_mase_helper)(data_dir / f, ftype, info, data) for f, info in paths)
    # Xs, infos = list(zip(*Xs))
    
    for i, (path, info) in enumerate(tqdm(paths)):
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            X = compute_affinity(X)# - X_mean
        elif data == 'aff':
            X = compute_affinity(X, step2=False)
        else:
            raise ValueError(f'{data} is not a valid argument')
        mase = mase.partial_fit(X, final_fit=(i==len(paths)-1))
    results_dict['ranks'] = mase.ranks_
    results_dict['eigenvalues'] = mase.Ds_

    assert len(mase.Us_) > 1
    # Compute scores, iteratively
    print(f'Transform ({ftype}, {data}): {datetime.now().strftime("%H:%M:%S")}')
    infos = []
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            X = compute_affinity(X)# - X_mean
        elif data == 'aff':
            X = compute_affinity(X, step2=False)
        else:
            raise ValueError(f'{data} is not a valid argument')
        if mase.latent_right_ is None:
            scores = mase.latent_left_.T @ X @ mase.latent_left_
        else:
            scores = mase.latent_left_.T @ X @ mase.latent_right_
        results_dict['latent'].append(mase.latent_left_ @ scores)
        results_dict['Us'].append(mase.Us_[i])
        results_dict['latent_right'].append(mase.latent_right_)
        results_dict['latent_left'].append(mase.latent_left_)

    return results_dict, infos


def _embed_gcca(paths, data, ftype):
    method = GCCA(n_components=5, center=(data=='raw'), max_rank=False)
    infos = []
    results_dict = defaultdict(list)

    print(f'Fit: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            X = compute_affinity(X)# - X_mean
        elif data == 'aff':
            X = compute_affinity(X, step2=False)
        else:
            # Done implicitly within gcca, but shouldn't
            X -= X.mean(1, keepdims=True)
            X /= X.std(1, keepdims=True)

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
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            X = compute_affinity(X)# - X_mean
        elif data == 'aff':
            X = compute_affinity(X, step2=False)
        scores = method.transform([X], view_idx=i)
        results_dict['latent'].append(scores)
    
    return results_dict, infos


def _embed_svd(paths, data, ftype, n_components=5):
    infos = []
    results_dict = defaultdict(list)
    print(f'Fit transform: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            X = compute_affinity(X)# - X_mean
            evals, evecs = eigsh(X, k=n_components + 1)
            sort_idx = np.argsort(evals)[::-1]
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]
            lambdas = evals[1:]  / (1 - evals[1:])
            vectors = evecs[:, 1:]# / evecs[:, [0]]
            vectors = vectors @ np.diag(lambdas)
        else:
            if data == 'raw':
                X -= X.mean(0)
            elif data == 'aff':
                X = compute_affinity(X, step2=False)
            else:
                raise ValueError(f'{data} is not a valid argument')
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


def _embed_dmap(paths, ftype):
    infos = []
    results_dict = defaultdict(list)

    print(f'Fit transform: {datetime.now().strftime("%H:%M:%S")}')
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype=ftype, info=info)
        X = compute_affinity(X, step2=False)
        emb, res = embed.compute_diffusion_map(X, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True, eigen_solver=eigsh, return_result=True)
        results_dict['rank'].append(res['n_components'])
        results_dict['eigenvalues'].append(res['lambdas'])
        results_dict['latent'].append(emb)
    
    return results_dict, infos


def _joint_embedding(paths, ftype):
    """
    Performs a joint embedding. Embeds each measurement with the group average
    per (Xu 2020): "Joint embedding: A scalable alignment to compare
    individuals in a connectivity space" 

    X : a raw voxel x timeseries scan
    C : covariance matrix
    W : cosine similarity matrix between covariance matrices
    E : diffusion map embeddings of cosine similarity
    """
    C_mean = None
    print(f'Mean: {datetime.now().strftime("%H:%M:%S")}')
    for path, info in tqdm(paths):
        X = read_file(data_dir / path, ftype=ftype, info=info)
        C = _processed_cov_mat(X)
        if C_mean is None:
            C_mean = C / len(paths)
        else:
            C_mean += C / len(paths)
        del X, C

    # Diffusion embedding of average connectivity matrix
    W_mean = _cos_similarity(C_mean)
    E_mean, res = embed.compute_diffusion_map(
        W_mean, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True,
        eigen_solver=eigsh, return_result=True)

    print(f'Joint: {datetime.now().strftime("%H:%M:%S")}')
    infos = []
    results_dict = defaultdict(list)
    for path, info in tqdm(paths):
        infos.append(info)
        X = read_file(data_dir / path, ftype=ftype, info=info)
        n = X.shape[0]
        C = _processed_cov_mat(X)
        W12 = _cos_similarity(C_mean, C)
        # Joint matrix of subject and average connectivity
        W_joint = np.block([
            [W_mean, W12],
            [W12.T, _cos_similarity(C)]
        ])
        del X, C, W12

        # Diffusion embedding of joint
        E_joint, res = embed.compute_diffusion_map(
            W_joint, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True,
            eigen_solver=eigsh, return_result=True)
        results_dict['rank'].append(res['n_components'])
        results_dict['eigenvalues'].append(res['lambdas'])
        
        E_joint_mean = E_joint[:n]
        E_subj = E_joint[n:]
        del E_joint, W_joint

        # Align joint group portion to the group embedding. Use same transform on individual
        E_joint_mean, xfm = procrustes(E_joint_mean, E_mean, return_transform=True)
        alignment_score = np.linalg.norm(E_joint_mean - E_mean, axis=0)
        results_dict['align_scores'].append(alignment_score)
        E_subj_aligned = E_subj.dot(xfm)

        results_dict['latent'].append(E_subj_aligned)
        del E_subj_aligned, E_joint_mean, E_subj
    
    return results_dict, infos

def _embed_grouppca(paths, ftype, n_components=5):
    return _embed_group_wrapper(paths, ftype, n_components, GroupPCA)

def _embed_groupica(paths, ftype, n_components=5):
    return _embed_group_wrapper(paths, ftype, n_components, GroupICA)

def _embed_group_wrapper(paths, ftype, n_components, model, prewhiten=True):
    infos = []
    results_dict = {}
    print(f'Fit transform: {datetime.now().strftime("%H:%M:%S")}')

    Xs = []
    elbows = []
    for i, (path, info) in enumerate(tqdm(paths)):
        infos.append(info)
        X = read_file(data_dir / path, ftype=ftype, info=info)
        if data == 'dmap':
            raise ValueError('Not yet implemented')
            # X = compute_affinity(X)# - X_mean
            # evals, evecs = eigsh(X, k=n_components + 1)
            # sort_idx = np.argsort(evals)[::-1]
            # evals = evals[sort_idx]
            # evecs = evecs[:, sort_idx]
            # lambdas = evals[1:]  / (1 - evals[1:])
            # vectors = evecs[:, 1:]# / evecs[:, [0]]
            # vectors = vectors @ np.diag(lambdas)
        elif data == 'raw':
            X -= X.mean(1, keepdims=True)
            X /= X.std(1, keepdims=True)
            _, s, _ = svds(X, k=50)
            s = np.sort(s)[::-1]
            (elbow1, elbow2, elbow3), _ = select_dimension(s, n_elbows=3)
            elbows.append(elbow2)
            Xs.append(X)
            del X

    embeddor = model(n_components=n_components, n_individual_components=elbows, prewhiten=prewhiten)
    Xs = embeddor.fit_transform(Xs)
        
    results_dict['rank'] = [n_components]*len(Xs)
    results_dict['elbows'] = elbows
    results_dict['eigenvalues'] = np.linalg.norm(Xs, axis=1)
    results_dict['latent'] = Xs

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
    data,
    save_dir,
    method_name,
    n_components=None,
    exclude_ids=None,
    tag=None,
    type='csv',
    debug=False,
):
    """
    Iteratively loads subjects and partial fits. Saves group embeddings.
    """

    # Get embeddings
    results_dict, infos = embed_all(data_dir, data, method_name, n_components, exclude_ids, ftype)

    # Save embeddings     
    if not debug:   
        save_embeddings(save_dir, results_dict, infos, tag, method_name)
    

if __name__ == '__main__':
    # Constants
    DATA_DIR = Path('/mnt/ssd3/ronan/data')
    RAW_DIR = DATA_DIR / 'raw'

    # Variable inputs
    parser = argparse.ArgumentParser()

    # Mandatory
    parser.add_argument('--method', type=str, help="", choices=['gcca', 'mase', 'svd', 'dmap', 'joint', 'grouppca', 'groupica'])
    # Optional
    parser.add_argument('--ftype', type=str, choices=['csv', 'mgz'], default='csv')
    parser.add_argument('--data', type=str, default='dmap', choices=['dmap', 'raw', 'aff'])
    parser.add_argument('--source', action='store', default=RAW_DIR)
    parser.add_argument('--n-components', type=int, default=None)
    parser.add_argument('--save', action='store', default=None)
    parser.add_argument('--tag', action='store', type=str, default=None)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str, default=None)
    parser.add_argument('--debug', default=False, action='store_true')

    args = vars(parser.parse_args())

    method = args['method']
    data = args['data']
    data_dir = args['source']
    save_dir = args['save']
    n_components = args['n_components']
    exclude_ids = args['exclude_ids']
    tag = args['tag']
    ftype = args['ftype']
    debug = args['debug']
    if save_dir is None:
        if tag is None: 
            save_dir = DATA_DIR / f'{method}_{datetime.now().strftime("%m-%d")}'
        else:
            save_dir = DATA_DIR / f'{method}_{tag}_{datetime.now().strftime("%m-%d")}'
    if debug:
        print('Debug mode, will not save')
    else:
        print(f'Saving Dir: {save_dir}')

    # Run
    main(data_dir, data, save_dir, method, n_components, exclude_ids, tag, ftype, debug)

# python embed_dmap_affinity.py --method gcca --tag raw --exclude-ids 073 --data raw
# python3 embed_dmap_affinity.py --method dmap --ftype mgz --exclude-ids 073 --tag replication