import numpy as np
import argparse
from pathlib import Path
import os
import re
import h5py
import sys; sys.path.append('../')
from src.tools import align, iterate_align
from src.tools import get_files, get_latents
from collections import defaultdict


def load_data(source_dir, data, exclude_ids):
    source_dir = Path(source_dir)
    if data == 'antoine':
        flag = "_emb"
        SOURCE = 'dmap'
        ftype = 'npy'
    elif data == 'gcca':
        flag = "_gcca"
        SOURCE = 'gcca'
        ftype = 'h5'
    elif data == 'svd':
        flag = '_svd'
        SOURCE = 'gcca'
        ftype = 'h5'
    elif data == 'dmap':
        flag = '_dmap'
        SOURCE = 'gcca'
        ftype = 'h5'
    return get_latents(
        source_dir, flag=flag, ids=True, ftype=ftype, source=SOURCE, subjects_exclude=exclude_ids, as_groups=False
    )


def save_data(target_dir, data, components, labels, subjs):
    target_dir = Path(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for group, (trait, state), subj in zip(components, labels, subjs):
        if data == 'antoine':
            data = 'dmap'
        #     np.save(target_dir / f'{trait}_sub-{subj}_ses-1_task-{state}_{data}.npy', group)
        # else: # gcca, svd, etc
        h5f = h5py.File(target_dir / f'{trait}_sub-{subj}_ses-1_task-{state}_{data}.h5', 'w')
        h5f.create_dataset('latent', data=group)
        h5f.close()


def group_svd(Xs: np.ndarray, scaled=False) -> np.ndarray:
    from scipy.sparse.linalg import svds
    from sklearn.preprocessing import normalize
    # Create a concatenated view of Us
    Sall = np.linalg.norm(Xs, axis=1)
    if scaled:
        Uall = Xs
    else:
        Uall = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    Uall_c = np.concatenate(Uall, axis=1)

    d = Sall.shape[1]
    UU, SS, VV = svds(Uall_c, d)
    sort_idx = np.argsort(SS)[::-1]
    SS = SS[sort_idx]
    UU = UU[:, sort_idx]
    VV = VV.T[:, sort_idx]
    VV = VV[:, :d]
    
    # SVDS the concatenated Us
    idx_end = 0
    projection_mats = []
    n = len(Xs)
    for i in range(n):
        idx_start = idx_end
        idx_end = idx_start + d
        VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)

        # Compute the canonical projections, unnormalized
        A = VVi # np.linalg.solve(np.diag(Sall[i]), VVi)
        projection_mats.append(A)
        
    return np.asarray([X @ pmat for X, pmat in zip(Xs, projection_mats)])


# Should use --norm by default
if __name__ == "__main__":
    # python3 align_gradients.py --source 
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source directory with files", type=str, required=True)
    parser.add_argument("-t", "--save", help="target directory to save files", type=str, required=True)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str)
    parser.add_argument("--thresh", help="threshold to stop iterative align", type=float, default=0)
    parser.add_argument("--max-iter", help="max number of iterations for the alignment", type=int, default=5)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("-d", "--data", help="list servers, storage, or both (default: %(default)s)", choices=['gcca', 'dmap', 'svd', 'antoine'], default="dmap")
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--fnorm", action='store_true', default=False)
    parser.add_argument("--mean-align", action='store_true', default=False)
    parser.add_argument("--group-svd", action='store_true', default=False)
    parser.add_argument("--scaled", action='store_true', default=False)

    args = parser.parse_args()

    components, labels, subjs = load_data(args.source, args.data, args.exclude_ids)
    if not args.debug:
        if args.group_svd:
            components = group_svd(components, scaled=args.scaled)
        else:
            components, labels, subjs = iterate_align(components, labels, subjs, args.thresh, args.max_iter, args.norm, args.mean_align, args.debug, args.fnorm)
        save_data(args.save, args.data, components, labels, subjs)
    else:
        objs, _, labels, subjs = iterate_align(components, labels, subjs, args.thresh, args.max_iter, args.norm, args.mean_align, args.debug)
        dmat = np.vstack((labels[:,0], labels[:,1], subjs, objs))
        # np.savetxt(Path(args.save) / f'./{args.data}_alignment_objs_mean-align={args.mean_align}.csv', dmat, delimiter=',', fmt="%s")
