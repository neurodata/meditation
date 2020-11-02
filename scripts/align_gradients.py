import numpy as np
import argparse
from pathlib import Path
import os
import re
import h5py
import sys; sys.path.append('../')
from src.tools import align
from src.tools import get_files, get_latents
from collections import defaultdict


def iterate_align(components, labels, subjs, thresh=0.001, max_iter=100, norm=False, group_align=False):
    if norm:
        embeddings = components /  np.linalg.norm(components, axis=1, keepdims=True)
    else:
        embeddings = components

    if max_iter <= 0:
        return embeddings

    if group_align:
        assert subjs is not None
        subj_embeddings = defaultdict(list)
        for embedding, subj, l in zip(embeddings, subjs, labels):
            subj_embeddings[subj].append((embedding, l))

        subj_ids = []
        subj_means = []
        subj_mats = []
        subj_labels = []
        for subj, embs_ls in subj_embeddings.items():
            embs, ls = list(zip(*embs_ls))
            subj_ids.append(subj)
            subj_labels.append(ls)
            subj_mats.append(embs)
            subj_means.append(np.mean(_iterate_align(embs, thresh, max_iter), axis=0))

        aligned_means, embeddings = _iterate_align(subj_means, thresh, max_iter, aux_mats=subj_mats)
        embeddings = np.vstack(embeddings)
        labels = np.vstack(subj_labels)
        subjs = np.hstack([[id]*3 for id in subj_ids])
    else:
        embeddings = _iterate_align(embeddings, thresh, max_iter)

    return embeddings, labels, subjs


def _iterate_align(embeddings, thresh, max_iter, aux_mats=None):
    if aux_mats is not None:
        embeddings, aux_mats = align(embeddings[1:], embeddings[0], firstpass=True, aux_mats=aux_mats)
    else:
        embeddings = align(embeddings[1:], embeddings[0], firstpass=True)
    prior_embed = None
    for i in range(max_iter):
        if aux_mats is not None:
            embeddings, aux_mats = align(
                embeddings, np.asarray(np.mean(embeddings, axis=0).squeeze()),
                firstpass = False, aux_mats=aux_mats)
        else:
            embeddings = align(
                embeddings, np.asarray(np.mean(embeddings, axis=0).squeeze()),
                firstpass = False)
        embeddings = np.asarray(embeddings)
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    if aux_mats is not None:
        return embeddings, aux_mats
    else:
        return embeddings

def load_data(source_dir, data, exclude_ids):
    source_dir = Path(source_dir)
    if data == 'dmap':
        flag = "_emb"
        SOURCE = 'dmap_raw'
        ftype = 'npy'
    elif data == 'gcca':
        flag = "_gcca"
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
        if data == 'dmap':
            np.save(target_dir / f'{trait}_sub-{subj}_ses-1_task-{state}_{data}.npy', group)
        elif data == 'gcca':
            h5f = h5py.File(target_dir / f'{trait}_sub-{subj}_ses-1_task-{state}_{data}.h5', 'w')
            h5f.create_dataset('latent', data=group)
            h5f.close()


if __name__ == "__main__":
    # python3 align_gradients.py --source 
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source directory with files", type=str, required=True)
    parser.add_argument("-t", "--save", help="target directory to save files", type=str, required=True)
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str)
    parser.add_argument("--thresh", help="threshold to stop iterative align", type=float, default=0.001)
    parser.add_argument("--max-iter", help="max number of iterations for the alignment", type=int, default=100)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("-d", "--data", help="list servers, storage, or both (default: %(default)s)", choices=['gcca', 'dmap'], default="dmap")
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--group-align", action='store_true', default=False)

    args = parser.parse_args()

    components, labels, subjs = load_data(args.source, args.data, args.exclude_ids)
    components, labels, subjs = iterate_align(components, labels, subjs, args.thresh, args.max_iter, args.norm, args.group_align)
    if not args.debug:
        save_data(args.save, args.data, components, labels, subjs)
