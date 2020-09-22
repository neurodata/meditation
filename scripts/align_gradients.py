import numpy as np
import argparse
from pathlib import Path
import os
import re
import h5py
import sys; sys.path.append('../')
from src.tools import align
from src.tools import get_files, get_latents

def iterate_align(groups, thresh=0.001, max_iter=100):
    embeddings = groups
    embeddings = align(embeddings[1:], embeddings[0], firstpass=True)

    prior_embed = None
    for i in range(max_iter):
        embeddings = align(embeddings, np.asarray(np.mean(embeddings, axis=0).squeeze()), firstpass = False)
        embeddings = np.asarray(embeddings)
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    return embeddings

def load_data(source_dir, ftype, exclude_ids):
    source_dir = Path(source_dir)
    flag = "_emb"
    SOURCE = 'dmap'
    return get_latents(
        source_dir, flag=flag, ids=True, ftype=ftype, source=SOURCE, subjects_exclude=exclude_ids, as_groups=False
    )

def save_data(target_dir, groups, labels, subjs):
    target_dir = Path(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for group, (trait, state), subj in zip(groups, labels, subjs):
        np.save(target_dir / f'{trait}_sub-{subj}_ses-1_task-{state}_dmap', group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source directory with files", type=str, required=True)
    parser.add_argument("-t", "--save", help="target directory to save files", type=str, required=True)
    parser.add_argument("--ftype", help="filetype ending to analyze", type=str, default='npy')
    parser.add_argument("-x", "--exclude-ids", help="list of subject IDs", nargs='*', type=str)
    parser.add_argument("--thresh", help="threshold to stop iterative align", type=float, default=0.001)
    parser.add_argument("--max-iter", help="max number of iterations for the alignment", type=int, default=100)

    args = parser.parse_args()

    groups, labels, subjs = load_data(args.source, args.ftype, args.exclude_ids)
    groups = iterate_align(groups, args.thresh, args.max_iter)
    save_data(args.save, groups, labels, subjs)
