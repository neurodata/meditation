import numpy as np
from collections import defaultdict

def procrustes(source, target, return_transform=False):
    u, s, v = np.linalg.svd(target.T.dot(source), full_matrices=False)
    xfm = v.T.dot(u.T)
    if return_transform:
        return source.dot(xfm), xfm
    else:
        return source.dot(xfm)

def align(sources, target, firstpass=False, aux_mats=None, debug=False):
    """
    Aligns source matrices to a target using SVD. Assuming centered source matrix P and target matrix Q,
    we wish to find matrix R to minimize \sum_{i=1}^n ||Q - RP||^2 which ammounts to maximize the quantity
    trace[R(PQ^T)] where PQ^T is the cross-covariance matrix.

    Taking the SVD PQ^T = USV^T, the matrix R=VU^T is the optimal one. We iterate this to a stopping criterion.

    sources : list of source matrices to align
    target : target matrix to align to
    aux_mats : list of lists matched to sources
    """
    realign = []
    if aux_mats is not None:
        realign_aux = []
    if firstpass:
        realign.append(target)
        if aux_mats is not None:
            assert len(aux_mats) == len(sources) + 1
            realign_aux.append(aux_mats[0])
            aux_mats = aux_mats[1:]
    elif aux_mats is not None:
        assert len(aux_mats) == len(sources)
    for i, source in enumerate(sources):
        u, s, v = np.linalg.svd(target.T.dot(source), full_matrices=False)
        xfm = v.T.dot(u.T)
        realign.append(source.dot(xfm))
        if aux_mats is not None:
            realign_aux.append([aux_mat.dot(xfm) for aux_mat in aux_mats[i]])
    if aux_mats is not None:
        return realign, realign_aux
    else:
        return realign

def iterate_align(components, labels, subjs, thresh=0.001, max_iter=10, norm=False, mean_align=False, reference_align=True, debug=False, fnorm=False):
    if norm:
        embeddings = components /  np.linalg.norm(components, axis=1, keepdims=True)
    elif fnorm:
        embeddings = components / np.linalg.norm(components, axis=(1, 2), keepdims=True)
    else:
        embeddings = components

    if mean_align:
        assert subjs is not None
        objs = []
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
            embs, obj = _iterate_align(embs, thresh, max_iter, reference_align=reference_align)
            objs.append(obj)
            subj_mats.append(embs)
            subj_means.append(np.mean(embs, axis=0))
        objs = np.hstack(objs)

        aligned_means, embeddings, obj = _iterate_align(subj_means, thresh, max_iter, aux_mats=subj_mats, reference_align=reference_align, debug=debug)
        objs = np.vstack((objs, obj))
        embeddings = np.vstack(embeddings)
        labels = np.vstack(subj_labels)
        subjs = np.hstack([[id]*3 for id in subj_ids])
    else:
        embeddings, objs = _iterate_align(embeddings, thresh, max_iter, reference_align=reference_align, debug=debug)

    if debug:
        return embeddings, labels, subjs, objs

    return embeddings, labels, subjs

def _iterate_align(embeddings, thresh, max_iter, aux_mats=None, reference_align=True, debug=False):
    objs = []
    n = embeddings[0].shape[1]
    if reference_align and aux_mats is not None:
        embeddings, aux_mats = align(embeddings[1:], embeddings[0], firstpass=True, aux_mats=aux_mats)
    elif reference_align:
        embeddings = align(embeddings[1:], embeddings[0], firstpass=True)
    prior_embed = None
    for i in range(max_iter):
        mean = np.mean(embeddings, axis=0).squeeze()
        if aux_mats is not None:
            embeddings, aux_mats = align(
                embeddings, np.asarray(mean),
                firstpass = False, aux_mats=aux_mats)
            objs.append([np.linalg.norm(mean - embed) / n for embed in np.vstack(aux_mats)])
        else:
            embeddings = align(
                embeddings, np.asarray(mean),
                firstpass=False, debug=True)
            objs.append([np.linalg.norm(mean - embed) / n for embed in embeddings])
        embeddings = np.asarray(embeddings)
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    if aux_mats is not None:
        return embeddings, aux_mats, objs
    else:
        return embeddings, objs