import numpy as np

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

def iterate_align(components, labels, subjs, thresh=0.001, max_iter=10, norm=False, mean_align=False, debug=False):
    if norm:
        embeddings = components /  np.linalg.norm(components, axis=1, keepdims=True)
    else:
        embeddings = components

    if max_iter <= 0:
        return embeddings

    if mean_align:
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
            embs = _iterate_align(embs, thresh, max_iter)
            subj_mats.append(embs)
            subj_means.append(np.mean(embs, axis=0))

        aligned_means, embeddings = _iterate_align(subj_means, thresh, max_iter, aux_mats=subj_mats, debug=debug)

        embeddings = np.vstack(embeddings)
        labels = np.vstack(subj_labels)
        subjs = np.hstack([[id]*3 for id in subj_ids])
    else:
        embeddings = _iterate_align(embeddings, thresh, max_iter, debug=debug)

    return embeddings, labels, subjs


def _iterate_align(embeddings, thresh, max_iter, aux_mats=None, debug=False):
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
                firstpass=False, debug=True)
        embeddings = np.asarray(embeddings)
        if debug and prior_embed is not None:
            print(np.linalg.norm(embeddings - prior_embed))
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    if aux_mats is not None:
        return embeddings, aux_mats
    else:
        return embeddings