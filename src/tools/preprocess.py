import numpy as np

def align(sources, target, firstpass = False, aux_mats = None):
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

def iterate_align(Xs, thresh=0.001, max_iter=100, norm=True, labels=None):
    if norm:
        embeddings = np.asarray(Xs) / np.linalg.norm(Xs, axis=1, keepdims=True)
    else:
        embeddings = np.asarray(Xs)

    embeddings = align(embeddings[1:], embeddings[0], firstpass=True)

    prior_embed = None
    for i in range(max_iter):
        embeddings = align(embeddings, np.asarray(np.mean(embeddings, axis=0).squeeze()), firstpass = False)
        embeddings = np.asarray(embeddings)
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    return embeddings

def _iterate_align(Xs, thresh=0.001, max_iter=100):
    if norm:
        embeddings = np.asarray(Xs) / np.linalg.norm(Xs, axis=1, keepdims=True)
    else:
        embeddings = np.asarray(Xs)

    embeddings = align(embeddings[1:], embeddings[0], firstpass=True)

    prior_embed = None
    for i in range(max_iter):
        embeddings = align(embeddings, np.asarray(np.mean(embeddings, axis=0).squeeze()), firstpass = False)
        embeddings = np.asarray(embeddings)
        if prior_embed is not None and np.linalg.norm(embeddings - prior_embed) < thresh:
            break
        prior_embed = embeddings

    return embeddings