#-*- coding: utf-8 -*-

__author__ = 'Ronan Perry'


import numpy as np
from scipy import linalg,stats
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

import tqdm


def _preprocess(x):
    x2 = stats.zscore(x,axis=1)
    x2 -= np.mean(x2,axis=0)
    return x2

def gcca(data, rank_tolerance=None, n_components=None):
    n = data[0].shape[0]

    Uall = []
    Sall = []
    Vall = []
    ranks = []
    
    for x in tqdm(data):
        # Preprocess
        x = _preprocess(x)
        x[np.isnan(x)] = 0

        # compute the SVD of the data
        v,s,ut = linalg.svd(x.T, full_matrices=False)

        Sall.append(s)
        Vall.append(v.T)
        # Dimensions to reduce to
        if rank_tolerance:
            rank = sum(S > rank_tolerance)
        else:
            rank = n_components
        ranks.append(rank)
        ut = ut.T[:,:rank]
        Uall.append(ut)

    d = min(ranks)

    # Create a concatenated view of Us
    Uall_c = np.concatenate(Uall,axis=1)

    _,_,VV=svds(Uall_c,d)
    VV = VV.T
    VV = VV[:,:min([d,VV.shape[1]])]

    # SVDS the concatenated Us
    idx_end = 0
    projX = []
    for i in range(len(data)):
        idx_start = idx_end
        idx_end = idx_start + ranks[i]
        VVi = normalize(VV[idx_start:idx_end,:],'l2')
        # Compute the canonical projections
        A = np.sqrt(n-1) * Vall[i][:,:rank]
        A = A @ (linalg.solve(np.diag(Sall[i][:rank]), VVi))
        projX.append(data[i] @ A)
        
    return(projX)