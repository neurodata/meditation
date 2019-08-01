#-*- coding: utf-8 -*-

__author__ = 'Ronan Perry'


import numpy as np
from scipy import linalg,stats
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

from tqdm import tqdm


def _preprocess(x):
    # Mean along rows using sample mean and sample std
    x2 = stats.zscore(x,axis=1,ddof=1) 
    # Mean along columns
    mu = np.mean(x2,axis=0)
    x2 -= mu
    return(x2)

def gcca(data, rank_tolerance=None, n_components=None):
    n = data[0].shape[0]
    
    data = [_preprocess(x) for x in data]
    
    Uall = []
    Sall = []
    Vall = []
    ranks = []
    
    for x in tqdm(data):
        # Preprocess
        x[np.isnan(x)] = 0
        
        temp['X'] = x

        # compute the SVD of the data
        u,s,vt = linalg.svd(x, full_matrices=False)

        Sall.append(s)
        Vall.append(vt.T)
        # Dimensions to reduce to
        if rank_tolerance:
            rank = sum(s > rank_tolerance)
        else:
            rank = n_components
            
        ranks.append(rank)
        u = u[:,:rank]
        Uall.append(u)
                
    d = min(ranks)

    # Create a concatenated view of Us
    Uall_c = np.concatenate(Uall,axis=1)

    _,_,VV=svds(Uall_c,d)
    VV = np.flip(VV.T,axis=1)
    VV = VV[:,:min([d,VV.shape[1]])]

    # SVDS the concatenated Us
    idx_end = 0
    projX = []
    for i in range(len(data)):
        idx_start = idx_end
        idx_end = idx_start + ranks[i]
        VVi = normalize(VV[idx_start:idx_end,:],'l2',axis=0)
        # Compute the canonical projections
        A = np.sqrt(n-1) * Vall[i][:,:ranks[i]]
        A = A @ linalg.solve(np.diag(Sall[i][:ranks[i]]), VVi)
        projX.append(data[i] @ A)
                
    return(projX)