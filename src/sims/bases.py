import numpy as np
from scipy.stats import ortho_group

def ortho_bases(n_subjs, n, d, n_bases, n_shared_bases=0, per_subj=2):
    """
    Multilevel mixture of bases, shared between all views and the rest per subj

    n_bases : int
        Total number of basis vectors per view
    n_shared_bases : int, default 0
        Number of basis vectors shared across all views
    """
    labels = np.hstack([[i]*2 for i in range(n_subjs)])

    basis = ortho_group.rvs(n)
    n_indiv_bases = n_bases - n_shared_bases

    shared_idx = np.arange(n_shared_bases)
    indiv_idx = np.arange(n_indiv_bases)
    Xs = np.concatenate([
        [basis[:,np.hstack((shared_idx, i*n_indiv_bases + n_shared_bases + indiv_idx))] @ ortho_group.rvs(n_bases)[:, :d]
        for _ in range(per_subj)] for i in range(n_subjs)
    ])

    return Xs, labels


def two_samp_bases(n_subjs=20, n=100, d_shared=2, d_indiv=2, n_shared_bases=3, n_indiv_bases=3, epsilon=0, per_subj=2):
    """
    Samples pairs of samples from common bases. Two sets of samples,
    where epsilon in [0,1] interpolates between two sets of orthogonal bases
    """
    basis = ortho_group.rvs(n)
    shared_idx = np.arange(2*n_shared_bases)
    indiv_idx = np.arange(n_indiv_bases) + len(shared_idx)
    assert d_shared <= n_shared_bases
    assert d_indiv <= n_indiv_bases

    Xs1 = np.concatenate([
        [np.hstack([
            basis[:, shared_idx] @ _2samp_rot(n_shared_bases, 0)[:, :d_shared],
            basis[:, indiv_idx + i*n_indiv_bases] @ ortho_group.rvs(n_indiv_bases)[:, :d_indiv]
        ])
        for _ in range(per_subj)] for i in range(n_subjs)
    ])

    offset = n_subjs * n_indiv_bases
    Xs2 = np.concatenate([
        [np.hstack([
            basis[:, shared_idx] @ _2samp_rot(n_shared_bases, epsilon)[:, :d_shared],
            basis[:, indiv_idx + i*n_indiv_bases + offset] @ ortho_group.rvs(n_indiv_bases)[:, :d_indiv]
        ])
        for _ in range(per_subj)] for i in range(n_subjs)
    ])

    Xs = np.vstack((Xs1, Xs2))
    y = np.hstack((np.zeros(len(Xs1)), np.ones(len(Xs2))))
    labels = np.hstack([[i]*2 for i in range(2*n_subjs)])

    return Xs, y, labels

def _2samp_rot(d, epsilon):
    rot1 = ortho_group.rvs(d) * np.sqrt(1-epsilon)
    rot2 = ortho_group.rvs(d) * np.sqrt(epsilon)
    rot = np.r_[rot1, rot2]

    return rot
