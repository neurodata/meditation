import numpy as np
import argparse
from pathlib import Path
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True, style='white', context='talk', font_scale=1)
PALETTE = sns.color_palette("Set1")

name_dict = {
    'Gradients 0':'Gradient 1',
    'Gradients 1':'Gradient 2',
    'Gradients 2':'Gradient 3',
    'Gradients (0, 1)':'Gradients 1,2',
    'Gradients (1, 2)':'Gradients 2,3',
    'Gradients (2, 0)':'Gradients 1,3',
    'Gradients (0, 1, 2)':'Gradients 1,2,3',
    'Gradients 0':'Gradient 1',
    'Gradients 1':'Gradient 2',
    'Gradients 2':'Gradient 3',
    'Gradients (0, 1)':'Gradients 1,2',
    'Gradients (1, 2)':'Gradients 2,3',
    'Gradients (2, 0)':'Gradients 1,3',
    'Gradients (0, 1, 2)':'Gradients 1,2,3',
    'Experts Resting vs. Experts Compassion':['EXP','res', 'EXP','com'],
    'Experts Resting vs. Experts Open Monitoring':['EXP','res', 'EXP','o m'],
    'Experts Open Monitoring vs. Experts Compassion':['EXP','o m', 'EXP','com'],
    'Experts Resting vs. Experts Meditating':['EXP','res', 'EXP','med'],
    'Novices Resting vs. Novices Compassion':['NOV','res', 'NOV','com'],
    'Novices Resting vs. Novices Open Monitoring':['NOV','res', 'NOV','o m'],
    'Novices Open Monitoring vs. Novices Compassion':['NOV','o m', 'NOV','com'],
    'Novices Resting vs. Novices Meditating':['NOV','res', 'NOV','med'],
    'Experts Resting vs. Novices Resting':['EXP','res', 'NOV','res'],
    'Experts Compassion vs. Novices Compassion':['EXP','com', 'NOV','com'],
    'Experts Open Monitoring vs. Novices Open Monitoring':['EXP','o m', 'NOV','o m'],
    'Experts Meditating vs. Novices Meditating':['EXP','med', 'NOV','med'],
    'Experts All vs. Novices All':['EXP','all', 'NOV','all'],
    'Experts Resting vs. Novices Compassion':['EXP','res', 'NOV','com'],
    'Experts Resting vs. Novices Open Monitoring':['EXP','res', 'NOV','o m'],
    'Experts Compassion vs. Novices Resting':['EXP','com', 'NOV','res'],
    'Experts Compassion vs. Novices Open Monitoring':['EXP','com', 'NOV','o m'],
    'Experts Open Monitoring vs. Novices Resting':['EXP','o m', 'NOV','res'],
    'Experts Open Monitoring vs. Novices Compassion':['EXP','o m', 'NOV','com'],
    'Resting vs. Compassion':['ALL','res', 'ALL','com'],
    'Resting vs. Open Monitoring':['ALL','res', 'ALL','o m'],
    'Compassion vs. Open Monitoring':['ALL','com', 'ALL','o m'],
    'Resting vs. Meditating':['ALL','res', 'ALL','med']
}

label_dict = {
    'EXP':'EXP',
    'NOV':'NOV',
    'ALL':'ALL',
    'o m': 'open',
    'med':'med ',
    'res':'rest',
    'com':'comp',
    'all':'all '
}

def make_heatmap(source_dir, save_path):
    # The 2nd file has more permutations, but no ksample runs
    # if tag is not None:
    #     tag = f'_{tag}'
    # else:
    #     tag = ''
    # pattern = f'DCORR_{method}_2-sample{tag}_pvalues_' + r'{}.csv'
    # files = glob.glob('../data/' + '2sample_tests/' + pattern.format('*'))
    # num = sorted(set([int(re.split(r'.*_(\d+)\.csv', f)[1]) for f in files]))[-1]
    files = glob.glob(str(Path(source_dir) / '2-*.csv'))
    pvalues = pd.read_csv(
        files[0],
        # Path('../data/') / '2sample_tests' / pattern.format(num),
        # Path('../data/') / '2sample_tests' / '073_unexcluded' / 'DCORR_gcca_restricted_perm_pvalues_100000_min_rank-ZG3.csv',
        index_col=0
    )
    # pvalues = pvalues.drop([
    #     'Experts Resting vs. Novices Meditating', 'Experts Meditating vs. Novices Resting'
    # ])

    pvalues.columns = [name_dict[v] for v in pvalues.columns]
    fmt_index = ["{:^3s}|{:<3s} - {:^3s}|{:<3s}".format(*[label_dict[vv] for vv in name_dict[v]]) for v in pvalues.index]
    pvalues.index = fmt_index

    # Add pvalues from k-sample tests
    # k_sample_paths = [
    #     f'DCORR_{method}_6-sample{tag}_pvalues_',
    #     f'DCORR_{method}_3-sample-experts{tag}_pvalues_',
    #     f'DCORR_{method}_3-sample-novices{tag}_pvalues_',
    # ]
    # k_sample_paths = [s + r'{}.csv' for s in k_sample_paths]
    # files = [glob.glob('../data/' + '/ksample_tests/' + pattern.format('*')) for pattern in k_sample_paths]
    # nums = [sorted(set([int(re.split(r'.*_(\d+)\.csv', f)[1]) for f in fs]))[-1] for fs in files]
    # kpvals = np.vstack([
    #     pd.read_csv(Path('../data/') / 'ksample_tests' / path.format(n), index_col=0).values 
    #     for n, path in zip(nums, k_sample_paths)
    # ])
    k_sample_paths = ['6-*.csv', '3E-*.csv', '3N-*.csv']
    files = [glob.glob(str(Path(source_dir) / path))[0] for path in k_sample_paths]
    kpvals = np.vstack([pd.read_csv(f, index_col=0).values for f in files])

    # Scale
    kpvals = np.asarray(kpvals) * 7
    kpvals[1:,:] = kpvals[1:,:] * 2
    df = pd.DataFrame(kpvals, columns = pvalues.columns)
    df.index = [
        '6-sample All',
        '3-sample EXP States',
        '3-sample NOV States'
    ]
    df[df > 1] = 1


    pvalues = pd.concat([df, pvalues])
    d = pvalues.values
    d[3:,:] *= np.multiply(*d[3:,:].shape)
    d[d > 1] = 1

    i_new = np.hstack((pvalues.index[:3], [pvalues.index[15]], pvalues.index[3:15], pvalues.index[16:]))
    d_new = np.vstack((d[:3], d[15], d[3:15], d[16:]))

    pvalues = pd.DataFrame(data=d_new, columns=pvalues.columns)
    pvalues.index = i_new

    mask = pvalues.copy()
    alpha = 0.05
    mask[:] = np.select([
        mask < alpha, mask >= alpha],
        ['X', ''],
        default=mask
    )

    f, ax = plt.subplots(1, figsize=(14,9))
    ax = sns.heatmap(
        pvalues.transform('log10'),
        ax=ax,
        annot=mask,#sig_labels,#True,
        #cmap='coolwarm_r',
        #center=np.log10(0.05),
        fmt='',
        square=False,
        linewidths=.5,
        #vmin=0,
        #vmax=0.1,
        #norm=log_norm,
        cbar_kws={"ticks": np.log10([0.01, 0.05,0.1,1])},
        #yticklabels = fmt_index
    )
    ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
    ax.collections[0].colorbar.set_ticklabels([0.01, 0.05,0.1,1])

    # x labels
    loc, xlabels = plt.xticks()
    ax.set_xticklabels(xlabels, rotation=40, ha='right')

    # y labels
    # plt.draw()  # this is needed because get_window_extent needs a renderer to work
    # yax = ax.get_yaxis()
    # # find the maximum width of the label on the major ticks
    # pad = max(T.label.get_window_extent().width for T in yax.majorTicks)

    # yax.set_tick_params(pad=pad)
    # plt.draw()

    ax.set_yticklabels(ax.get_yticklabels(), ha='right', fontdict={'family' : 'monospace'})

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # python3 align_gradients.py --source 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--tag", help="source directory with files", type=str, default=None)
    parser.add_argument("--source", help="source directory with files", type=str, default=None)
    parser.add_argument("-t", "--save", help="target directory to save files", type=str, required=True)
    # parser.add_argument("--method", help="method used", type=str, required=True)

    args = parser.parse_args()
    # make_heatmap(args.method, args.tag, args.save)
    make_heatmap(args.source, args.save)

    
