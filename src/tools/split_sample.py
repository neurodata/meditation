import re
from pathlib import Path
import random
import os

def get_files(path,
              level='(e|n)',
              subject='([0-9]{3})',
              task='(.+?)',
              ftype='csv',
              flag=''):
    files = []
    query = f'^{level}_sub-'
    query += f'{subject}_ses-1_'
    query += f'task-{task}{flag}\.{ftype}'
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))

    return(files)

def _chunks(a, n):
    """
    Yield successive n-sized chunks from a
    """
    for i in range(0,n):
        yield a[i::n]

def split_sample(path, **kwargs):
    files = get_files(path, **kwargs)
    subjects = list(set([f[1][1] for f in files]))
    random.seed(1)
    random.shuffle(subjects)
    tasks = list(set([f[1][2] for f in files]))
    g1_subjects, g2_subjects = list(_chunks(subjects, 2))
    g1_task_groups = list(_chunks(g1_subjects, 3))
    g2_task_groups = list(_chunks(g2_subjects, 3))

    file_dict = {(f[1][1], f[1][2]):f for f in files}

    f1 = []
    f2 = []

    tasks = ['compassion', 'openmonitoring', 'restingstate']

    for group, t1, t2, t3 in zip(g1_task_groups, tasks[2:] + tasks[:2], tasks[1:] + tasks[:1] , tasks):
        f1 += [file_dict[(subj, t1)] for subj in group]
        f1 += [file_dict[(subj, t2)] for subj in group]
        f2 += [file_dict[(subj, t3)] for subj in group]
    
    for group, t1, t2, t3 in zip(g2_task_groups, tasks[2:] + tasks[:2], tasks[1:] + tasks[:1] , tasks):
        f2 += [file_dict[(subj, t1)] for subj in group]
        f2 += [file_dict[(subj, t2)] for subj in group]
        f1 += [file_dict[(subj, t3)] for subj in group]

    return (f1,f2)