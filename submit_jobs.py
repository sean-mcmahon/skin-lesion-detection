import numpy as np
import os
import sys
import subprocess
import random
import argparse


def fill_csv_names(csvlist, path):
    nl = [os.path.join(path, n ) + '.csv' for n in csvlist]  
    fncheck = [os.path.isfile(n) for n in nl]
    invalfns = [nl[idx]
                for idx in [i for i, x in enumerate(fncheck) if x == False]]
    assert all(fncheck), 'Invalid csvs {}'.format()
    return nl

if __name__ == '__main__':
    # Lrs = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # loss_ws = [1.0]  # , 0.5, 0.1, 0.05]
    # random.shuffle(Lrs)
    # random.shuffle(loss_ws)

    bash_script = '/home/n8307628/fastai/courses/projects/train_hpc.bash'
    path = '/home/n8307628/skin_cancer/'

    gputypes = ['K40', 'M40']
    gpidx = 1

    # run crossval
    jnames = ['Mel_seg', 'SK_seg', 'Mel', 'SK']
    trncsvs = ['train_Mel_seg_isic17_dermo',
               'train_SK_seg_isic17_dermo', 'train_multi_Mel_half', 'train_mutli_SK_half']

    tstcsvs = ['test_Mel_seg_isic17_dermo',
               'test_SK_seg_isic17_dermo', 'test_mel_17', 'test_ker_17']
    wnames = ['res101_Mel_seg', 'res101_SK_seg',
              'res101_Mel_mutli_half', 'res101_SK_mutli_half']

    trncsvs = fill_csv_names(trncsvs, path)
    tstcsvs = fill_csv_names(tstcsvs, path)

    # splits = ['_2_4', '_3_4', '_4_4']
    for name, traincsv, testscv, wname in zip(jnames, trncsvs, tstcsvs, wnames):
        jobname = '%s' % name
        cmd = "qsub -v TRAINCSV='{}',TESTCSV='{}',WEIGHTN='{}' -l gputype={} -N {} {}".format(
            traincsv, testscv, wname, gputypes[gpidx], jobname, bash_script)
        print -'*20'
        print 'running cmd:\n"{}"'.format(cmd)
        # subprocess.check_call(cmd, shell=True)
        gpidx ^= 1  # toggle
