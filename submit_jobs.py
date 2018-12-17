import numpy as np
import os
import sys
import subprocess
import random
import argparse


def check_csv_names(csvlist, path):
    nl = [os.path.join(path, n ) + '.csv' for n in csvlist]
    fncheck = [os.path.isfile(n) for n in nl]
    print fncheck
    invalfns = [nl[idx]
                for idx in [i for i, x in enumerate(fncheck) if x == False]]
    assert all(fncheck), 'Invalid csvs {}'.format(invalfns)

if __name__ == '__main__':

    bash_script = '/home/n8307628/fastai/courses/projects/train_hpc.bash'
    path = '/home/n8307628/skin_cancer/'

    gputypes = ['K40', 'M40']
    gpidx = 1

    # training sets and weight names
    jnames = ['cls_seg', 'Mel_seg', 'SK_seg']
    trncsvs = ['train_seg_classes_multi',
               'train_seg_melanoma_multi', 'train_seg_keratosis_multi']

    tstcsvs = ['test_seg_classes_multi',
               'test_seg_melanoma_multi', 'test_seg_keratosis_multi']
    test_folders = ['ISIC/2017_Test_v2_Data_Classification_lesion_seg/']*3
    wnames = ['res_101_class_seg_pret', 'res_101_mel_seg_pret',
              'res_101_sk_seg_pret']


    trncsvs = [ss+'.csv' for ss in trncsvs]
    tstcsvs = ['ISIC/'+ ss + '.csv' for ss in tstcsvs]

    check_csv_names(trncsvs, path)
    check_csv_names(tstcsvs, path)

    for name, traincsv, testscv, wname, tstfolder in zip(jnames, trncsvs, tstcsvs, wnames, test_folders):
        jobname = '%s' % name
        cmd = "qsub -v TRAINCSV='{}',TESTCSV='{}',WEIGHTN='{}',TESTFOLDER='{}' -l gputype={} -N {} {}".format(
            traincsv, testscv, wname, tstfolder, gputypes[gpidx], jobname, bash_script)
        print '-'*20
        print 'running cmd:\n"{}"'.format(cmd)
        subprocess.check_call(cmd, shell=True)
        gpidx ^= 1  # toggle


    '''
    Initial parameters. 
    Train on segmentation of isic + dermo
    train on non seg multi datastets
    train the two binary classifiers.
    '''
    # training sets and weight names
    # jnames = ['Mel_seg', 'SK_seg', 'Mel', 'SK']
    # trncsvs = ['train_Mel_seg_isic17_dermo',
    #            'train_SK_seg_isic17_dermo', 'train_multi_Mel_half', 'train_mutli_SK_half']

    # tstcsvs = ['test_Mel_seg_isic17_dermo',
    #            'test_SK_seg_isic17_dermo', 'test_mel_17', 'test_ker_17']
    # test_folders = ['ISIC/ISIC-2017_Test_v2_Data_lesion_seg/', 'ISIC/ISIC-2017_Test_v2_Data_lesion_seg/',
    #                 'ISIC/ISIC-2017_Test_v2_Data_Classification/',
    #                 'ISIC/ISIC-2017_Test_v2_Data_Classification/']

    # wnames = ['res101_Mel_seg', 'res101_SK_seg',
    #           'res101_Mel_mutli_half', 'res101_SK_mutli_half']
