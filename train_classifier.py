from fastai.conv_learner import *
import pandas as pd
import numpy as np
import pathlib
import fastai.plots as fp
import sklearn.metrics as metrics
import classifier as cl
from classifier import ClassifierTrainer


if __name__ == "__main__":
    ldir = Path('/home/sean/hpc-home/')
    hdir = ldir if ldir.exists() else Path('/home/n8307628')
    PATH = hdir / 'skin_cancer/'
    arch = resnet101
    im_size = 224
    bs = 64

    train_csv = PATH / 'val_isic17.csv'
    test_csv = PATH / 'test_all_17.csv'
    test_path = PATH / 'ISIC/ISIC-2017_Test_v2_Data_Classification/'
    assert all([train_csv.exists(), test_csv.exists(), test_path.is_dir()])

    # val_idx should be the last 150 images from the train csv
    train_df = pd.read_csv(train_csv)
    trlen = len(train_df)
    val_idx = None #list(range(trlen - 150, trlen))

    weight_name = 'resnet101_all'

    trainer = ClassifierTrainer(PATH, arch, im_size, bs, train_csv, 
                                sn=weight_name, test_csv=test_csv, test_folder=test_path)
    trainer.set_lr(5e-2)
    print('training.')
    trainer.init_fit(weight_name + '_1')
    trainer.test_val()
