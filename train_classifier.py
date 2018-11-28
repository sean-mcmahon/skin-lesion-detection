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

    arch = resnet101
    im_size = 256
    bs = 64

    train_csv = PATH / 'train_multi_half_ia_nervi.csv'
    test_csv = PATH / 'ISIC/test_all_17.csv'
    test_path = PATH / 'ISIC/ISIC-2017_Test_v2_Data_Classification/'
    assert all([train_csv.exists(), test_csv.exists(), test_path.is_dir()]),  [
        train_csv.exists(), test_csv.exists(), test_path.is_dir()]

    # val_idx should be the last 150 images from the train csv
    train_df = pd.read_csv(train_csv)
    trlen = len(train_df)
    val_idx = list(range(trlen - 150, trlen))

    weight_name = 'resnet101_all_no_ia_nervi'

    trainer = ClassifierTrainer(PATH, arch, im_size, bs, train_csv,
                                 sn=weight_name, test_csv=test_csv, test_folder=test_path,
                                 val_idx=val_idx)
                            
    trainer.set_lr(1e-2)
    print('training.')
    trainer.init_fit(weight_name + '_1')
    trainer.test_val(tta=False, sf=False)
    trainer.inter_fit(weight_name + '_2')
    trainer.test_eval(tta=False, sf=False)
    trainer.final_fit(weight_name + '_3')
    trainer.test_val(sf=False)
    trainer.test_eval(sf=False)
