from fastai.conv_learner import *
import pandas as pd
import numpy as np
import pathlib
import fastai.plots as fp
import sklearn.metrics as metrics
import classifier as cl
from classifier import ClassifierTrainer
import json
import datetime
import argparse


def create_trainer(params_dict):
    # I want a consistant class initialisation
    return ClassifierTrainer(**params_dict)    


def save_params(fn, params):
    p_strs = {}
    for key, value in params.items():
        if isinstance(value, str):
            p_strs[key] = value
        elif isinstance(value, Path):
            p_strs[key] = str(value)
        else:
            p_strs[key] = value.__str__()

    if not str(fn).endswith('.json'): fn += '.json'
    with open(fn, 'w') as fp:
        json.dump(p_strs, fp, indent=2, sort_keys=True)
    print('Params saved as {}'.format(fn))

def get_val_idx(train_csv):
    # val_idx should be the last 150 images from the train csv
    train_df = pd.read_csv(train_csv)
    trlen = len(train_df)
    return range(trlen - 150, trlen)


def weights_params_paths(weight_name):
    # Parameter filename
    dt_str = datetime.datetime.now().strftime('_%Y-%m-%d__%H-%M_')
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'param_files')
    params_file_name = os.path.join(save_dir, weight_name + dt_str +'.json')
    # Weights filename
    weight_name += datetime.datetime.now().strftime('_%Y-%m-%d')
    # Create dir if needed
    if not os.path.isdir(os.path.dirname(params_file_name)):
        os.mkdir(os.path.dirname(params_file_name))
    return weight_name, params_file_name


def main():
    ldir = Path('/home/sean/hpc-home/')
    hdir = ldir if ldir.exists() else Path('/home/n8307628')
    PATH = hdir / 'skin_cancer/'

    arch = resnet101
    im_size = args.im_size
    bs = args.batchsize
    num_workers = args.num_workers

    train_csv = PATH / args.train_csv
    test_csv = PATH / args.test_csv
    test_folder = 'ISIC/ISIC-2017_Test_v2_Data_Classification/'
    test_path = PATH / test_folder
    weight_name = args.weight_name


    assert all([train_csv.exists(), test_csv.exists(), test_path.is_dir()]),  [
        train_csv.exists(), test_csv.exists(), test_path.is_dir()]

    # val_idx should be the last 150 images from the train csv
    val_idx = get_val_idx(train_csv)

    weight_name, params_file_name = weights_params_paths(weight_name)

    params_dict = {'path': PATH, 'arch': arch, 'sz': im_size,
                   'bs': bs, 'trn_csv': train_csv, 'sn': weight_name,
                   'test_csv': test_csv, 'test_folder': test_folder, 'val_idx': val_idx,
                   'precom': False, 'num_workers': num_workers, 'lr': 1e-2, 
                   'aug_tfms': transforms_top_down, 'params_fn': params_file_name}
    
    # Train model with params
    save_params(params_file_name, params_dict)
    trainer = create_trainer(params_dict)
    trainer.check_test_names()
    print('training.')
    trainer.train(lr=params_dict['lr'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--weight_name', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--im_size', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=64)

    args = parser.parse_args()
    main()


# class ClassifierTrainer():

#     def __init__(self, path, arch, sz, bs, trn_csv, aug_tfms=transforms_top_down,
#                  train_folder='', test_folder=None, val_idx=None, test_csv=None,
#                  lr=1e-2, sn=None, num_workers=8, precom=True):

