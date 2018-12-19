from segmenter import *
from classifier import *
from train_classifier import create_trainer
import numpy as np

def load_segmentation():
    base_arch = resnet34
    sz = 128
    _, vtfm = tfms_from_model(
        base_arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO)
    # denorm = vtfm.denorm
    net = build_unet(base_arch)
    weights = '/home/sean/src/docker_fastai/128unet_dermofit_isic17_1.h5'
    if not os.path.isfile(weights):
        weights = '/app/128unet_dermofit_isic17_1.h5'
    if not os.path.isfile(weights):
        raise FileNotFoundError(f'Invalid: {weights}')
    load_model(net, weights)  # loads weights to "net" - fastai function
    return net, vtfm


def load_classifier(weights, trn_csv, sz=200, path='/home/sean/hpc-home/skin_cancer/', arch=resnet101, bs=1, workers=1):
    path = Path(path)  # the pathway to more paths
    if not path.exists(): raise FileNotFoundError(f'Invalid path "{path}"')
    p_dict = {'path': path, 'arch': arch, 'sz': sz,
            'bs': bs, 'trn_csv': path / trn_csv, 'sn': 'Unused: can be anything here',
            'test_csv': None, 'test_folder': None, 'val_idx': None,
            'precom': False, 'num_workers': workers, 'lr': 1e-2, 'aug_tfms': transforms_top_down,
            'params_fn': 'dont matter', 'precom': False}

    _, v_class_tfms = tfms_from_model(
        p_dict['arch'], p_dict['sz'], aug_tfms=p_dict['aug_tfms'])
    trainer = create_trainer(p_dict)
    trainer.load(weights)

    return trainer.learn.model, v_class_tfms


def load_bin_classifiers():
    '''
    There's more bloat around classification in fastai. 
    Didn't have time to figure out how to condense it all.
    '''
    # Load Classifier
    PATH = Path('/home/sean/src/docker_cloudvis')
    if not PATH.exists():
        PATH = Path('/app/')
    arch = resnet101
    im_size = 128
    bs = 1
    num_workers = 1
    test_folder = None  # 'ISIC/ISIC-2017_Test_v2_Data_Classification/'
    train_csv = PATH / 'train_multi_Mel_half.csv'
    test_csv = None
    # test_path = PATH / test_folder
    # val_idx should be the last 150 images from the train csv
    val_idx = None

    params_file_name = 'blah'

    weight_name = "res101_Mel_mutli_half_2018-12-11"

    p_dict = {'path': PATH, 'arch': arch, 'sz': im_size,
              'bs': bs, 'trn_csv': train_csv, 'sn': weight_name,
              'test_csv': test_csv, 'test_folder': test_folder, 'val_idx': val_idx,
              'precom': False, 'num_workers': num_workers, 'lr': 1e-2, 'aug_tfms': transforms_top_down,
              'params_fn': params_file_name, 'precom': False}

    _, v_class_tfms = tfms_from_model(
        arch, im_size, aug_tfms=p_dict['aug_tfms'])
    # class_denorm = v_class_tfms.denorm

    mel_trainer = create_trainer(p_dict)
    mel_trainer.load('res_101_mel_multi_seg_pret_2018-12-18_2')
    sk_trainer = create_trainer(p_dict)
    sk_trainer.load('res_101_sk_multi_seg_pret_2018-12-17_2')
    return sk_trainer.learn.model, mel_trainer.learn.model, v_class_tfms


def run_model(m, im, prepIm=None):
    if prepIm:
        im = prepIm(im)
    # pytorch expects a batch dimension
    im = np.expand_dims(im, 0) if im.ndim == 3 else im
    m.eval()
    if hasattr(m, 'reset'):
        m.reset()
    p = to_np(m(VV(im))).squeeze()
    return p, im


def denorm_img(im, denorm_func, reverse_ch=True):
    # Convert from (bs, c, w, h) to (bs, w, h, c)
    imd = np.rollaxis(im, 1, 4)
    # denormalise, based on how you preprocess images for your model.
    # Takes a np array and returns one.
    imd = denorm_func(imd)
    # Batch size (bs) should be 1, remove that for plotting.
    # Also, values need to be between 0-255. Type change for PIL
    imd = (imd.squeeze() * 255).astype(np.uint8)
    # For encoding back to jpg byte array, BGR is expected.
    # So reverse the channel dimension
    if reverse_ch:
        imd = imd[:, :, ::-1]
    return imd