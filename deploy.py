import matplotlib.gridspec as gridspec
from segmenter import *
from classifier import *
from train_classifier import create_trainer
import numpy as np

'''
By Sean McMahon - 2018
'''

def load_segmentation(path='/ home/sean/src/docker_fastai/'):
    '''
    Loads a version of U-Net for semantic segmentation. 

    There's less bloat here because segmentation in fastai is more custom than classification, in pytorch as well. 
    
    '''
    # resnet34 is a fastai class for construction the resent34 architecture.
    base_arch = resnet34
    sz = 128
    # another fastai function (in transforms.py). The TfmType handles the transformations of the y variable.
    _, vtfm = tfms_from_model(
        base_arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO)
    # denorm = vtfm.denorm
    # this is my function lifted from one of the fastai tutorials. See segmenter.py
    net = build_unet(base_arch)
    weights = os.path.join(str(path), 'models/128unet_dermofit_isic17_1.h5')
    if not os.path.isfile(weights):
        # load weights in docker image. If running jupyter this does nothing
        weights = '/app/128unet_dermofit_isic17_1.h5'
    if not os.path.isfile(weights):
        raise FileNotFoundError(f'Invalid: {weights}')
    # fastai function, Defined in fastai/old/fastai/torch_imports.py
    load_model(net, weights)  # loads weights to "net" - fastai function
    return net, vtfm


def load_classifier(weights, trn_csv, sz=200, path='/home/sean/hpc-home/skin_cancer/', arch=resnet101, bs=1, workers=1):
    '''
    More general classifier loader than load_bin_classifiers().
    Make sure the path has the train csv and the weights somewhere in it. 

    reset101 is a pytorch model built by fastai. Use this by default

    Returns the pytorch network model and the image transformation function.
    '''
    path = Path(path)  # the pathway to more paths
    if not path.exists(): raise FileNotFoundError(f'Invalid path "{path}"')
    p_dict = {'path': path, 'arch': arch, 'sz': sz,
            'bs': bs, 'trn_csv': path / trn_csv, 'sn': 'Unused: can be anything here',
            'test_csv': None, 'test_folder': None, 'val_idx': None,
            'precom': False, 'num_workers': workers, 'lr': 1e-2, 'aug_tfms': transforms_top_down,
            'params_fn': 'dont matter', 'precom': False}

    # another fastai function (in transforms.py).
    # we only want the validation transformer, no random augmentations.
    _, v_class_tfms = tfms_from_model(
        p_dict['arch'], p_dict['sz'], aug_tfms=p_dict['aug_tfms'])
    # trainer is a wrapper class I wrote around fastai network training.
    trainer = create_trainer(p_dict)
    trainer.load(weights) # wraps around fastai's 'load_model'

    return trainer.learn.model, v_class_tfms


def load_bin_classifiers():
    '''
    Load the two binary classifiers, for melanoma and keratosis.

    There's more bloat around classification in fastai. 
    Didn't have time to figure out how to condense it all.
    '''

    # Network parameters.
    # Docker is used for the cloudvis demo.
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
    val_idx = None

    params_file_name = 'blah'

    weight_name = "res101_Mel_mutli_half_2018-12-11"

    p_dict = {'path': PATH, 'arch': arch, 'sz': im_size,
              'bs': bs, 'trn_csv': train_csv, 'sn': weight_name,
              'test_csv': test_csv, 'test_folder': test_folder, 'val_idx': val_idx,
              'precom': False, 'num_workers': num_workers, 'lr': 1e-2, 'aug_tfms': transforms_top_down,
              'params_fn': params_file_name, 'precom': False}

    # Fastai code
    _, v_class_tfms = tfms_from_model(
        arch, im_size, aug_tfms=p_dict['aug_tfms'])
    # class_denorm = v_class_tfms.denorm

    # my wrapper aroud fastai data loader and model creation
    mel_trainer = create_trainer(p_dict)
    mel_trainer.load('res_101_mel_multi_seg_pret_2018-12-18_2')
    sk_trainer = create_trainer(p_dict)
    sk_trainer.load('res_101_sk_multi_seg_pret_2018-12-17_2')
    return sk_trainer.learn.model, mel_trainer.learn.model, v_class_tfms


def run_model(m, im, prepIm=None):
    '''
    Forward pass of a pytorch model.

    m - A Pytorch model.
    prepIm - Image transoform function. 
    Networks expect images with certain transformations appplied to them, usually resize and image normalisation.
    This can be optional for many networks, so it's just a kwarg.

    Returns the model predictions (np.array) and the input transformed image.
    '''
    if prepIm: 
        im = prepIm(im)
    # pytorch expects a batch dimension, so add one if it's not there.
    im = np.expand_dims(im, 0) if im.ndim == 3 else im
    m.eval()
    if hasattr(m, 'reset'):
        # weird fastai shit
        m.reset()
    # the actual forward pass. 
    # VV reads im as a pytorch variable. m() does the forward pass
    # to_np converts it back to a numpy array.
    # squeeze removes the batch dimension from the prediction (1).
    p = to_np(m(VV(im))).squeeze()
    return p, im


def denorm_img(im, denorm_func, reverse_ch=True):
    # Convert from (bs, c, w, h) to (bs, w, h, c)
    imd = np.rollaxis(im, 1, 4)
    # denormalise, based on how you preprocess images for your model.
    # Takes a np array and returns one.
    imd = denorm_func(imd)
    # Batch size (bs) should be 1, remove that for plotting.
    # Also, values need to be between 0-255. Type change for PIL Image format
    imd = (imd.squeeze() * 255).astype(np.uint8)
    # For encoding back to jpg byte array, BGR is expected.
    # So reverse the channel dimension
    if reverse_ch:
        imd = imd[:, :, ::-1]
    return imd


def segment_and_classify(fnames, classifier, segmenter, class_tfm=None, seg_tfm=None):
    '''
    fnames - either a single file name or iteratble of many. 
    (classifier, class_tfm) - Tuple of the classification Pytorch model and it's tranformation function (can be None)
    (segmenter, seg_tfm) - Tuple of the segmentation Pytorch model and it's tranformation function (can be None)

    Returns a list of classifications and segmentations or images, or just a single instance of both.
    '''
    def run(model, tfm, im_name):
        '''
        A wrapper around run_model, loads the image from a filename and passes it through network

        models - A Pytorch model.
        tfm - Networks expect images with certain transformations appplied to them, usually resize and image normalisation. 
        This can be optional for many networks, so it's just a kwarg.

        Returns the model predictions (np.array) and the input transformed image.
        Returns a list rather than tuple for mutability when operating on the segmentation results.
        '''
        input = open_image(im_name)
        return list(run_model(model, input, prepIm=tfm))

    if not isinstance(fnames, str) and isinstance(fnames, collections.Iterable):
        classifications = [np.argmax(run(classifier, class_tfm, i)[0]) for i in fnames]
        segmentations = [run(segmenter, seg_tfm, i) for i in fnames]
        for s in segmentations:
            s[0] = (s[0] > 0).astype(np.uint8)*255
            s[1] = denorm_img(s[1], seg_tfm.denorm, reverse_ch=False)
    else:
        classifications = np.argmax(run(classifier, class_tfm, fnames)[0])
        segmentations = run(segmenter, seg_tfm, fnames)
        segmentations[0] = (segmentations[0] > 0).astype(np.uint8)*255
        segmentations[1] = denorm_img(
            segmentations[1], seg_tfm.denorm, reverse_ch=False)
    return classifications, segmentations


def get_basename(fn):
    return os.path.splitext(os.path.basename(str(fn)))[0]


def visualise(clss, sss, inp_ims=None, lbl_df=None, cls2str={0: 'keratosis', 1: 'melanoma', 2: 'benign nevus'}):
    '''
    Plots 12 examples from the segmantation and classification network.

    clss - classifier predictions
    sss - segmentater predictions
    inp_ims - the image filenames used to generate the predictions
    lbl_df - The Pandas dataframe containing the ground truth labels. DF info -> Index=image basenames, col='classes' class names
    cls2str - some kind of lookup table converting the integer classifications to words

    The bulk of the plotting code is from: https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
    '''
    grid_layout = (4, 3)
    num_grids = grid_layout[0] * grid_layout[1]
    fig = plt.figure(figsize=(20, 18))
    outer = gridspec.GridSpec(*grid_layout, wspace=0.2, hspace=0.2)

    for i in range(num_grids):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            if j == 1:
                ax.imshow(sss[i][1])
                ax.imshow(sss[i][0], alpha=0.3)
                ax.text(0.5, -0.1, 'Lesion Segmentation', size=12,
                        ha="center", transform=ax.transAxes)
            else:
                ax.imshow(sss[i][1])
                ax.text(0.5, -0.1, 'Image', size=12,
                        ha="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        
        mstr = f'Lesion classified as {cls2str[clss[i]]} \n'
        if lbl_df is not None and inp_ims is not None:
            gt_lbl = lbl_df.loc[get_basename(inp_ims[i]), 'classes']
            mstr += f'Dermatologist labelled lesion as {gt_lbl}\n'
        t = ax.text(0.5, 0.5, mstr, fontsize=14)
        t.set_verticalalignment('bottom')
        t.set_ha('center')
