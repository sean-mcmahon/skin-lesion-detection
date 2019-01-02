from fastai.conv_learner import *
from fastai.dataset import *
import sklearn.metrics as metrics

'''
By Sean McMahon - 2018. 

This script contains a series of classes and functions for performing semantic segmentation.

This is includes a custom dataset for pytorch/fastai loading, the UNet achitecture which is built on the ResNet architecure.

Other utility functions for evaluating model performance, and for visualising both data and network restuls.


(Fastai is a high level wrapper around Pytorch)
'''

def show_img(im, figsize=(5,5), ax=None, alpha=None):
    '''
    Plots and image, returns the Matplotlib axis for the figure.
    This is so we can plot two images on top of another, a segmentation mask and the input mask
    '''
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


class MatchedFilesDataset(FilesDataset):
    '''
    From Fastai Lesson 14 (U-Net or Carvanna).
    This is a dataset class for the Fastai library, returns an image as an input (rgb) and an image as a label (mask)
    '''
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


def get_base(arch, cut):
    '''
    Gets the base of the U-Net architecture. 
    Basically cuts off some of the layers of a base classifier network, such as resnet, to use as the base for UNet.
    From Fastai Lesson 14 (U-Net or Carvanna).
    '''
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)


def dice(pred, targs):
    '''
    Dice performance metric, similar to IoU. From Fastai Lesson 14 (U-Net or Carvanna).
    This takes Pytorch Variables (tensors) as inputs, does not work with numpy arrays
    '''
    pred = (pred > 0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def dice_n(pred, targs):
    '''
    Dice performance metric, similar to IoU. From Fastai Lesson 14 (U-Net or Carvanna).
    This uses numpy array, does not work with pytorch tensors
    '''
    pred = (pred > 0).astype(float)
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def jaccard(pred, targs):
    '''
    Also known as intersection over union (IoU), similar to the dice metric. 
    Never got a chance to finish this function. 
    '''
    # iou = TP / (TP+FP+FN)
    pred = (pred > 0).astype(float)



class SaveFeatures():
    '''
    This classes creates 'hooks', a pytorch functionality which allows us to pull logits from specific layers of a model
    From Fastai Lesson 14 (U-Net or Carvanna).
    '''
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    '''
    A mini network for the skip or (unet) connections of Unet.
    From Fastai Lesson 14 (U-Net or Carvanna).
    '''
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):
    '''
    Model for UNet34, which is unet based on Resnet34. 
    Uses the Unetblocks for the skip connections, and relies 
    on the hooks from SaveFeatures.
    From Fastai Lesson 14 (U-Net or Carvanna), 
    watch the lesson video for more info https://course.fast.ai/lessons/lesson14.html
    '''
    def __init__(self, rn):
        # define the layers you want to use.
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        # how the layers from __init__ get connected for the forward pass of the network.
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:, 0]

    def close(self):
        for sf in self.sfs:
            sf.remove()


def build_unet(backbone):
    '''
    Combines the above network construction code to build an instance of Unet34.
    backbone - Pytorch model (nn.Module), should be ResNet34 - 'resnet34' in fastai code

    Returns an instance of the model loaded onto a gpu if available.
    
    From Fastai Lesson 14 (U-Net or Carvanna)
    '''
    cut, lr_cut = model_meta[backbone]
    m_base = get_base(backbone, cut)
    unet = Unet34(m_base)
    # to_gpu will put model on gpu if one is available
    return to_gpu(unet)



def plot_loss(train_losses, val_losses, rec_metrics, num_ep, epoch_iters):
    '''
    Plots the loss functions from training. 

    train_losses - np array of trainining losses per iteration.
    val_losses   - np array of validation losses per iteration.
    rec_metrics  - the performance metrics at each epoch.
    num_ep       - The number of epochs trained for.
    epoch_iters  - The iteraction indexes for each epoch.
    '''
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].grid()
    ax[0].plot(list(range(num_ep)), val_losses, label='Validation loss')
    ax[0].plot(list(range(num_ep)), [train_losses[i-1]
                                     for i in epoch_iters], label='Training loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')
    ax[1].plot(list(range(num_ep)), rec_metrics)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim(bottom=min(0.5, min(rec_metrics)))
    ax[1].grid()


def plot_preds(x, y, py, splots=(5, 6), fsize=(12, 10)):
    '''
    Plot predictions with the input image. 
    Based on  plot_img_n_preds.

    x - input image np array (num_ims, w,h,3)
    y - ground truth mask, np array
    py - ouput mask from model, np array
    '''
    _, axes = plt.subplots(*splots, figsize=fsize)
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax = show_img(x[i], ax=ax)
            show_img(y[i], ax=ax, alpha=0.3)
            ax.set_title(f'Ground Truth {i}')
        else:
            ax = show_img(x[i-1], ax=ax)
            show_img(py[i-1] > 0, ax=ax, alpha=0.3)
            ax.set_title(f'Prediction {i-1}')
    plt.tight_layout(pad=0.1)


def plot_img_n_preds(x, py, splots=(5, 6), fsize=(12, 10)):
    '''
    plots the first splots[0] * splots[1] images from x
    x - array of input images, np array (num_ims, w,h,3)
    py - array of network prediction masks, np array
    splots - subplot shapes
    fsize - size of figure containing subplots (see matplotlib docs)
    '''
    _, axes = plt.subplots(*splots, figsize=fsize)
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax = show_img(x[i], ax=ax)
            # show_img(y[i], ax=ax, alpha=0.3)
            ax.set_title(f'Input Image {i}')
        else:
            ax = show_img(x[i-1], ax=ax)
            show_img(py[i-1] > 0, ax=ax, alpha=0.3)
            ax.set_title(f'Segmentation (yellow) {i-1}')
    plt.tight_layout(pad=0.1)


def plot_data(x, y, title='Ground Truth', splots=(5,6), fsize=(12,10)):
    '''
    Plots the first splots[0] * splots[1] images from x.
    x - array of input images, np array (num_ims, w,h,3).
    y - array of ground truth masks, np array.
    splots - subplot shapes.
    fsize - size of figure containing subplots (see matplotlib docs).

    Could this be refactored with plot_img_n_preds? Yes, but meh.
    '''
    _, axes = plt.subplots(*splots, figsize=fsize)
    for i, ax in enumerate(axes.flat):
            ax = show_img(x[i], ax=ax)
            show_img(y[i], ax=ax, alpha=0.3)
            ax.set_title(f'{title} {i}')
    plt.tight_layout(pad=0.1)


def g_fns(fpath, ext='.png'):
    '''
    get filenames.
    fpath - instance of Path of string containing the base directory for the images.
    '''
    if isinstance(fpath, str): fpath = Path(fpath)
    return np.array(sorted([f for f in fpath.glob('*'+ext)]))


def im_hist(fn, name='', path=None):
    '''
    Plots histogram of image sizes named in fn.

    fn list of filenames to get sizes from.
    '''
    if path:
        sizes = [PIL.Image.open(os.path.join(path, x)).size for x in fn]
    else:
        sizes = [PIL.Image.open(x).size for x in fn]
    row_sz, col_sz = list(zip(*sizes))
    plt.hist(row_sz)
    plt.title(name + ' Row Distributions')
    plt.figure()
    plt.hist(col_sz)
    plt.title(name + ' Col Distributions')
    print(f'Min row size {min(row_sz)}; Min col size {min(col_sz)}')
