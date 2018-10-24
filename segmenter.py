from fastai.conv_learner import *
from fastai.dataset import *
import sklearn.metrics as metrics


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


def get_base(arch, cut):
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)


def dice(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def jaccard(pred, targs):
    # iou = TP / (TP + FP + FN)
    pred = (pred > 0).float()
    return metrics.jaccard_similarity_score(targs, pred)


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
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
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
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


class UnetModel():
    def __init__(self, model, lr_cut, name='unet'):
        self.model, self.name, = model, name
        self.lr_cut = lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]


def plot_loss(train_losses, val_losses, rec_metrics, num_ep, epoch_iters):
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

def plot_preds(x,y,py):
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
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


def plot_data(x, y):
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
            ax = show_img(x[i], ax=ax)
            show_img(y[i], ax=ax, alpha=0.3)
            ax.set_title(f'Ground Truth {i}')
    plt.tight_layout(pad=0.1)


def g_fns(fpath, ext):
    return np.array(sorted([f for f in fpath.glob('*'+ext)]))


def im_hist(fn, n=None):
    nn = n if n is not None else ''
    sizes = [PIL.Image.open(x).size for x in fn]
    row_sz, col_sz = list(zip(*sizes))
    plt.hist(row_sz)
    plt.title(nn + ' Row Distributions')
    plt.figure()
    plt.hist(col_sz)
    plt.title(nn + ' Col Distributions')
