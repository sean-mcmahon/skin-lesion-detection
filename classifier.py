from fastai.conv_learner import *
from fastai.plots import plot_confusion_matrix
import sklearn.metrics as metrics
import os
import matplotlib.pyplot as plt

'''
By Sean McMahon - 2018
'''


def rand_by_mask(mask, preds, mpl=4): 
    '''
    From Fastai Lesson 1. Make a random selection from the data.
    '''
    return np.random.choice(np.where(mask)[0], min(len(preds), mpl), replace=False)


def rand_by_correct(is_correct, preds, val_y): 
    '''
    From Fastai Lesson 1. Randomly select correct predictions
    '''
    return rand_by_mask((preds == val_y)==is_correct, preds)


def plots(ims, figsize=(12,6), rows=1, titles=None):
    '''
    Plots images in a subplot.
    '''
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


def sample_ims(path, c, data_it, numimgs=9, figsize=(24, 12)):
    '''
    Based off code from Fastai lesson 1. Plots some random images for class c.
    path - path to the dataset
    c - the class index.
    data_it - Fastai data iterator for loading data.
    '''
    ys = data_it.trn_y
    ds = data_it.trn_ds
    cls2n = data_it.classes
    idm = rand_by_mask(c == ys, ys, mpl=numimgs)
    ims = [load_img_id(path, ds, i) for i in idm]
    r = np.ceil(len(ims) / 3).astype(int)
    tt = 'Sample {} images (class id={}; {} total)'.format(
        cls2n[c], c, np.sum(ys == c))
    plots(ims, figsize=figsize, rows=r, titles=[
          "Im Id: {}".format(i) for i in idm])
    plt.suptitle(tt, fontsize=24)


def load_img_id(path, ds, idx): 
    '''
    Loads an image.
    path - path to the data set.
    ds - dataset, a fastai class, different for different datasets and tasks
    idx - the index number of the image to load.
    '''
    return np.array(PIL.Image.open(str(path / ds.fnames[idx])))


def most_by_mask(mask, mult, probs):
    '''
    Return the image indexes for the top four probabilites. From Fastai lesson 1.

    probs - Network prediction probabilities (usually final logits after softmax)
    mask  - used to select preds from certain class
    mult  - used to either get the highest or lowest scores (1 or -1) from probs.
    '''
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]


def most_by_correct(y, is_correct, preds, probs, val_y):
    '''
    Returns the 4 predictions with the highest probability per class. From Fastai lesson 1.
    '''
    mult = -1 if (y == 1) == is_correct else 1
    return most_by_mask(((preds == val_y) == is_correct) & (val_y == y), mult, probs)


def plot_val_with_title(d, probs, preds, idxs, title=''):
    '''
    Plots some images from the validation set with their class.
    d     - dataset class instance. A fastai data. 
    probs - Output probabilites from the network/model
    preds - Final predictions from the model. Could just have probs as input and calculate preds, but meh.
    idxs  - The indexs for the images to plot with classification results.

    '''
    imgs = [load_img_id(d.path, d.val_ds, x) for x in idxs]
    p_str = '{}\nGT: {}\nPred: {}'
    title_probs = [p_str.format(round(
        probs[x], 2), d.classes[d.val_y[x]], d.classes[preds[x]]) for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16, 8)) if len(imgs) > 0 else print('Not Found.')


def load_csv_labels(csv_, folder, s='.jpg'):
    '''
    For loading the test set labels from test_csv.
    csv_course is a Fastai function.
    '''
    _, y, _ = csv_source(folder, csv_, suffix=s)
    print('Loading ys from csv; shape {}; vals {}; in folder "{}"'.format(
        y.shape, np.unique(y), folder))
    return y


def run_test(learner, ts=False, test_csv=None, sf=False,
             tta=True, test_folder=None):
    '''
    Generate values for printing or plotting the three metrics:
    confusion matrix (cm)
    ROC curve values (Receiver Operating Characteristic)
    Accuracy value
    
    Works for binary and multi-class classification problems.

    learner - Fastai learner class (fastai/old/learner.py)
    ts - test (bool)
    test_csv - csv to load the labels from
    sf - show figures
    tta - test time augmentation, averages classifications across 5 augmentations of a single image.
        Gets better performance, but takes longer to run. 
    test_folder - the folder where the test_csv is within 'path'

    returns:
    final network classifications
    y - Ground truth labels
    acc - Accuracy
    cm - confusion matrix
    roc_auc - ROS area under curve for all classes
    fpr - false positive rates
    tpr - true positive rates
    '''
    if ts and test_csv is None:
        raise ValueError(
            'Need if running on testset, provide a test_csv file for labels')
    if ts and test_folder is None:
        raise ValueError(
            'Need if running on testset, provide a test_folder where the test images are within PATH')
    if ts and not os.path.isfile(str(test_csv)): 
        raise FileNotFoundError(
            f'test_csv does not exist - "{test_csv}"')
    if tta:
        # TTA method within fast ai learner
        log_preds, y = learner.TTA(is_test=ts)
        probs = np.exp(log_preds).mean(axis=0)  # average of TTA
    else:
        dl1 = learner.data.test_dl if ts else learner.data.val_dl
        # from fastai - models.py
        log_preds, y = predict_with_targs(learner.model, dl1)
        probs = np.exp(log_preds)
    if ts and np.all(y == 0):
        # My wrapper around fastai csv loading function
        y = load_csv_labels(csv_=test_csv, folder=test_folder)
    
    final_preds = np.argmax(probs, 1)
    # accuracy
    acc = metrics.accuracy_score(y, final_preds)
    print(f'Accuracy = %0.2f' % acc)
    # confusion matrix
    cm = metrics.confusion_matrix(y, final_preds)
    print(f'Confusion Matrix:\n{cm}')
    # ROC curve values
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(learner.data.classes)):
        cy = np.array(y == i).astype(int)
        cp = probs[:, i]
        fpr[i], tpr[i], _ = metrics.roc_curve(cy, cp)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        print('AUC for class {}, id {} = {:0.2f}'.format(
            i, learner.data.classes[i], roc_auc[i]))

    if sf:
        print('-'*40)
        # plot the performance figures
        performance_figs(learner.data.classes, cm, roc_auc, fpr, tpr)
    return final_preds, y, acc, cm, roc_auc, fpr, tpr


def performance_figs(classes, cm, roc_auc, fpr, tpr):
    '''
    Plot the perforamance metrics for visualisation

    classes - the class names used in the dataset
    cm - confusion matrix
    ros_auc - Receiver Operating Characteristic (ROC) - Area Under Curve
    fpr - False positive rate
    tpr - True positive rate
    '''
    plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues)
    for i in range(len(classes)):
        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='AUC (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for {} id {}'.format(classes[i], i))
        plt.legend(loc="lower right")
        plt.show()


class ClassifierTrainer():

    def __init__(self, path, arch, sz, bs, trn_csv, aug_tfms=transforms_top_down,
                  train_folder='', test_folder=None, val_idx=None, test_csv=None,
                 lr=1e-2, sn=None, num_workers=8, precom=False, params_fn=None, weights=None):
        '''
        Wrapper class around the fastai network training classes. Prints information about the training data,
        Creates a data iterator, intialises the network leaner. Contains plotting evaluation and training methods.

        path - path to directory where all the data is stored. Weights will also be saved an loaded to 'path/models'.
        arch - the CNN architecture, usually resnet101. 'resnet101' is actually a handle for a fastai class which initalises the network in Pytorch.
        sz - image size
        bs - batch size, reduce if running out of memory
        trn_csv - the csv file containing all the image names and their label. Two columns assumed, col 1 with image name relative to 'path' 
            and col 2 with the labels as either numbers of strings
        aug_tfms - agumentation transformations 
        train_folder - the folder within 'path' containing the images, I usually leave this blank, opting to put the folder in col 1 of trn_csv.
        test_folder - the folder within 'path' where the test images are, this is used to load the images into the network when testing.
        test_csv - used by my weak wrapper around fastai code to get the testset labels.
        val_idx - the  indexs within trn_csv for images in the valiation set. If None, a random selection of the training set will be used.
        lr - Learning rate, 1e-2 should work most the time, try lowering to 1e-3, 1e-4 if training doesnt work. 
        sn - Save name, the name to save the weighs to during training. Again weights saves as, path / "models" / sn .
        precom - whether or not to precompute some CNN latent features before training. In general, dont use this. Here for historical reasons
        num_workers - number of data loaders to intialise, keep within number of cpus on computer.
        params_fn - filename of the parameters file, used when training on HPC so I know where to save the loss function plots too.
        weights - If you want to initialise the CNN from a previou traininig session add the string. Otherwise ImageNet pre-training will be used.

        Note. Fastai is a library built on top of the deep learning library Pytorch. 
        '''
        self.arch = arch
        self.dlr = lr
        self.test_folder = test_folder
        self.test_csv = test_csv
        self.params_fn = params_fn
        if sn is None:
            self.sn = 'train_' + self.arch.__name__
        else:
            self.sn = sn
        print(f'Saving model as "{self.sn}"')
        print('-> Train set value counts')
        trn_df = pd.read_csv(path / trn_csv)
        print(trn_df.iloc[:, -1].value_counts())

        if test_csv:
            print('Test set value counts')
            tst_df = pd.read_csv(path / test_csv)
            print(tst_df.iloc[:, -1].value_counts())
        
        # Dataset augmentations from fastai/old/fastai/transforms.py
        tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down)

        # The Fastai dataloader, used for training and evaluation, has numerous useful functions for:
        # loading data, preprocessing, batching, obtaining basic stats, and more
        self.data = ImageClassifierData.from_csv(path, train_folder, trn_csv, tfms=tfms,
                                                 suffix='', bs=bs, test_name=self.test_folder,
                                                 val_idxs=val_idx, num_workers=num_workers)
        if self.test_folder: self.data.test_ds.fnames = sorted(self.data.test_ds.fnames)
        print('Dataset has: {} classes'.format(self.data.classes))

        # Initialse the Fast ai Convolutional Neural Network (CNN or Conv) Learner. 
        self.learn = ConvLearner.pretrained(
            self.arch, self.data, precompute=precom)

        if weights:
            print('Loading weights "{}"'.format(weights))
            self.load(weights)

        # If Cuda is true, we training on a GPU, which is what we want.
        # If Cudnn is true our training should be slightly quicker.
        print('Cuda: {}; Cudnn {}'.format(
            torch.cuda.is_available(), torch.backends.cudnn.enabled))
        # Can comment this if statement out if you want to train on a CPU.
        if not torch.cuda.is_available():
            raise Exception('Cuda not available ' + 'Cuda: {}; Cudnn {}'.format(
                torch.cuda.is_available(), torch.backends.cudnn.enabled))

    def lr_find(self, sf=True):
        '''
        Fastai method for determinining the initial learning rate.
        This will generate a plot and you want the highest LR before the Loss starts to increase.
        Usually wise to add a bit or margen between your chosen LR and where the Loss starts to increase.
        '''
        lrf = self.learn.lr_find()
        if sf: self.learn.sched.plot()
        return lrf

    def set_lr(self, lr):
        self.dlr = lr

    def init_fit(self, name=None):
        '''
        Some intial trainining. Don't have to use
        '''
        sn = name if name else self.sn
        self.learn.fit(self.dlr, 2)
        self.learn.fit(self.dlr, 2, cycle_len=1)
        if self.learn.precompute:
            self.learn.precompute = False
            self.learn.fit(self.dlr, 1)
        self.learn.save(sn)
        print('Saved weights as "{}"'.format(sn))

    def inter_fit(self, name=None):
        '''
        Some intermediate trainining. Don't have to use
        '''
        sn = name if name else self.sn
        self.learn.precompute = False
        self.learn.fit(self.dlr, 2, cycle_len=1, cycle_mult=2)
        self.learn.save(sn)
        print('Saved weights as "{}"'.format(sn))
        self.plot_training(sn)

    def final_fit(self, name=None):
        '''
        Some final trainining. Don't have to use. 
        In fact for skin cancer, the models/networks tend to overfit during this training scheme. 
        '''
        wd = 5e-4
        sn = name if name else self.sn
        self.learn.unfreeze()
        lrs = np.array([self.dlr / 100, self.dlr / 10, self.dlr])
        self.learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, wds=wd)
        self.plot_training(sn + 'a')
        self.learn.fit(lrs, 5, cycle_len=3, wds=wd)
        self.plot_training(sn + 'b')
        self.learn.save(sn)
        print('Saved weights as "{}"'.format(sn))
        
    def test_val(self, tta=False, sf=False):
        '''
        Test model on the validation set
        Prints perforamnce metrics the return is for double checking and debugging.
        returns predictions and performance metrics
        '''
        *res, = run_test(self.learn, sf=sf, tta=tta)
        return res

    def test_eval(self, t_csv=None, tta=False, sf=False):
        '''
        Test model on the evaluation or test set
        Prints perforamnce metrics the return is for double checking and debugging.
        returns predictions and performance metrics
        '''
        if t_csv: self.test_csv = t_csv
        if self.test_csv is None:
            print('no test labels given')
            return
        self.check_test_names()
        *res, = run_test(
            self.learn, ts=True, sf=sf, test_csv=self.test_csv,
            test_folder=self.test_folder, tta=tta)
        return res

    def train(self, sn=None, lr=None):
        '''
        Overall training scheme I used when training models on the HPC. 
        Would not reccomend for use in jupyter notebooks, too time consuming.
        '''
        if lr: self.set_lr(lr)
        if sn: self.sn = sn
        
        self.init_fit(self.sn + '_1')
        print('-'*50)
        self.test_val()
        self.inter_fit(self.sn + '_2')
        self.test_val()
        self.test_eval()
        print('-'*50)
        self.set_lr(self.dlr / 2)
        self.final_fit(self.sn + '_3')
        self.test_val()
        print('-'*50)
        self.test_eval(tta=True, sf=True)

    def load(self, fn, pc=False):
        '''
        Load the weights fn into the network.
        learn.load is a fastai function.
        pc - precompute
        '''
        # if precompute set to True b default
        # but when loading models without precomute, it needs to be false.
        self.learn.precompute = pc
        self.learn.load(fn)
        # load_model(self.learn.model, self.get_model_path(name))

    def check_test_names(self, suf='.jpg'):
        '''
        Double check the labels and test images are loaded in the same order. 
        Most of the time this is correct. 
        '''
        def folder_and_name(path):
            folder_n = os.path.basename(os.path.dirname(path))
            fname = os.path.basename(path)
            return os.path.join(folder_n, fname)
        def base_fnames(paths):
            if isinstance(paths, str):
                return folder_and_name(paths)
            elif isinstance(paths, list) or isinstance(paths, tuple):
                return [folder_and_name(ss) for ss in paths]
            else:
                return paths
        fnames, _, _ = csv_source(self.test_folder, self.test_csv, suffix=suf)
        d_tfold = os.path.join(self.test_folder, self.test_folder)
        fnames = [str(ff).replace(suf*2, suf).replace(d_tfold, self.test_folder)
                  for ff in fnames]
        if self.learn.precompute:
            t_fns = self.data.test_ds.fnames
        else:
            t_fns = self.learn.data.test_ds.fnames
        # fnames = base_fnames(fnames)
        # t_fns = base_fnames(t_fns)
        if fnames != t_fns:
            print('Csv fnames:\n{}'.format(fnames[0:5]))
            print('Data loader fnames:\n{}'.format(t_fns[0:5]))
            es = 'Testset file names do no match! Try sorting "self.learn.data.test_ds.fnames"'
            raise Exception(RuntimeError(es))

    def plot_training(self, save_name):
        '''
        Saves a plot of the training and validation losses and the validation accuries to file. 
        Optional.
        '''
        def plot_loss(train_losses, val_losses, rec_metrics, num_ep, epoch_iters):
            fig, ax = plt.subplots(2, 1, figsize=(8, 12))
            ax[0].grid()
            ax[0].plot(list(range(num_ep)), val_losses, label='Validation loss')
            ax[0].plot(list(range(num_ep)), [train_losses[i-1]
                                            for i in epoch_iters], 
                                            label='Training loss')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend(loc='upper right')
            ax[1].plot(list(range(num_ep)), rec_metrics)
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_ylim(bottom=min(0.5, min(rec_metrics)))
            ax[1].grid()
            return fig, ax

        trnl = self.learn.sched.losses
        vall = self.learn.sched.val_losses
        metrics = self.learn.sched.rec_metrics
        num_ep = self.learn.sched.epoch
        ep_iterations = self.learn.sched.epochs
        fig, ax = plot_loss(trnl, vall, metrics, num_ep, ep_iterations)

        fig_fold = os.path.splitext(self.params_fn)[0]
        if not os.path.isdir(fig_fold):
            os.mkdir(fig_fold)
        fig_fn = os.path.join(fig_fold, os.path.basename(save_name))
        fig_fn = os.path.splitext(fig_fn)[0] + '.png'
        print('Saving plots as "{}"'.format(fig_fn))
        fig.savefig(fig_fn, dpi=fig.dpi)
