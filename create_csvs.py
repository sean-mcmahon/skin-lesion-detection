from classifier import *
import os

'''
This script creates the various training, validation and testing data splits
used for creating a skin lesion classifier.

All the skin cancer detection tasks are based around the ISIC 2017 Challenge:
https://www.isic-archive.com/#!/topWithHeader/tightContentTop/challenges
Which has numerous tasks, part 1 being semantic segmentaion of skin lesions,
and part 3 for classification between 3 forms of skin lesion.


This script assumes a certain stucture within the data directory (PATH).
There are four main datasets used;
1. ISIC 2017 Challenge (will have to create a login https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab)
    Paths used, and assumed in this code.
    a. Training Data (PATH/ISIC/ISIC-2017_Training_Data/) labels obtained from learningtitans rep (see below)
    b. Validation Data (PATH/ISIC/ISIC-2017_Validation_Data/) and Csv (PATH/ISIC/val_isic17.csv)
    a. Testing Data (PATH/ISIC/ISIC-2017_Test_v2_Data_Classification/) and Csv (PATH/ISIC/test_all_17.csv)
2. ISIC Archive Data (modified paths from https://github.com/learningtitans/isbi2017-part3#additional-isic-archive-images)
3. Dermofit Dataset (we paid for access - https://licensing.eri.ed.ac.uk/i/software/dermofit-image-library.html)
4. Ph2 Dataset (modified paths from https://github.com/learningtitans/isbi2017-part3#the-ph2-dataset)


The data structure and ground truths is adopted from https://github.com/learningtitans/isbi2017-part3
This is repostory from one of the top scoring teams of the ISIC 2017 Part 3 Challenge. Instructions
for ISIC Archive Data, and Ph2 datsets are in this repositoy, it also contains 'deploy2017.txt',
which I used to get the labels for all the data, except for the ISIC17p3 validation and testset.

Acessing HPC Version of all this data:
I've copied skin_cancer to the cyphy folder of HPC.
It's in /work/cyhpy/SeanMcMahon/datasets/skin_cancer
Can mount /work/cyhpy/ on your local machine for access.
To get a login to HPC, email hpc-support@qut.edu.au
Mounting script (will need to modify):
#!/bin/bash
sudo mount -t cifs //hpc-fs.qut.edu.au/work/cyphy /home/sean/hpc-cyphy -o user=n8307628,dom=QUTAD,uid=1000


By Sean McMahon - 2018
'''

# I've copied skin_cancer to the cyphy folder of HPC.
# It's in /work/cyhpy/SeanMcMahon/datasets/skin_cancer
# Can mount /work/cyhpy/ on your local machine for access (hpc).
# To get a login to HPC, email hpc-support@qut.edu.au
PATH = Path('/home/sean/hpc-home/skin_cancer/')

def getds(data): 
    '''
    loads a pandas data frame from deploy.txt

    This text file is from https://github.com/learningtitans/isbi2017-part3/blob/master/data/deploy2017.txt
    and contains the 3-class labels for the ISIC17 part3 challenge.

    data - the rows to extract, corresponding to one of the four datasets;
    challenge (isic17 training), isic (isic_archive), dermofit and ph2. 
    There are more but I could not find the data for them.

    returns a data frame for the desired dataset with the image name and label, either melanoma or keratorsis, neither is nevus.
    '''
    dep = PATH / 'isic_archive/recod_titans_sub/data/deploy2017.txt'
    rec = pd.read_csv(dep)
    rec.columns = [ss.strip() for ss in rec.columns]
    rec = rec.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return rec.loc[rec['dataset']
                   == data][['image', 'melanoma', 'keratosis']]


def vis_classes(df, col='classes', sf=True):
    '''
    Visualse the distribution of the 3 classes

    Make sure add_class_col() is run beforehand. 
    '''
    if sf: df[col].value_counts().plot(kind='barh')
    print(df[col].value_counts())

def add_class_col(df, cats=['melanoma', 'keratosis']):
    '''
    Adds a third column class to the dataframe obtained from getds().

    '''
    df['classes'] = np.nan
    for c in cats:
        # print('adding %s to class' % c)
        df['classes'].iloc[df[c].nonzero()[0]] = c
    
    # if neither a melanoma or keratosis, its assumed to be a nevus
    df['classes'].fillna('nevus', inplace=True)
    # df.head()
    return df

def load_isic17_train():
    '''
    Load the isic 2017 challenge part 3 training data.
    The data and can be downloded from the isic challenge website. 
    As can a csv with the labels, here the deploy.txt is used for the labels
    '''
    print('--- Load ISIC 17 Train set')
    # load the labels from the learningtitans repo
    ict = getds('challenge')
    # fastai assumes the image names to be the index, so there.
    ict.index = 'ISIC/ISIC-2017_Training_Data/' + ict['image'] + '.jpg'
    ict = ict[['melanoma', 'keratosis']]
    # add classes colums
    ict = add_class_col(ict)
    # visualise distrubtions and print the head of the DF. 
    # Gives a feel for the data and great for spotting errors.
    vis_classes(ict, sf=False)
    print(ict.head())
    return ict

def load_isic17_val():
    '''
    Load the isic 2017 challenge part 3 validation data. T
    The data and the csv used can be downloded from the isic challenge website
    '''
    print('--- Load ISIC 17 Validation')
    icv = pd.read_csv(PATH / 'val_isic17.csv')
    icv.columns = ['image', 'class']

    icv['melanoma'] = (icv['class'] == 'MEL').astype(int)
    icv['keratosis'] = (icv['class'] == 'SK').astype(int)
    icv.set_index(icv['image'], inplace=True)

    icv.drop(columns=['class', 'image'], inplace=True)
    icv = add_class_col(icv)

    vis_classes(icv, sf=False)
    print(icv.head())
    return icv

def load_isic_archive(p_stuff=True):
    '''
    Loads the isic archive data. This is additional labelled data not used in the ICIS challenge. 
    The authors of the 'learningtitans' claim to remove duplicates from the isic17 training set, did not validate.
    ISIC Archive data source (modified paths from https://github.com/learningtitans/isbi2017-part3#additional-isic-archive-images)
    '''
    if p_stuff: print('--- Load ISIC Archive Dataset')
    ia = getds('isic')
    ia = ia[['melanoma', 'keratosis']].set_index(
        'isic_archive/images/' + ia['image'] + '.jpg')
    ia = add_class_col(ia)
    if p_stuff:
        vis_classes(ia, sf=False)
        print(ia.head())
    return ia

def load_isic_archive_half_nevus():
    '''
    The isic arhive data, has tonnes of nevus, making it imbalanced. 
    As this is the least importatn class, I randomly extract hald the nevus class images from the dataset.
    This seemed to reduce the models overfitting.
    '''
    print('--- Load Half Nevus ISIC Archive Dataset')
    ia = load_isic_archive(p_stuff=False)

    # may not be the most effective, but it's what i know.
    # randomly sample half the nevus from icic archive dataset
    ia_nervi_id = np.array((ia['classes'] != 'nevus'))
    nervi_id = np.where(ia_nervi_id == False)[0]
    non_n_id = np.where(ia_nervi_id)[0]
    half_nervi = np.array(random.sample(nervi_id.tolist(), len(nervi_id)//2))
    new_idxs = np.concatenate([half_nervi, non_n_id])

    half_nervi_id = np.zeros(np.shape(ia_nervi_id)).astype(bool)
    half_nervi_id[new_idxs] = True

    half_nervi_df = ia.iloc[half_nervi_id, :]
    vis_classes(half_nervi_df, sf=False)
    print(half_nervi_df.head())
    return half_nervi_df

def load_dermofit():
    '''
    Load the dermofit dataset. 
    Use the deploy.txt labels, they converted dermofit labels into a 3class problem for the isic 2017 challenge.
    This is a paid dataset, QUT has paid for access, might have to ask around.

    There is also a csv file with the dermofit library that needs to be downloaded.
    '''
    print('--- Load Dermofit Dataset')
    ddf = getds('dermofit')
    ddf = add_class_col(ddf)
    ddf_t = pd.read_csv(PATH / 'dermofit/train.csv')
    ddf_t.columns = ['image', 'class']

    ddf_ids = ddf['image'].tolist()
    ddf_pths = ddf_t['image'].tolist()
    [os.path.basename(ss) for ss in ddf_pths] == ddf_ids
    dermfit_image_id = ['dermofit/' + i + '.png' for i in ddf_pths]
    print(all([(PATH / i).exists() for i in dermfit_image_id]))
    ddf.index = dermfit_image_id
    ddf.drop(columns=['image'], inplace=True)

    vis_classes(ddf, sf=False)
    print(ddf.head())
    return ddf

def load_ph2():
    '''
    Loads the ph2 dataset.
    Data download (modified paths from https://github.com/learningtitans/isbi2017-part3#the-ph2-dataset)
    '''
    print('--- Load PH2 Dataset')
    pdf = getds('ph2')
    pdf = add_class_col(pdf)
    # pdf.set_index(pdf['image'], inplace=True)
    pims = pdf['image'].tolist()
    pims = ['ph2dataset/PH2_Dataset_images/'+i +
            f'/{i}_Dermoscopic_Image/{i}.bmp' for i in pims]
    print(all([(PATH / ss).exists() for ss in pims]))
    pdf.index = pims
    pdf.drop(columns=['image'], inplace=True)
    # pdf = pdf[['melanoma', 'keratosis']]
    vis_classes(pdf, sf=False)
    print(pdf.head())
    return pdf

def load_isic17_test():
    '''
    Loads the icis 2017 part 3 testset. 
    Can download the data and csv with labels from the website. 
    You might have to do some modifications of your own to the csv downloaded from there (hopefully not).
    '''
    print('--- Load ISIC 17 Testset')
    test_all_p = PATH / 'ISIC/test_all_17.csv'
    test_df = pd.read_csv(test_all_p, index_col=0)
    test_df['classes'][test_df['classes'] ==
                       'seborrheic_keratosis'] = 'keratosis'
    test_df['melanoma'] = (test_df['classes'] == 'melanoma').astype(int)
    test_df['keratosis'] = (test_df['classes'] == 'keratosis').astype(int)
    test_df.index = [ss.replace(
        '/ISIC-2017_Test_v2_Data/', '/ISIC-2017_Test_v2_Data_Classification/') for ss in test_df.index]
    vis_classes(test_df, sf=False)
    print(test_df.head())
    return test_df

def combine_training(half_ia_n=False):
    '''
    Load the training set, theres a flag to load half the nevus images from the isic archive dataset (half_ia_n).
    Would reccomend using it.

    Returns a Pandas DataFrame. 
        Indexs are the image filenames within PATH
        3 Columns 'melanoma', 'keratosis' and 'class'
    '''
    isic_train = load_isic17_train()
    archive    = load_isic_archive_half_nevus() if half_ia_n else load_isic_archive()
    dermo      = load_dermofit()
    ph2        = load_ph2()
    isic_val   = load_isic17_val()
    print('\n------\n')
    multi_train = pd.concat([isic_train, archive, dermo, ph2, isic_val])
    print('Multi Dataset Trainset... {} elements'.format(len(multi_train)))
    vis_classes(multi_train, sf=False)
    print(multi_train.head())
    return multi_train

def create_multi_train_test(class_col, train_fn, test_fn, half_ia_n=False):
    '''
    Get a training and testing set, as pandas data frames. 
    There are three tasks you can try. 3-class classiciation, binary classification of melanomas or binary classification of keratosis
    '''
    train = combine_training(half_ia_n=half_ia_n)
    print(' ---- ')
    test = load_isic17_test()
    train.loc[:, class_col].to_csv(train_fn, co)
    test.loc[:, class_col].to_csv(test_fn)
    return train, test

def create_half_nevus_datasets(path_):
    '''
    This is the function I ended up using to genereate the training and testing splits. 
    Automatically saves train and test csvs to file (for fastai data loading).
    Two for each of the three tasks; 3-class classiciation, binary classification of melanomas or binary classification of keratosis

    Note the save paths and filenames.
    '''
    classes = ('melanoma', 'keratosis', 'classes')
    if not os.path.isdir(str(path_)): os.mkdir(str(path_))
    trn_n = str(path_ / 'train_{}_multi_halfn.csv')
    tst_n = str(path_ / 'ISIC/test_{}_multi_halfn.csv')

    train = combine_training(half_ia_n=True)
    print(' ---- ')
    test = load_isic17_test()
    
    for cls_col in classes:
        train.to_csv(trn_n.format(cls_col), columns=[cls_col])
        test.to_csv(tst_n.format(cls_col), columns=[cls_col])
    return train, test


# ------------------------------------------------------------------
# Lesion Segmentation Dataset Functions
# ------------------------------------------------------------------
'''
To improve performance, I tried initially training the networks on images segmented around the lesions,
with minimal background. 

This does improve performance, but takes longer to train. 
Plus you need to save the additional images and create additional csv label files; which is what the code below is for ;)
'''


def check_paths(path_, iterable): 
    '''
    Makes sure the images within iterable, which are filenames within path_, are valid.
    '''
    f_checks = [os.path.exists(path_ / i) for i in list(iterable)]
    check = all(f_checks)
    # if not check:
    #     inval = [i for c, i in enumerate(list(iterable)) if not f_checks[c]]
    #     print(inval)
    return check


def seg_lesion(im, mask):
    '''
    Returns a new image segmented around the lesion, given an image to segment 
    and a mask containing the pixel locations of the lesion.

    im is either a full path to an image or a numpy array of an image
    same with mask.

    returns a new image with the background removed, segmented around the lesion
    '''
    im = im if isinstance(im, np.ndarray) else open_image(im)
    mask = mask if isinstance(mask, np.ndarray) else open_image(mask)
    mask = mask[:, :, 0] if mask.ndim > 2 else mask
    blob_coords = np.where(mask == 1)
    blob_ys = blob_coords[0]
    blob_xs = blob_coords[1]

    ly = min(blob_ys)
    uy = max(blob_ys)
    lx = min(blob_xs)
    ux = max(blob_xs)

    t_left = [lx, ly]
    h = uy - ly
    w = ux - lx

    return im[t_left[1]:t_left[1]+h, t_left[0]:t_left[0]+w]

def create_seg_images(img_names, mask_names):
    '''
    With a list of image names and corresponding mask names. Create new segemented images.
    '''
    if len(img_names) != len(mask_names):
        raise ValueError('Number of images does not match number of masks')

    for count, (im, mask) in enumerate(zip(img_names, mask_names)):
        im = str(im)
        par_dir = str(os.path.dirname(im)) # dirname strips final '/'
        if '_lesion_seg' in par_dir: raise ValueError('Invalid filename {}'.format(par_dir))
        n_par_dir = par_dir + '_lesion_seg'
        nname = im.replace(par_dir, n_par_dir)
        if not os.path.isdir(os.path.dirname(nname)):
            os.mkdir(os.path.dirname(nname))
        if os.path.exists(nname):
            continue
        # if filename does not exist do this
        if count % 100 == 0:
            print('Saved {}/{} images'.format(count, len(img_names)))
        seg_im = seg_lesion(im, mask)
        scipy.misc.imsave(nname, seg_im)


def g_fns(fpath, ext='.png'):
    '''
    Get filenames with ext within directory fpath.
    '''
    return np.array(sorted([f for f in fpath.glob('*'+ext)]))

def get_isic_ims_n_mask(image_folder, mask_folder):
    '''
    Gets images and masks from ISIC dataset. 
    This because of the structure I used to download all the ISIC 2017 challenge data.
    '''
    xtrn_i = PATH / image_folder
    ytrn_i = PATH / mask_folder

    ims = g_fns(xtrn_i, '.jpg')
    mks = g_fns(ytrn_i, '.png')
    return ims, mks

def get_train_isic17_ims_and_mask():
    return get_isic_ims_n_mask('ISIC/ISIC-2017_Training_Data', 
                               'ISIC/ISIC-2017_Training_Part1_GroundTruth')

def get_val_isic17_ims_and_mask():
    return get_isic_ims_n_mask('ISIC/ISIC-2017_Validation_Data',
                               'ISIC/ISIC-2017_Validation_Part1_GroundTruth')

def get_test_isic17_ims_and_mask():
    return get_isic_ims_n_mask('ISIC/ISIC-2017_Test_v2_Data',
                               'ISIC/ISIC-2017_Test_v2_Part1_GroundTruth')

def get_files_in_dir(p, fl, ext):
    '''
    Recursively searches a directory (p) for all files with the desired extension (ext).
    fl is a filter; this functionality was not tested.

    returns a list of all the files matching the criteria
    '''
    ims_l = []
    for root, dirs, files in os.walk(p):
        for fn in files:
            if fn.endswith(ext) and fl in fn:
                # The right filetype and the image we want!
                ims_l.append(os.path.join(root, fn))
    if len(ims_l) == 0:
        print('No images found')
    return ims_l

def get_dermo_ims_and_mask():
    '''
    find the dermofit images and masks. 
    xtrn_d is the path to where you've downloaded the images. If the names are unchanges it'll work bro.
    '''
    xtrn_d = PATH / 'dermofit/'
    d_files = get_files_in_dir(xtrn_d, '', '.png')
    ims = np.array(sorted([x for x in d_files if not 'mask' in x]))
    msks = np.array(sorted([x for x in d_files if 'mask' in x]))
    return ims, msks

def get_ph2_ims_and_mask():
    '''
    from d_path with the ph2 dataset images, recursively search to find all images.
    '''
    d_path = PATH / 'ph2dataset/PH2_Dataset_images/'
    d_files = get_files_in_dir(d_path, '', '.bmp')

    ims = np.array(sorted([x for x in d_files if 'Dermoscopic_Image' in x]))
    msks = np.array(sorted([x for x in d_files if '_lesion' in x]))
    return ims, msks

def segment_images():
    '''
    Finds all the images and masks, then segmentes them.
    '''
    def get_and_append(ims, masks, cb):
        '''
        adds to ims and masks from new ims and masks pulled from cb. 

        ims - np array of all images gathered so far
        masks - np array of all masks gathered so far
        cb - callback for getting more images and masks
        '''
        ims_, masks_ = cb()
        # remove segmented filenames from list. The create_seg_images will skip already segmented images.
        ims_ = np.array([i for i in ims_ if '_lesion_seg/' not in str(i)])
        es = 'Different Lengths; ims {} and masks {}\nims: {}\nmasks: {}'
        assert len(ims_) == len(masks_), es.format(
            len(ims_), len(masks_), ims_[0:5], masks_[0:5])
        return np.concatenate([ims, ims_]), np.concatenate([masks, masks_])
    print('Getting image and mask filenames')
    ims, masks = np.array(()), np.array(())
    ims, masks = get_and_append(ims, masks, get_train_isic17_ims_and_mask)
    ims, masks = get_and_append(ims, masks, get_dermo_ims_and_mask)
    ims, masks = get_and_append(ims, masks, get_ph2_ims_and_mask)
    ims, masks = get_and_append(ims, masks, get_val_isic17_ims_and_mask)
    
    print('Getting testset image and mask filenames')
    test_ims, test_masks = np.array(()), np.array(())
    test_ims, test_masks = get_and_append(test_ims, test_masks, get_test_isic17_ims_and_mask)

    print('Segmenting images.')
    create_seg_images(ims, masks)
    create_seg_images(test_ims, test_masks)
    print('Finished segmenting.')


def create_seg_csvs():
    '''
    With the segmented input images created, this creates the csv files needed for the fastai dataloader.

    Bad function name, actually creates Pandas dataframes which are returned 'create_seg_datasets' saves them as csvs
    '''
    def seg_paths(df):
        ids = list(df.index)
        ni = []
        for i in ids:
            par_dir = os.path.basename(os.path.dirname(i)).strip()
            n = par_dir + '_lesion_seg'
            ni.append(i.replace(par_dir+'/', n+'/'))
        return ni
    isic_train = load_isic17_train()
    dermo = load_dermofit()
    ph2 = load_ph2()
    isic_val = load_isic17_val()
    print('\n------\n')
    multi_train = pd.concat([isic_train, dermo, ph2, isic_val])
    print('Multi Dataset Trainset... {} elements'.format(len(multi_train)))
    #print(multi_train.head())

    multi_train.index = seg_paths(multi_train)
    vis_classes(multi_train, sf=False)

    test_df = load_isic17_test()
    test_df.index = seg_paths(test_df)
    vis_classes(test_df, sf=False)
    
    return multi_train, test_df
    

def create_seg_datasets(path_):
    '''
    With the segmented images created, this script gets the Pandas DataFrame labels and saves the three columns as three csvs

    Two csvs (train and test) for each tasks; 3 class classification, and binary classification of melanoma and keratosis
    The three lesions are nevus, melanoma and keratosis.
    '''
    train, test = create_seg_csvs()
    classes = ('melanoma', 'keratosis', 'classes')
    if not os.path.isdir(str(path_)):
        os.mkdir(str(path_))
    trn_n = str(path_ / 'train_seg_{}_multi.csv')
    tst_n = str(path_ / 'ISIC/test_seg_{}_multi.csv')
    for cls_col in classes:
        train.to_csv(trn_n.format(cls_col), columns=[cls_col])
        test.to_csv(tst_n.format(cls_col), columns=[cls_col])
    return train, test
