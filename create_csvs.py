from classifier import *
import os

PATH = Path('/home/sean/hpc-home/skin_cancer/')

def getds(data): 
    dep = PATH / 'isic_archive/recod_titans_sub/data/deploy2017.txt'
    rec = pd.read_csv(dep)
    rec.columns = [ss.strip() for ss in rec.columns]
    rec = rec.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return rec.loc[rec['dataset']
                   == data][['image', 'melanoma', 'keratosis']]


def vis_classes(df, col='classes', sf=True):
    if sf: df[col].value_counts().plot(kind='barh')
    print(df[col].value_counts())

def add_class_col(df, cats=['melanoma', 'keratosis']):
    df['classes'] = np.nan
    for c in cats:
        # print('adding %s to class' % c)
        df['classes'].iloc[df[c].nonzero()[0]] = c
    df['classes'].fillna('nevus', inplace=True)
    # df.head()
    return df

def load_isic17_train():
    print('--- Load ISIC 17 Train set')
    ict = getds('challenge')
    ict.index = 'ISIC/ISIC-2017_Training_Data/' + ict['image'] + '.jpg'
    ict = ict[['melanoma', 'keratosis']]
    ict = add_class_col(ict)
    vis_classes(ict, sf=False)
    print(ict.head())
    return ict

def load_isic17_val():
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
    train = combine_training(half_ia_n=half_ia_n)
    print(' ---- ')
    test = load_isic17_test()
    train.loc[:, class_col].to_csv(train_fn)
    test.loc[:, class_col].to_csv(test_fn)
    return train, test

def create_half_nevus_datasets(path_):
    classes = ('melanoma', 'keratosis', 'classes')
    if not os.path.isdir(str(path_)): os.mkdir(str(path_))
    trn_n = path_ / 'train_{}_multi_halfn.csv'
    tst_n = path_ / 'test_{}_multi_halfn.csv'
    for cls_col in classes:
        create_multi_train_test(cls_col, trn_n.format(
            cls_col), tst_n.format(cls_col), half_ia_n=True)


# ------------------------------------------------------------------
# Lession Segmentation Dataset Functions
# ------------------------------------------------------------------

def check_paths(path_, iterable): 
    f_checks = [os.path.exists(path_ / i) for i in list(iterable)]
    check = all(f_checks)
    # if not check:
    #     inval = [i for c, i in enumerate(list(iterable)) if not f_checks[c]]
    #     print(inval)
    return check


def seg_lesion(im, mask):
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
    return np.array(sorted([f for f in fpath.glob('*'+ext)]))

def get_isic_ims_n_mask(image_folder, mask_folder):
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
    xtrn_d = PATH / 'dermofit/'
    d_files = get_files_in_dir(xtrn_d, '', '.png')
    ims = np.array(sorted([x for x in d_files if not 'mask' in x]))
    msks = np.array(sorted([x for x in d_files if 'mask' in x]))
    return ims, msks

def get_ph2_ims_and_mask():
    d_path = PATH / 'ph2dataset/PH2_Dataset_images/'
    d_files = get_files_in_dir(d_path, '', '.bmp')

    ims = np.array(sorted([x for x in d_files if 'Dermoscopic_Image' in x]))
    msks = np.array(sorted([x for x in d_files if '_lesion' in x]))
    return ims, msks

def segment_images():
    def get_and_append(ims, masks, cb):
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
    
    test_ims, test_masks = np.array(()), np.array(())
    test_ims, test_masks = get_and_append(test_ims, test_masks, get_test_isic17_ims_and_mask)

    print('Segmenting images.')
    create_seg_images(ims, masks)
    create_seg_images(test_ims, test_masks)
    print('Finished segmenting.')


def create_seg_csvs():
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
    train, test = create_seg_csvs()
    classes = ('melanoma', 'keratosis', 'classes')
    if not os.path.isdir(str(path_)):
        os.mkdir(str(path_))
    trn_n = str(path_ / 'train_seg_{}_multi.csv')
    tst_n = str(path_ / 'ISIC/test_seg_{}_multi.csv')
    for cls_col in classes:
        train.to_csv(trn_n.format(cls_col), columns=[cls_col])
        test.to_csv(tst_n.format(cls_col), columns=[cls_col])
