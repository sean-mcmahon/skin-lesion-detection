from classifier import *

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

