def read_txt(root, flag):
    img_path = []
    bin_path = []
    inst_path = []

    train_file = ops.join(root, 'train.txt')
    val_file = ops.join(root, 'val.txt')
    test_file = ops.join(root, 'test.txt')

    if flag == 'train':
        assert exists(train_file)
        file_open = train_file
    elif flag == 'valid':
        assert exists(val_file)
        file_open = val_file
    else:
        assert exists(test_file)
        file_open = test_file

    df = pd.read_csv(file_open, header=None, delim_whitespace=True, names=['img', 'bin', 'inst'])
    #print(df)
    img_path = df['img'].values
    bin_path = df['bin'].values
    inst_path = df['inst'].values

    #print(img_path)
    return img_path, bin_path, inst_path

def preprocessing(img_path, bin_path, inst_path, resize=(512,256))
    image_ds = []
    for i, image_name in enumerate(img_path):
        image = cv2.imread(image_name)
        image = Image.fromarray(image)
        image = image.resize(resize)
        image_ds.append(np.array(image, dtype=np.float32))
     
    mask_ds = []
    for i, image_name in enumerate(bin_path):
        image = cv2.imread(image_name, 0)
        image = Image.fromarray(image)
        image = image.resize(resize)
        label_binary = np.zeros([resize[1], resize[0]], dtype=np.uint8)
        mask = np.where(np.array(image)[:,:] != [0])
        #print(np.unique(np.array(image)))
        label_binary[mask] = 1
        mask_ds.append(np.array(label_binary, dtype=np.uint8))
        
    inst_ds = []
    for i, image_name in enumerate(inst_path):
        image = cv2.imread(image_name, 0)
        image = Image.fromarray(image)
        image = image.resize(resize)
        #print(np.unique(np.array(image)))
        inst_ds.append(np.array(ex,dtype=np.float32))
        
    return image_ds, bin_ds, inst_ds