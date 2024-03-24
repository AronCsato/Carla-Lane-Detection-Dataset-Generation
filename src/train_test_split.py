from sklearn.model_selection import train_test_split

tf.random.set_seed(40)
X_train, X_test, bin_train, bin_test, inst_train, inst_test = train_test_split(image_ds, mask_ds, inst_ds, test_size = 0.15, random_state=0)
