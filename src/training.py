from focal_loss import BinaryFocalLoss
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import pickle

LR = 1e-4                 # Learning rate
EPOCHS = 100              # Number of epochs
BS = 15                   # Number of batch size
LossWeights = [10,1]      # Loss function weights for Binary Segmentation and Instance Segmentation
terminate = TerminateOnNaN()   # Terminate the training process if the loss reaches NaN
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)       #Early Stopping if model overfits
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #Save path for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  #Tensorboard
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,    #Checkpoint every epoch
                             save_weights_only=True, mode='auto', period=1)
filepath = "./output/saved-model.hdf5"     #save path for the checkpoint function

tf.random.set_seed(40)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, decay=LR/EPOCHS)   #Optimizer configuration
model.compile(optimizer=optimizer,
                  loss=[BinaryFocalLoss(gamma=2), instance_loss],
                  loss_weights=LossWeights,
                  metrics={'bin_seg': iou, 'inst_seg': "accuracy"})

history = model.fit(X_train, [bin_train, inst_train],
                       batch_size = BS,
                       verbose=1,
                       epochs=EPOCHS,
                       validation_data=(X_test, [bin_test, inst_test]),
                        shuffle=False,
                       callbacks=[terminate, tensorboard_callback, checkpoint, es])

np.save('./output/final/lane_4lane_6_model.npy',history.history)        #Save the training loss, acc, and meanIou
model.save('./output/final/lanenet_4lane2_6_model.hdf5')                #Save the trained weight
