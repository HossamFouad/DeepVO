# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from keras import metrics
from keras.layers import Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from keras.layers import Dropout, LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import keras.backend as K
K.set_image_data_format('channels_last')
from keras import optimizers
import pickle
from readdata import DataReader
import tensorflow as tf
from keras.models import load_model
with open('weight.pkl', 'rb') as f:
    weights_intializer = pickle.load(f)


# Hyperparameter
n_a = 1000 #hidden state 
TRAINABLE=False
time_sequence=16
BATCH_SIZE=32
epochs=200
output_dim=6
learning_rate= 0.001
k_param=100
load_existing_model=False
steps=0
def CNNModel(shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.     
    model=Sequential()
    
    model.add(TimeDistributed(ZeroPadding2D((3, 3)),input_shape=(time_sequence,1280,384,6)))
    model.add(TimeDistributed(Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1'
               ,kernel_initializer = lambda x: weights_intializer['conv1.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv1.0.bias'])))
    #model.layers[1].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((2, 2))))
    model.add(TimeDistributed(Conv2D(128, (5, 5), strides = (2, 2), name = 'conv2' 
               ,kernel_initializer = lambda x: weights_intializer['conv2.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv2.0.bias'],trainable=TRAINABLE)))
    #model.layers[4].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((2, 2))))
    model.add(TimeDistributed(Conv2D(256, (5, 5), strides = (2, 2), name = 'conv3'
                ,kernel_initializer = lambda x: weights_intializer['conv3.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv3.0.bias'],trainable=TRAINABLE)))
    #model.layers[7].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3_1'
               ,kernel_initializer = lambda x: weights_intializer['conv3_1.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv3_1.0.bias'],trainable=TRAINABLE)))
    #model.layers[10].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
   
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(512, (3, 3), strides = (2, 2), name = 'conv4' 
               ,kernel_initializer = lambda x: weights_intializer['conv4.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv4.0.bias'],trainable=TRAINABLE)))
    #model.layers[13].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(512, (3, 3), strides = (1, 1), name = 'conv4_1'
               ,kernel_initializer = lambda x: weights_intializer['conv4_1.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv4_1.0.bias'],trainable=TRAINABLE)))    
    #model.layers[16].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(512, (3, 3), strides = (2, 2), name = 'conv5'
               ,kernel_initializer = lambda x: weights_intializer['conv5.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv5.0.bias'],trainable=TRAINABLE)))
    #model.layers[19].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(512, (3, 3), strides = (1, 1), name = 'conv5_1'
               ,kernel_initializer = lambda x: weights_intializer['conv5_1.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv5_1.0.bias'],trainable=TRAINABLE)))
   # model.layers[22].trainable = TRAINABLE
    model.add(TimeDistributed(Activation('relu')))
        
    model.add(TimeDistributed(ZeroPadding2D((1, 1))))
    model.add(TimeDistributed(Conv2D(1024, (3, 3), strides = (2, 2), name = 'conv6'
               ,kernel_initializer = lambda x: weights_intializer['conv6.0.weight'].transpose((3,2,1,0))
               ,bias_initializer=lambda x:weights_intializer['conv6.0.bias'],trainable=TRAINABLE)))
  #  model.layers[25].trainable = TRAINABLE))    
    model.add(TimeDistributed(Dropout(0.80)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_a, return_sequences=True, name="lstm_layer1"))
    model.add(TimeDistributed(Dropout(0.80)))
    model.add(LSTM(n_a, return_sequences=True, name="lstm_layer2"))
    model.add(TimeDistributed(Dropout(0.80)))
    model.add(TimeDistributed(Dense(6)))

    
    ### END CODE HERE ###
    
    return model

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred[:,:,0:3] - y_true[:,:,0:3])+k_param*K.square(y_pred[:,:,3:] - y_true[:,:,3:]), axis=-1)




tf.reset_default_graph()
model = CNNModel((time_sequence,1280,384,6))
print(model.summary())
#Adagrad optimizer
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=mse, optimizer=sgd, metrics=[metrics.mae, metrics.categorical_accuracy])
if load_existing_model:
    model = load_model('vol/my_model'+str(steps)+'.h5')
data_reader = DataReader()
NUM_BATCHES_PER_EPOCH=int(data_reader.num_images/BATCH_SIZE)
NUM_TEST_DATA=int(data_reader.total_test/BATCH_SIZE)
print('Num of batches per epoch :',NUM_BATCHES_PER_EPOCH)

for e in range(epochs):
    print("epoch %d" % e)
    for i in range(NUM_BATCHES_PER_EPOCH):
        steps+=1
        X_batch, Y_batch = data_reader.load_train_batch(BATCH_SIZE)
        model.fit(X_batch, Y_batch, batch_size=BATCH_SIZE, nb_epoch=1)
        if steps% 100 == 0 or steps==NUM_BATCHES_PER_EPOCH*epochs-1:
            model.save('vol/my_model'+str(steps)+'.h5')  # creates a HDF5 file 'my_model.h5'
    score=0
    for j in range(NUM_TEST_DATA):
        x_test, y_test = data_reader.load_test_data(BATCH_SIZE)
        score += model.evaluate(x_test, y_test)/ NUM_TEST_DATA 
    print(print("Testing... error=%g " % (score)))
#weights = model.layers[2].get_weights()
