
# coding: utf-8

# In[1]:

# from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Add, MaxPooling2D, Conv2DTranspose
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import scipy
from keras.optimizers import Adam


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os
import numpy as np



# # Model

# In[73]:

def myModel(input_height=256, input_width=256, channels=1):
    
    img_input = Input(shape=(input_height,input_width, channels))
    nClasses = 2
    filters = 8
    IMAGE_ORDERING =  "channels_last" 
    ## Block 1 - (256,256,1)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x 
    
    # Block 2 - (128,128,16)
    x = Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3 - (64,64,32)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4 - (32,32,64)
    x = Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5 - (16,16,128)
    x = Conv2D(filters*16, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(filters*16, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*16, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)
    
    # (8,8,256)
    x = Conv2D(filters*32, (5, 5), activation='relu', padding='valid', name='block6_conv1', data_format=IMAGE_ORDERING )(pool5)
#     print("block6_conv1 : {}".format(x._keras_shape))
    
    # (4,4,512)
    n = 4096
    conv7 = ( Conv2D( int(n/4), ( 4 , 4 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(x)
    conv8 = ( Conv2D( n*2 , ( 1 , 1 ) , activation='relu' , padding='same', name="conv8", data_format=IMAGE_ORDERING))(conv7)
#     print("conv8 : {}".format(conv8._keras_shape))
    
    conv8_4 = Conv2DTranspose( nClasses ,kernel_size=(8,8), strides=(8,8) ,use_bias=False, name='deconv1', data_format=IMAGE_ORDERING )(conv8)
#     conv8_4 = Conv2DTranspose( nClasses ,kernel_size=(2,2), strides=(2,2) ,use_bias=False, data_format=IMAGE_ORDERING )(conv8_4)
#     print("conv8_4 : {}".format(conv8_4._keras_shape))
    

    pool411 = (Conv2D( nClasses, (1,1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING ))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False, name='deconv2', data_format=IMAGE_ORDERING ))(pool411)
#     print("pool411_2 : {}".format(pool411_2._keras_shape))
    
    pool311 = ( Conv2D( nClasses , (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
#     print("pool311 : {}".format(pool311._keras_shape))  
        
    o = Add(name="add")([pool411_2, pool311, conv8_4])
    
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, name='deconv3', data_format=IMAGE_ORDERING )(o)
#     o = Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)
    return model

    
    
model = myModel(channels=2)
model.summary()


# # Loading Data

# In[37]:

# import preprocessing notebok
get_ipython().magic(u'run Preprocessing.ipynb')


# In[38]:

Data = loadAllSlices(os.getcwd()+'/images/all_slices', False)
Data.shape


# In[96]:

Data[:,:,:,4].max()


# In[39]:

fig = plt.figure()
for i in range(1,8):
    a = fig.add_subplot(3, 3, i)
    imgplot = plt.imshow(Data[5,:,:,i-1])


# In[74]:

print(Data.shape)
from keras.utils import to_categorical
train_to = 400
val_from = 400
X_train = Data[:train_to,:,:,[2,4]]
X_test = Data[val_from:,:,:,[2,4]]
y_train = to_categorical(Data[:train_to,:,:,6:])
y_test = to_categorical(Data[val_from:,:,:,6:])
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[75]:

from keras import optimizers


sgd = optimizers.SGD(lr=0.03, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[76]:

hist1 = model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=32,epochs=20,verbose=2)


# In[14]:

model.save(os.getcwd() + '/Outputs_models/FCN3/100epochs.h5')


# In[6]:

from keras.models import load_model

model = load_model(os.getcwd() + '/weights/FCN1.h5')


# In[79]:

#print the results

pred_sample = 420
X_predict = Data[pred_sample:pred_sample+1,:,:,[2,4]]
y_actual = Data[pred_sample,:,:,6]
y_predict = model.predict(X_predict)


fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(y_actual)
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(y_predict[0,:,:,1])


# In[35]:

y_predict


# In[16]:

def plotOneImage(pred_sample):
    
    X_predict = Data[pred_sample:pred_sample+1,:,:,4:5]
    y_actual = Data[pred_sample,:,:,6]
    y_predict = model.predict(X_predict)
    y_predict = y_predict[0,:,:,1]
    
    #plot predictions
    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    ax1 = fig.add_subplot(1,2,1)
    ax1.title.set_text('Prediction')
    imgplot = plt.imshow(y_predict)
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.title.set_text('Actual')
    imgplot = plt.imshow(y_actual)
    
    plt.savefig(os.getcwd()+'/Outputs_models/FCN3/{}_epoch100.png'.format(pred_sample), bbox_inches='tight')
    plt.close()

case_num = np.arange(400,502)
for i in case_num:
    plotOneImage(i)
    print("case no. {0} saved".format(i))


# In[ ]:




# In[36]:

def FCN8( nClasses ,  input_height=256, input_width=256):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 1)) ## Assume 224,224,3
    filters = 16
    ## Block 1
    x = Conv2D(filters, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)
    
    
#     vgg  = Model(  img_input , pool5  )
#     vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    n = 1024
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)

    return model

model = FCN8(nClasses     = 2,  
             input_height = 256, 
             input_width  = 256)
model.summary()


# In[ ]:



