## Import the basic libs
import scipy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os
import numpy as np

## Import libs required for creating a keras CNN
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
from keras.optimizers import Adam



## Model
# input_height = height of the input images
# input_width = width of the input images
# channels = numbe of inputs
def myModel(input_height=256, input_width=256, channels=2):
    # We'll use 2 channels in our case
    img_input = Input(shape=(input_height,input_width, channels))
    nClasses = 2
    filters = 8 #defines the humber of filters in the first layer of network
    IMAGE_ORDERING =  "channels_last"
	
    ## Block 1 - (256,256,2)
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
    
    # (4,4,512)
    n = 4096
    conv7 = ( Conv2D( int(n/4), ( 4 , 4 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(x)
    conv8 = ( Conv2D( n*2 , ( 1 , 1 ) , activation='relu' , padding='same', name="conv8", data_format=IMAGE_ORDERING))(conv7)
    
	# Now we deconvolute
	
    conv8_4 = Conv2DTranspose( nClasses ,kernel_size=(8,8), strides=(8,8) ,use_bias=False, name='deconv1', data_format=IMAGE_ORDERING )(conv8)
    
    pool411 = (Conv2D( nClasses, (1,1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING ))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False, name='deconv2', data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv8_4])
    
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, name='deconv3', data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)
    return model

    
#define channels/no of inputs here
model = myModel(channels=2)
print(model.summary())


## Loading Data
# import preprocessing notebook
get_ipython().magic(u'run Preprocessing.ipynb')
#loadAllSlices is a function in Preprocessing
Data = loadAllSlices(os.getcwd()+'/images/all_slices', False)


print(Data.shape)
from keras.utils import to_categorical
partition = 400
X_train = Data[:partition,:,:,[0,1]]				#training input
X_test = Data[partition:,:,:,[0,1]]					#test/val input
y_train = to_categorical(Data[:partition,:,:,2:]) 	#using [2:] to maintain the shape of array
y_test = to_categorical(Data[partition:,:,:,2:])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



from keras import optimizers
sgd = optimizers.SGD(lr=0.03, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



hist1 = model.fit(X_train,y_train,
                  validation_data=(X_train,y_train),
                  batch_size=32,epochs=20,verbose=2)



model.save(path_to_model_repo)



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


y_predict



def plotOneImage(pred_sample):
    
    X_predict = Data[pred_sample:pred_sample+1,:,:,4:5] #using ranges to maintain shape
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
    
    plt.savefig(path_to_results + '/{}_epoch100.png'.format(pred_sample), bbox_inches='tight')
    plt.close()

case_num = np.arange(partition,502)
for i in case_num:
    plotOneImage(i)
    print("case no. {0} saved".format(i))

