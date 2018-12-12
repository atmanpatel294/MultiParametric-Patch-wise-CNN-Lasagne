from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import random


def function_threeInputs_singlePatch(network,inputs_var1,inputs_var2,inputs_var3):
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([inputs_var1, inputs_var2, inputs_var3, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    #train_fn1 = theano.function([inputs_var1, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    val_fn = theano.function([inputs_var1, inputs_var2, inputs_var3, target_var], [test_loss, test_acc, prediction], on_unused_input='warn')

    return (train_fn,val_fn)

def function_singleInput_twoPatch(network,inputs_var1,inputs_var2):
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([inputs_var1, inputs_var2, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    #train_fn1 = theano.function([inputs_var1, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    val_fn = theano.function([inputs_var1, inputs_var2, target_var], [test_loss, test_acc, prediction], on_unused_input='warn')

    return (train_fn,val_fn)

def functions_for_all(network,inputs_var11,inputs_var12,inputs_var13,inputs_var21,inputs_var22,inputs_var23):
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([inputs_var11,inputs_var12,inputs_var13,inputs_var21,inputs_var22,inputs_var23, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    #train_fn1 = theano.function([inputs_var1, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    val_fn = theano.function([inputs_var11,inputs_var12,inputs_var13,inputs_var21,inputs_var22,inputs_var23, target_var], [test_loss, test_acc, prediction], on_unused_input='warn')

    return (train_fn,val_fn)

def functions_for_singleInput(network, inputs_var1):
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([inputs_var1,target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    #train_fn1 = theano.function([inputs_var1, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    val_fn = theano.function([inputs_var1, target_var], [test_loss, test_acc, prediction], on_unused_input='warn')

    return (train_fn,val_fn)

def function_for_two(network,inputs_var11,inputs_var12,inputs_var21,inputs_var22):
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([inputs_var11,inputs_var12,inputs_var21,inputs_var22, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    #train_fn1 = theano.function([inputs_var1, target_var], [loss, prediction], updates=updates, on_unused_input='warn')
    val_fn = theano.function([inputs_var11,inputs_var12,inputs_var21,inputs_var22, target_var], [test_loss, test_acc, prediction], on_unused_input='warn')

    return (train_fn,val_fn)

def allsize_allchannel(TMAX,tmax):
    # target_var = T.ivector('targets')
    largeInputsTmax = T.tensor4('largeInputs11')
    largeInputsTTP = T.tensor4('largeInputs12')
    largeInputsADC = T.tensor4('largeInputs13')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l11_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsTmax)
    l11_conv1 = lasagne.layers.Conv2DLayer(l11_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool1 = lasagne.layers.MaxPool2DLayer(l11_conv1, pool_size=(2, 2))
    l11_conv2 = lasagne.layers.Conv2DLayer(l11_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool2 = lasagne.layers.MaxPool2DLayer(l11_conv2, pool_size=(2,2))
    l11_conv3 = lasagne.layers.Conv2DLayer(l11_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_conv4 = lasagne.layers.Conv2DLayer(l11_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool3 = lasagne.layers.MaxPool2DLayer(l11_conv4, pool_size=(2, 2))
    l11_conv5 = lasagne.layers.Conv2DLayer(l11_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l12_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsTTP)
    l12_conv1 = lasagne.layers.Conv2DLayer(l12_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool1 = lasagne.layers.MaxPool2DLayer(l12_conv1, pool_size=(2, 2))
    l12_conv2 = lasagne.layers.Conv2DLayer(l12_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool2 = lasagne.layers.MaxPool2DLayer(l12_conv2, pool_size=(2, 2))
    l12_conv3 = lasagne.layers.Conv2DLayer(l12_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_conv4 = lasagne.layers.Conv2DLayer(l12_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool3 = lasagne.layers.MaxPool2DLayer(l12_conv4, pool_size=(2, 2))
    l12_conv5 = lasagne.layers.Conv2DLayer(l12_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l13_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsADC)
    l13_conv1 = lasagne.layers.Conv2DLayer(l13_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l13_pool1 = lasagne.layers.MaxPool2DLayer(l13_conv1, pool_size=(2, 2))
    l13_conv2 = lasagne.layers.Conv2DLayer(l13_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l13_pool2 = lasagne.layers.MaxPool2DLayer(l13_conv2, pool_size=(2, 2))
    l13_conv3 = lasagne.layers.Conv2DLayer(l13_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l13_conv4 = lasagne.layers.Conv2DLayer(l13_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l13_pool3 = lasagne.layers.MaxPool2DLayer(l13_conv4, pool_size=(2, 2))
    l13_conv5 = lasagne.layers.Conv2DLayer(l13_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l1_merge = lasagne.layers.ConcatLayer((l11_conv5, l12_conv5, l13_conv5), axis=1)
    
    smallInputsTmax = T.tensor4('inputs21')
    smallInputsTTP = T.tensor4('inputs22')
    smallInputsADC = T.tensor4('inputs23')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l21_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsTmax)
    l21_conv1 = lasagne.layers.Conv2DLayer(l21_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l21_conv12 = lasagne.layers.Conv2DLayer(l21_conv1, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify)
    # l21_pool1 = lasagne.layers.MaxPool2DLayer(l21_conv1, pool_size=(2, 2))
    l21_conv2 = lasagne.layers.Conv2DLayer(l21_conv12, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l21_conv3 = lasagne.layers.Conv2DLayer(l21_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l22_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsTTP)
    l22_conv1 = lasagne.layers.Conv2DLayer(l22_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l22_conv12 = lasagne.layers.Conv2DLayer(l22_conv1, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify)
    # l22_pool1 = lasagne.layers.MaxPool2DLayer(l22_conv1, pool_size=(2, 2))
    l22_conv2 = lasagne.layers.Conv2DLayer(l22_conv12, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l22_conv3 = lasagne.layers.Conv2DLayer(l22_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l23_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsADC)
    l23_conv1 = lasagne.layers.Conv2DLayer(l23_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l23_conv12 = lasagne.layers.Conv2DLayer(l23_conv1, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify)
    # l23_pool1 = lasagne.layers.MaxPool2DLayer(l23_conv1, pool_size=(2, 2))
    l23_conv2 = lasagne.layers.Conv2DLayer(l23_conv12, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l23_conv3 = lasagne.layers.Conv2DLayer(l23_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l2_merge = lasagne.layers.ConcatLayer((l21_conv3, l22_conv3, l23_conv3), axis=1)

    l1_dense1 = lasagne.layers.DenseLayer(l1_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l2_dense1 = lasagne.layers.DenseLayer(l2_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l_merge = lasagne.layers.ConcatLayer((l1_dense1, l2_dense1), axis=1)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_merge, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,largeInputsTmax,largeInputsTTP,largeInputsADC,smallInputsTmax, smallInputsTTP, smallInputsADC)

def three_layer_three_channel1(TMAX,tmax):
    # target_var = T.ivector('targets')
    inputs_var1 = T.tensor4('inputs1')
    inputs_var2 = T.tensor4('inputs2')
    inputs_var3 = T.tensor4('inputs3')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l1_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l1_conv1 = lasagne.layers.Conv2DLayer(l1_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool1 = lasagne.layers.MaxPool2DLayer(l1_conv1, pool_size=(2, 2))
    l1_conv2 = lasagne.layers.Conv2DLayer(l1_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool2 = lasagne.layers.MaxPool2DLayer(l1_conv2, pool_size=(2,2))
    l1_conv3 = lasagne.layers.Conv2DLayer(l1_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv4 = lasagne.layers.Conv2DLayer(l1_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool3 = lasagne.layers.MaxPool2DLayer(l1_conv4, pool_size=(2, 2))
    l1_conv5 = lasagne.layers.Conv2DLayer(l1_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l2_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l2_conv1 = lasagne.layers.Conv2DLayer(l2_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l2_pool1 = lasagne.layers.MaxPool2DLayer(l2_conv1, pool_size=(2, 2))
    l2_conv2 = lasagne.layers.Conv2DLayer(l2_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l2_pool2 = lasagne.layers.MaxPool2DLayer(l2_conv2, pool_size=(2, 2))
    l2_conv3 = lasagne.layers.Conv2DLayer(l2_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l2_conv4 = lasagne.layers.Conv2DLayer(l2_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l2_pool3 = lasagne.layers.MaxPool2DLayer(l2_conv4, pool_size=(2, 2))
    l2_conv5 = lasagne.layers.Conv2DLayer(l2_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l3_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l3_conv1 = lasagne.layers.Conv2DLayer(l3_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l3_pool1 = lasagne.layers.MaxPool2DLayer(l3_conv1, pool_size=(2, 2))
    l3_conv2 = lasagne.layers.Conv2DLayer(l3_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l3_pool2 = lasagne.layers.MaxPool2DLayer(l3_conv2, pool_size=(2, 2))
    l3_conv3 = lasagne.layers.Conv2DLayer(l3_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l3_conv4 = lasagne.layers.Conv2DLayer(l3_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l3_pool3 = lasagne.layers.MaxPool2DLayer(l3_conv4, pool_size=(2, 2))
    l3_conv5 = lasagne.layers.Conv2DLayer(l3_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l_merge = lasagne.layers.ConcatLayer((l1_conv5, l2_conv5, l3_conv5), axis=1)

    l1_dense1 = lasagne.layers.DenseLayer(l_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,inputs_var1,inputs_var2,inputs_var3)

def three_layer_three_channel2(TMAX,tmax):
    # target_var = T.ivector('targets')
    inputs_var1 = T.tensor4('inputs1')
    inputs_var2 = T.tensor4('inputs2')
    inputs_var3 = T.tensor4('inputs3')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l1_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l1_conv1 = lasagne.layers.Conv2DLayer(l1_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l1_pool1 = lasagne.layers.MaxPool2DLayer(l1_conv1, pool_size=(2, 2))
    l1_conv2 = lasagne.layers.Conv2DLayer(l1_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv3 = lasagne.layers.Conv2DLayer(l1_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l2_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var2)
    l2_conv1 = lasagne.layers.Conv2DLayer(l2_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l2_pool1 = lasagne.layers.MaxPool2DLayer(l2_conv1, pool_size=(2, 2))
    l2_conv2 = lasagne.layers.Conv2DLayer(l2_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l2_conv3 = lasagne.layers.Conv2DLayer(l2_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l3_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var3)
    l3_conv1 = lasagne.layers.Conv2DLayer(l3_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l3_pool1 = lasagne.layers.MaxPool2DLayer(l3_conv1, pool_size=(2, 2))
    l3_conv2 = lasagne.layers.Conv2DLayer(l3_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l3_conv3 = lasagne.layers.Conv2DLayer(l3_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l_merge = lasagne.layers.ConcatLayer((l1_conv3, l2_conv3, l3_conv3), axis=1)

    l1_dense1 = lasagne.layers.DenseLayer(l_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,inputs_var1,inputs_var2,inputs_var3)

def three_layer_single_channel1(TMAX,tmax):
    # target_var = T.ivector('targets')
    inputs_var1 = T.tensor4('inputs1')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l1_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l1_conv1 = lasagne.layers.Conv2DLayer(l1_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool1 = lasagne.layers.MaxPool2DLayer(l1_conv1, pool_size=(2, 2))
    l1_conv2 = lasagne.layers.Conv2DLayer(l1_pool1, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool2 = lasagne.layers.MaxPool2DLayer(l1_conv2, pool_size=(2,2))
    l1_conv3 = lasagne.layers.Conv2DLayer(l1_pool2, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv4 = lasagne.layers.Conv2DLayer(l1_conv3, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv5 = lasagne.layers.Conv2DLayer(l1_conv4, num_filters=1024, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l1_dense1 = lasagne.layers.DenseLayer(l1_conv5, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=1024, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,inputs_var1)

def three_layer_single_channel2(TMAX,tmax):
    # target_var = T.ivector('targets')
    inputs_var1 = T.tensor4('inputs1')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l1_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l1_conv1 = lasagne.layers.Conv2DLayer(l1_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv2 = lasagne.layers.Conv2DLayer(l1_conv1, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv3 = lasagne.layers.Conv2DLayer(l1_conv2, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv4 = lasagne.layers.Conv2DLayer(l1_conv3, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv5 = lasagne.layers.Conv2DLayer(l1_conv4, num_filters=1024, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l1_dense1 = lasagne.layers.DenseLayer(l1_conv5, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=1024, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,inputs_var1)

def network_singleInput_twoPatch(TMAX,tmax):
    # target_var = T.ivector('targets')
    largeInputsTmax = T.tensor4('largeInputs11')
    smallInputsTmax = T.tensor4('smallInputs21')
    
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l11_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsTmax)
    l11_conv1 = lasagne.layers.Conv2DLayer(l11_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool1 = lasagne.layers.MaxPool2DLayer(l11_conv1, pool_size=(2, 2))
    l11_conv2 = lasagne.layers.Conv2DLayer(l11_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool2 = lasagne.layers.MaxPool2DLayer(l11_conv2, pool_size=(2,2))
    l11_conv3 = lasagne.layers.Conv2DLayer(l11_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_conv4 = lasagne.layers.Conv2DLayer(l11_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool3 = lasagne.layers.MaxPool2DLayer(l11_conv4, pool_size=(2, 2))
    l11_conv5 = lasagne.layers.Conv2DLayer(l11_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)    

    l21_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsTmax)
    l21_conv1 = lasagne.layers.Conv2DLayer(l21_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l21_pool1 = lasagne.layers.MaxPool2DLayer(l21_conv1, pool_size=(2, 2))
    l21_conv2 = lasagne.layers.Conv2DLayer(l21_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l21_conv3 = lasagne.layers.Conv2DLayer(l21_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l1_dense1 = lasagne.layers.DenseLayer(l11_conv5, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l2_dense1 = lasagne.layers.DenseLayer(l21_conv3, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l_merge = lasagne.layers.ConcatLayer((l1_dense1, l2_dense1), axis=1)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_merge, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,largeInputsTmax,smallInputsTmax)

def twoSize_twoChannel(TMAX,tmax):
    # target_var = T.ivector('targets')
    largeInputsTmax = T.tensor4('largeInputs11')
    largeInputsTTP = T.tensor4('largeInputs12')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l11_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsTmax)
    l11_conv1 = lasagne.layers.Conv2DLayer(l11_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool1 = lasagne.layers.MaxPool2DLayer(l11_conv1, pool_size=(2, 2))
    l11_conv2 = lasagne.layers.Conv2DLayer(l11_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool2 = lasagne.layers.MaxPool2DLayer(l11_conv2, pool_size=(2,2))
    l11_conv3 = lasagne.layers.Conv2DLayer(l11_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l11_conv4 = lasagne.layers.Conv2DLayer(l11_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l11_pool3 = lasagne.layers.MaxPool2DLayer(l11_conv4, pool_size=(2, 2))
    l11_conv5 = lasagne.layers.Conv2DLayer(l11_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l12_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=largeInputsTTP)
    l12_conv1 = lasagne.layers.Conv2DLayer(l12_in1, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool1 = lasagne.layers.MaxPool2DLayer(l12_conv1, pool_size=(2, 2))
    l12_conv2 = lasagne.layers.Conv2DLayer(l12_pool1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool2 = lasagne.layers.MaxPool2DLayer(l12_conv2, pool_size=(2, 2))
    l12_conv3 = lasagne.layers.Conv2DLayer(l12_pool2, num_filters=64, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)
    l12_conv4 = lasagne.layers.Conv2DLayer(l12_conv3, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l12_pool3 = lasagne.layers.MaxPool2DLayer(l12_conv4, pool_size=(2, 2))
    l12_conv5 = lasagne.layers.Conv2DLayer(l12_pool3, num_filters=512, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    l1_merge = lasagne.layers.ConcatLayer((l11_conv5, l12_conv5), axis=1)
    
    smallInputsTmax = T.tensor4('inputs21')
    smallInputsTTP = T.tensor4('inputs22')


    l21_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsTmax)
    l21_conv1 = lasagne.layers.Conv2DLayer(l21_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l21_pool1 = lasagne.layers.MaxPool2DLayer(l21_conv1, pool_size=(2, 2))
    l21_conv2 = lasagne.layers.Conv2DLayer(l21_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l21_conv3 = lasagne.layers.Conv2DLayer(l21_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l22_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatch_width, smallPatch_width), input_var=smallInputsTTP)
    l22_conv1 = lasagne.layers.Conv2DLayer(l22_in1, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l22_pool1 = lasagne.layers.MaxPool2DLayer(l22_conv1, pool_size=(2, 2))
    l22_conv2 = lasagne.layers.Conv2DLayer(l22_pool1, num_filters=128, filter_size=(3, 1), nonlinearity=lasagne.nonlinearities.rectify)
    l22_conv3 = lasagne.layers.Conv2DLayer(l22_conv2, num_filters=512, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify)

    l2_merge = lasagne.layers.ConcatLayer((l21_conv3, l22_conv3), axis=1)

    l1_dense1 = lasagne.layers.DenseLayer(l1_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l2_dense1 = lasagne.layers.DenseLayer(l2_merge, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    l_merge = lasagne.layers.ConcatLayer((l1_dense1, l2_dense1), axis=1)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_merge, p=.5), num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense3, p=.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return (network,largeInputsTmax,largeInputsTTP,smallInputsTmax, smallInputsTTP)



def inceptionV4(TMAX, tmax):
    target_var = T.ivector('targets')
    largeInputsTmax = T.tensor4('inputs11')
    largeInputsTTP = T.tensor4('inputs21')
    largeInputsADC = T.tensor4('inputs31')
    smallInputsTmax = T.tensor4('inputs12')
    smallInputsTTP = T.tensor4('inputs22')
    smallInputsADC = T.tensor4('inputs32')
    num_patches, num_channels, largePatchHeight, largePatchWidth = TMAX.shape
    num_patches, num_channels, smallPatchHeight, smallPatchWidth = tmax.shape
    # 49
    l11_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatchHeight, largePatchWidth), input_var=largeInputsTmax)
    l11_conv1 = lasagne.layers.Conv2DLayer(l11_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l11_conv2 = lasagne.layers.Conv2DLayer(l11_conv1, num_filters=32, filter_size=(3, 3),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    l21_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatchHeight, largePatchWidth), input_var=largeInputsTTP)
    l21_conv1 = lasagne.layers.Conv2DLayer(l21_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l21_conv2 = lasagne.layers.Conv2DLayer(l21_conv1, num_filters=64, filter_size=(3, 3),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    l31_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatchHeight, largePatchWidth), input_var=largeInputsADC)
    l31_conv1 = lasagne.layers.Conv2DLayer(l31_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l31_conv2 = lasagne.layers.Conv2DLayer(l31_conv1, num_filters=64, filter_size=(3, 3),
                                           nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    l1_merge1 = lasagne.layers.ConcatLayer((l11_conv2, l21_conv2, l31_conv2), axis=1)
    # 22
    l1_conv3 = lasagne.layers.Conv2DLayer(l1_merge1, num_filters=96, filter_size=(3, 3),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv4 = lasagne.layers.Conv2DLayer(l1_conv3, num_filters=172, filter_size=(3, 3), stride=(2, 2),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool4 = lasagne.layers.MaxPool2DLayer(l1_conv3, pool_size=(3, 3), stride=(2, 2))
    # 10
    l1_merge2 = lasagne.layers.ConcatLayer((l1_conv4, l1_pool4), axis=1)
    # 10
    l1_conv51 = lasagne.layers.Conv2DLayer(l1_merge2, num_filters=172, filter_size=(1, 1),
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv52 = lasagne.layers.Conv2DLayer(l1_conv51, num_filters=172, filter_size=(3, 1), pad="same",
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv53 = lasagne.layers.Conv2DLayer(l1_conv52, num_filters=172, filter_size=(1, 3), pad="same",
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv54 = lasagne.layers.Conv2DLayer(l1_conv53, num_filters=172, filter_size=(3, 3),
                                          nonlinearity=lasagne.nonlinearities.rectify)

    l1_conv61 = lasagne.layers.Conv2DLayer(l1_merge2, num_filters=172, filter_size=(1, 1), pad="same",
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_conv62 = lasagne.layers.Conv2DLayer(l1_conv61, num_filters=172, filter_size=(3, 3),
                                          nonlinearity=lasagne.nonlinearities.rectify)
    # 8

    l1_merge3 = lasagne.layers.ConcatLayer((l1_conv62, l1_conv54), axis=1)

    l1_conv7 = lasagne.layers.Conv2DLayer(l1_merge3, num_filters=360, filter_size=(3, 3),
                                         nonlinearity=lasagne.nonlinearities.rectify)

    l1_dense1 = lasagne.layers.DenseLayer(l1_conv7, num_units=360, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=128,
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=2,
                                          nonlinearity=lasagne.nonlinearities.rectify)
    
#     l12_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatchHeight, smallPatchHeight), input_var=smallInputsTmax)
#     l12_conv1 = lasagne.layers.Conv2DLayer(l12_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#     l12_conv2 = lasagne.layers.Conv2DLayer(l12_conv1, num_filters=32, filter_size=(3, 3),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

#     l22_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatchHeight, smallPatchHeight), input_var=smallInputsTTP)
#     l22_conv1 = lasagne.layers.Conv2DLayer(l22_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#     l22_conv2 = lasagne.layers.Conv2DLayer(l22_conv1, num_filters=64, filter_size=(3, 3),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

#     l32_in1 = lasagne.layers.InputLayer(shape=(None, 1, smallPatchHeight, smallPatchHeight), input_var=smallInputsADC)
#     l32_conv1 = lasagne.layers.Conv2DLayer(l32_in1, num_filters=32, filter_size=(3, 3), stride=(2, 2),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#     l32_conv2 = lasagne.layers.Conv2DLayer(l32_conv1, num_filters=64, filter_size=(3, 3),
#                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

#     l2_merge1 = lasagne.layers.ConcatLayer((l12_conv2, l22_conv2, l32_conv2), axis=1)
#     # 22
#     l2_conv3 = lasagne.layers.Conv2DLayer(l2_merge1, num_filters=96, filter_size=(3, 3),
#                                          nonlinearity=lasagne.nonlinearities.rectify)
#     l2_conv4 = lasagne.layers.Conv2DLayer(l2_conv3, num_filters=172, filter_size=(3, 3), stride=(2, 2),
#                                          nonlinearity=lasagne.nonlinearities.rectify)
#     l2_pool4 = lasagne.layers.MaxPool2DLayer(l2_conv3, pool_size=(3, 3), stride=(2, 2))
#     # 10
#     l2_merge2 = lasagne.layers.ConcatLayer((l2_conv4, l2_pool4), axis=1)
#     # 10
#     l2_conv51 = lasagne.layers.Conv2DLayer(l2_merge2, num_filters=172, filter_size=(1, 1),
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l2_conv52 = lasagne.layers.Conv2DLayer(l2_conv51, num_filters=172, filter_size=(3, 1), pad="same",
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l2_conv53 = lasagne.layers.Conv2DLayer(l2_conv52, num_filters=172, filter_size=(1, 3), pad="same",
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l2_conv54 = lasagne.layers.Conv2DLayer(l2_conv53, num_filters=172, filter_size=(3, 3),
#                                           nonlinearity=lasagne.nonlinearities.rectify)

#     l2_conv61 = lasagne.layers.Conv2DLayer(l2_merge2, num_filters=172, filter_size=(1, 1), pad="same",
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l2_conv62 = lasagne.layers.Conv2DLayer(l2_conv61, num_filters=172, filter_size=(3, 3),
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     # 8

#     l2_merge3 = lasagne.layers.ConcatLayer((l2_conv62, l2_conv54), axis=1)

#     l2_conv7 = lasagne.layers.Conv2DLayer(l2_merge3, num_filters=360, filter_size=(3, 3),
#                                          nonlinearity=lasagne.nonlinearities.rectify)

#     l2_dense1 = lasagne.layers.DenseLayer(l2_conv7, num_units=360, nonlinearity=lasagne.nonlinearities.rectify)
#     l2_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l2_dense1, p=.5), num_units=128,
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l2_dense3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l2_dense2, p=.5), num_units=2,
#                                           nonlinearity=lasagne.nonlinearities.rectify)
#     l_merge = lasagne.layers.ConcatLayer((l2_dense3, l1_dense3), axis=1)
    
#     network = lasagne.layers.DenseLayer(l_merge, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    
    return l1_dense3, largeInputsTmax,largeInputsTTP,largeInputsADC#, smallInputsTmax, smallInputsTTP, smallInputsADC 