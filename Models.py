from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import random


def functions(network,inputs_var1,inputs_var2,inputs_var3):
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

    train_fn = theano.function([inputs_var1, inputs_var2, inputs_var3, target_var], [loss, prediction], updates=updates,
                               on_unused_input='warn')
    val_fn = theano.function([inputs_var1, inputs_var2, inputs_var3, target_var], [test_loss, test_acc, prediction],
                             on_unused_input='warn')
    return (train_fn,val_fn)

def three_layer_three_channel(TMAX,tmax):
    # target_var = T.ivector('targets')
    inputs_var1 = T.tensor4('inputs1')
    inputs_var2 = T.tensor4('inputs2')
    inputs_var3 = T.tensor4('inputs3')
    num_patches, num_channels, largePatch_height, largePatch_width = TMAX.shape
    num_patches, num_channels, smallPatch_height, smallPatch_width = tmax.shape

    l1_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var1)
    l1_conv1 = lasagne.layers.Conv2DLayer(l1_in1, num_filters=32, filter_size=(5, 5),
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l1_pool1 = lasagne.layers.MaxPool2DLayer(l1_conv1, pool_size=(2, 2))
    l1_conv2 = lasagne.layers.Conv2DLayer(l1_pool1, num_filters=32, filter_size=(3, 3),
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l1_pool2 = lasagne.layers.MaxPool2DLayer(l1_conv2, pool_size=(2, 2))

    l2_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var2)
    l2_conv1 = lasagne.layers.Conv2DLayer(l2_in1, num_filters=32, filter_size=(5, 5),
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l2_pool1 = lasagne.layers.MaxPool2DLayer(l2_conv1, pool_size=(2, 2))
    l2_conv2 = lasagne.layers.Conv2DLayer(l2_pool1, num_filters=32, filter_size=(3, 3),
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l2_pool2 = lasagne.layers.MaxPool2DLayer(l2_conv2, pool_size=(2, 2))

    l3_in1 = lasagne.layers.InputLayer(shape=(None, 1, largePatch_height, largePatch_width), input_var=inputs_var3)
    l3_conv1 = lasagne.layers.Conv2DLayer(l3_in1, num_filters=32, filter_size=(5, 5),
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l3_pool1 = lasagne.layers.MaxPool2DLayer(l3_conv1, pool_size=(2, 2))
    l3_conv2 = lasagne.layers.Conv2DLayer(l3_pool1, num_filters=32, filter_size=(3, 3),
                                          nonlinearity=lasagne.nonlinearities.rectify)
    l3_pool2 = lasagne.layers.MaxPool2DLayer(l3_conv2, pool_size=(2, 2))

    l_merge = lasagne.layers.ConcatLayer((l1_pool2, l2_pool2, l3_pool2), axis=1)

    l1_dense1 = lasagne.layers.DenseLayer(l_merge, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    l1_dense2 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense1, p=.5), num_units=50,
                                          nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(l1_dense2, p=.5), num_units=2,
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return (network,inputs_var1,inputs_var2,inputs_var3)

