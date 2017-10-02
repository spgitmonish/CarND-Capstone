import random
import cv2
import os
import numpy as np
import time
import csv
import scipy.misc
import tensorflow as tf
import glob
import rospy

from tensorflow.contrib.layers import flatten

# The fire blocks of the squeezenet model
def fire_module(input, fire_id, channel, s1, e1, e2):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e2: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor
    """

    fire_weights = {'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1])),
                    'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1])),
                    'conv_e_2': tf.Variable(tf.truncated_normal([3, 3, s1, e2]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
                   'conv_e_2': tf.Variable(tf.truncated_normal([e2]))}

    with tf.name_scope(fire_id):
        # Squeeze layer 1x1 convolutions
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.bias_add(output, fire_biases['conv_s_1'])
        squeeze1 = tf.nn.relu(output)

        # Expand 1 layer 1x1 convolutions
        output = tf.nn.conv2d(squeeze1, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        output = tf.nn.bias_add(output, fire_biases['conv_e_1'])
        expand1 = tf.nn.relu(output)

        # Expand 2 layer 3x3 convolutions
        output = tf.nn.conv2d(squeeze1, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        output = tf.nn.bias_add(output, fire_biases['conv_e_1'])
        expand2 = tf.nn.relu(output)

        # NOTE: The 2nd parameter donotes which axis to concatenate on
        result = tf.concat([expand1, expand2], 3, name='concat_e1_e2')
        result = tf.nn.relu(result)

        # Return the result
        return result

# SqueezeNet architecture
def squeeze_net(input, classes):
    """
    SqueezeNet model written in tensorflow. It provides AlexNet level accuracy with 50x fewer parameters
    and smaller model size.
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor
    """

    # Input has 3 channels, output has 96 channels
    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 3, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes])),
               'fc12': tf.Variable(tf.truncated_normal(shape=(1425, classes)))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes])),
              'fc12': tf.Variable(tf.truncated_normal([classes]))}

    # Layer 1: Convolutional with 96 output channels
    output = tf.nn.conv2d(input, weights['conv1'], strides=[1, 2, 2, 1], padding='SAME', name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])
    conv1 = tf.nn.relu(output)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')

    # Layer 2: Fire module with 96 output channels
    output = fire_module(conv1_pool, s1=16, e1=64, e2=64, channel=96, fire_id='fire2')
    fire2 = tf.nn.relu(output)

    # Layer 3: Fire module with 128 output channels
    output = fire_module(fire2, s1=16, e1=64, e2=64, channel=128, fire_id='fire2')
    fire2 = tf.nn.relu(output)

    # Layer 4: Fire module with 128 output channels
    output = fire_module(fire2, s1=32, e1=128, e2=128, channel=128, fire_id='fire4')
    fire4 = tf.nn.relu(output)
    fire4_pool = tf.nn.max_pool(fire4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    # Layer 5: Fire module with 256 output channels
    output = fire_module(fire4_pool, s1=32, e1=128, e2=128, channel=256, fire_id='fire5')
    fire5 = tf.nn.relu(output)

    # Layer 6: Fire module with 256 output channels
    output = fire_module(fire5, s1=48, e1=192, e2=192, channel=256, fire_id='fire6')
    fire6 = tf.nn.relu(output)

    # Layer 7: Fire module with 384 output channels
    output = fire_module(fire6, s1=48, e1=192, e2=192, channel=384, fire_id='fire7')
    fire7 = tf.nn.relu(output)

    # Layer 8: Fire module with 384 output channels
    output = fire_module(fire7, s1=64, e1=256, e2=256, channel=384, fire_id='fire8')
    fire8 = tf.nn.relu(output)
    fire8_pool = tf.nn.max_pool(fire8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    # Layer 9: Fire module with 512 output channels
    output = fire_module(fire8_pool, s1=64, e1=256, e2=256, channel=512, fire_id='fire9')
    fire9 = tf.nn.relu(output)
    fire9_dropout = tf.nn.dropout(fire9, keep_prob=0.5, name='dropout9')

    # Layer 10 : 1x1 convolution
    output = tf.nn.conv2d(fire9_dropout, weights['conv10'], strides=[1, 1, 1, 1], padding='VALID', name='conv10')
    conv10 = tf.nn.bias_add(output, biases['conv10'])
    conv10_pool = tf.nn.avg_pool(conv10, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')

    # Layer 11: Flatten
    flatten11 = flatten(conv10_pool)

    # Layer12: Fully connected layer
    output = tf.matmul(flatten11, weights['fc12']) + biases['fc12']
    fc12 = tf.nn.relu(output)

    # Return the logits
    return fc12

def createModel(num_classes=4, lr = 0.00001, epochs = 100, batch_size = 16):
    ''' Function creates a model object, optimizer, training operation,
        epochs, batch_size and returns it back
    '''
    # Placeholders for training in batches(600(w)x800(h)x3(ch))
    X = tf.placeholder(tf.float32, (None, 600, 800, 3))
    Y = tf.placeholder(tf.int32, (None))

    # One hot encoding
    one_hot_y = tf.one_hot(Y, num_classes)

    # Create the model which returns the last layer output
    model_logits = squeeze_net(X, num_classes)

    # Softmax, cross entropy and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_logits, labels=one_hot_y)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    training_operation = optimizer.minimize(cross_entropy_loss)

    # Accuracy of predictions for testing
    correct_prediction = tf.equal(tf.argmax(model_logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return model_logits, cross_entropy_loss, training_operation, X, Y, accuracy_operation

def inferOnImage(sess, model_logits, X, image):
    # Prediction label
    prediction = tf.argmax(model_logits, 1)

    # Initial array for testing on a new image
    images = np.zeros((1, 600, 800, 3))
    images[0] = image

    # Get the predicted label
    predicted_label = sess.run(prediction, feed_dict={X: images})

    # Return the predicted label
    return predicted_label
