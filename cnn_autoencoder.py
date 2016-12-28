## convolution network autoencoder
import os                                                                                            
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

def cnn_autoencoder(input):
    
    chanel = input.get_shape()[3]
    init = tf.truncated_normal_initializer(stddev=0.1)
    kernel1 =  tf.get_variable(name='kernel1',shape=[5,5,chanel,8],initializer=init);
    kernel2 =  tf.get_variable(name='kernel2',shape=[3,3,8,8],initializer=init);
    bias1 = tf.Variable(tf.zeros([8]))
    bias2 = tf.Variable(tf.zeros([8]))
    dbias2 = tf.Variable(tf.zeros([8]))
    dbias1 = tf.Variable(tf.zeros([1]))

    conv1 = tf.nn.conv2d(input, kernel1, strides=[1,2,2,1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
#    pool1 = tf.nn.maxpool(relu1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.conv2d(relu1, kernel2, strides=[1,2,2,1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
    
    deconv2 = tf.nn.conv2d_transpose(relu2, kernel2, relu1.get_shape(), strides=[1,2,2,1], padding='SAME')
    derelu2 = tf.nn.relu(tf.nn.bias_add(deconv2,dbias2))
    deconv1 = tf.nn.conv2d_transpose(derelu2, kernel1, input.get_shape(),strides=[1,2,2,1], padding='SAME')
    derelu1 = tf.nn.relu(tf.nn.bias_add(deconv1,dbias1))

    loss = tf.reduce_sum(tf.square(derelu1-input))
    return (derelu1,loss)

def optimizer(loss,lr=0.003):
    return tf.train.AdamOptimizer(lr).minimize(loss)


def train_mnist():
    mnist = get_mnist_data()
    mean_img = np.mean(mnist.train.images, axis=0).reshape([28,28,1])
    batch_size = 64
    epochs = 2
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 28,28,1])
    y, loss =  cnn_autoencoder(x)
    opt = optimizer(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
            train_x = np.array([(img - mean_img) for img in batch])
            y_img,loss_y,_ = sess.run([y,loss,opt], feed_dict = {x:train_x})

        print("epoch: %d ,loss: %f" % (epoch,loss_y))

### test
    test_data = mnist.test.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
    test_x = np.array([(img - mean_img) for img in test_data])
    test_y = sess.run(y,feed_dict={x:test_x})
    show_img_square(test_data, test_y,mean_img)


if __name__ =='__main__':
    train_mnist()
