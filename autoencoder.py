import os
import numpy as np
import math
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

def _get_data():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    return mnist
def autoencoder(input,numOuts=[784,256,64]):
    
    encoder =[]
    current_x = input
    for l, n_out in enumerate(numOuts[1:]):
        n_in = int(current_x.get_shape()[1]);
        shape = [n_in, n_out]
        w = tf.Variable(tf.random_uniform( shape, -1.0/math.sqrt(float(n_in)), 1.0/math.sqrt(float(n_in)) ))
        b_h = tf.Variable(tf.zeros([n_out]))
        encoder.append(w)
        h = tf.nn.tanh(tf.matmul(current_x, w) + b_h)
        current_x=h

    encoder.reverse()
    numOuts.reverse();
    current_h = current_x
    
    for l, n_out in enumerate(numOuts[1:]):
        n_in = current_h.get_shape()[1];
        shape = [n_in, n_out]
        w_t = tf.transpose(encoder[l])
        b_y = tf.Variable(tf.zeros([n_out]))
        y = tf.matmul(current_h, w_t)
        y = tf.nn.tanh(y+ b_y)
        current_h = y

    y = current_h
    loss = tf.reduce_sum(tf.square(y-input))

    return (y,loss)

def train_mnist():
    mnist = _get_data()
    mean_img = np.mean(mnist.train.images, axis=0)

    batch_size = 64
    epochs = 50
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    y, loss =  autoencoder(x)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0]
            train_x = np.array([img - mean_img for img in batch])
            y_img,loss_y,_ = sess.run([y,loss,optimizer], feed_dict = {x:train_x})

        print("epoch: %d ,loss: %f" % (epoch,loss_y))

       
train_mnist()
