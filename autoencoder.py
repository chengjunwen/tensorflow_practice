import os
import numpy as np
import math
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

def _get_data():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    return mnist
def get_noise_image(input, noise_ratio):
### denoise
    noise_image = np.random.uniform(-0.5,0.5,
                        (input.get_shape())).astype('float32')
    return noise_ratio*noise_image + (1-noise_ratio)*input
    
    
def autoencoder(input,numOuts=[784,256,64],noise_ratio=0.5):
    
    encoder =[]
    current_x = get_noise_image(input, noise_ratio)

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

def show_img(test_data,test_y, mean_img,nimgs=10):
    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(nimgs):
        axs[0][i].imshow(
            np.reshape(test_data[i, :], (28, 28)))
        axs[1][i].imshow(
            np.reshape([test_y[i, :] + mean_img], (28, 28)))

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

def train_mnist():
    mnist = _get_data()
    mean_img = np.mean(mnist.train.images, axis=0)
    batch_size = 64
    epochs = 2
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    y, loss =  autoencoder(x)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0]
            train_x = np.array([(img - mean_img) for img in batch])
            y_img,loss_y,_ = sess.run([y,loss,optimizer], feed_dict = {x:train_x})

        print("epoch: %d ,loss: %f" % (epoch,loss_y))

### test
    test_data = mnist.test.next_batch(batch_size)[0]
    test_x = np.array([(img - mean_img) for img in test_data])
    test_y = sess.run(y,feed_dict={x:test_x})
    show_img(test_data, test_y,mean_img)
train_mnist()
