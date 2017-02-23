import os
import math
import numpy as np
import tensorflow as tf
from utils import *

class VAEModel:

    def __init__(self):
        pass

    def encoder(self, input,numOuts=[256,64]):
        current_x = input
    #        current_x = get_noise_image(input, 0.5)
        for l, n_out in enumerate(numOuts[:-1]):
            n_in = int(current_x.get_shape()[1])
            shape=[n_in, n_out]
            w = tf.Variable(tf.truncated_normal( shape, stddev=0.001))
            b = tf.Variable(tf.zeros([n_out]))
            hidden= tf.nn.relu(tf.matmul(current_x,w)+b)
            current_x = hidden
        
        shape = [n_out, numOuts[-1]]
        w_var = tf.Variable(tf.truncated_normal( shape, stddev = 0.001))
        b_var = tf.Variable(tf.zeros([numOuts[-1]]))
        w_mean = tf.Variable(tf.truncated_normal( shape, stddev=0.001))
        b_mean = tf.Variable(tf.zeros([numOuts[-1]]))

        z_var = tf.matmul(hidden,w_var) + b_var
        z_mean = tf.matmul(hidden,w_mean) + b_mean
        return (z_mean, z_var)

    def sample_z(self, z_mean, z_var, std=1.0):

        epsilon = tf.random_normal(z_var.get_shape(),mean=0, stddev=std)
        z = tf.mul(tf.exp(0.5*z_var) , epsilon) + z_mean
        return z

    def decoder(self, input,numOuts=[256,784]):
        current_h = input
        for l, n_out in enumerate(numOuts):
            n_in = int(current_h.get_shape()[1])
            shape=[n_in, n_out]
            w = tf.Variable(tf.truncated_normal( shape, stddev=0.001 ))
            b = tf.Variable(tf.zeros([n_out]))
            if(l == (len(numOuts)-1)):
                hidden= tf.matmul(current_h,w)+b
            else:
                hidden= tf.nn.relu(tf.matmul(current_h,w)+b)
            current_h = hidden
        return hidden

    def loss(self, x, y, z_mean,z_var):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(x,y) 
#        entropy = tf.square(x-y)
        loss =  tf.reduce_sum(entropy,1)
        kl_loss = -0.5*tf.reduce_mean(1+z_var - tf.square(z_mean)-tf.exp(z_var) ,1)
        all_loss = tf.reduce_mean(loss + kl_loss)
        return all_loss

    def train(self,lr,loss):
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        return optimizer


def train_VAE():

    mnist = get_mnist_data()
    mean_img = np.mean(mnist.train.images, axis=0)
    batch_size = 64
    epochs = 5
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    print(x.get_shape())
    vae = VAEModel()
    mean,var = vae.encoder(x)
    z = vae.sample_z(mean, var)
    y = vae.decoder(z)
    loss= vae.loss(x,y,mean,var)
    print("=====")
    print(loss)
    opt = vae.train(lr, loss)
#    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0]
            train_x = np.array([(img - mean_img) for img in batch])
            feed_dict = {x:train_x}
            y_loss,_ = sess.run([loss,opt], feed_dict = feed_dict)
        print("epoch: %d ,loss: %f" % (epoch,y_loss))
    test_data = mnist.test.next_batch(batch_size)[0]
    test_x = np.array([(img - mean_img) for img in test_data])
    test_y = sess.run(y,feed_dict={x:test_x})
    show_img(test_data, test_y,mean_img)
 
        

if __name__=='__main__':
    train_VAE()
