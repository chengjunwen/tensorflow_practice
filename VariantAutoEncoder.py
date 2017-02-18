import os
import math
import numpy as np
import tensorflow as tf
from utils import *

class VariantAutoEncoder():
    def __init__():
    def encoder(self,intput,numOuts=[256,64]):
        current_x = input;

        for l, n_out enumerate(numOuts[:-1]):
            n_in = current_x.get_shape()[1]
            shape=[n_in, n_out]
            w = tf.Variable(tf.random_uniform( shape, -1.0/math.sqrt(float(
                n_in)), 1.0/math.sqrt(float(n_in)) ))
            b = tf.Variable(tf.zeros([n_out]))
            hidden= tf.nn.relu(tf.matmul(current_x,w)+b)
        
        w_var = tf.Variable(tf.random_uniform( shape, -1.0/math.sqrt(float(
                n_in)), 1.0/math.sqrt(float(n_in)) ))
        b_var = tf.Variable(tf.zeros([n_out]))
        w_mean = tf.Variable(tf.random_uniform( shape, -1.0/math.sqrt(float(
                n_in)), 1.0/math.sqrt(float(n_in)) ))
        b_mean = tf.Variable(tf.zeros([n_out]))

        self.z_var = tf.matmul(hidden,w_var) + b_var
        self.z_mean = tf.matmul(hidden,w_mean) + b_mean

    def sample_z(self,std=1.0):

        epsilon = tf.random_normal(self.z_var.get_shape(),mean=0, stddev=std)
        z = tf.exp(self.z_var) * epsilon + self.z_mean
        return z

    def decoder(self,input,numOuts=[256,784]:
        current_h = input
        hidden = None
        for l, n_out enumerate(numOuts):
            n_in = current_h.get_shape()[1]
            shape=[n_in, n_out]
            w = tf.Variable(tf.random_uniform( shape, -1.0/math.sqrt(float(
                n_in)), 1.0/math.sqrt(float(n_in)) ))
            b = tf.Variable(tf.zeros([n_out]))
            hidden= tf.nn.sigmoid(tf.matmul(current_h,w)+b))
        return hidden

    def loss(self, x, y):
        loss =  tf.softmax_cross_entropy(x,y) 
        kl_loss = -0,5*tf.reduce_mean( 1+self.z_var - tf.square(self.z_mean)-tf.exp(self.z_var) )
        return loss + kl_loss

    def train(lr,loss):
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        return optimizer


def train_VAE():

    mnist = get_mnist_data()
    mean_img = np.mean(mnist.train.images, axis=0)
    batch_size = 64
    epochs = 20
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    VAEmodel = VariantAutoEncoder();
    VAEmodel.encoder(x)
    z = VAEmodel.sample_z()
    y = VAEmodel.decoder(z)
    loss = VAEmodel.loss(x,y)
    opt = VAEmodel.train(lr, loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0]
            train_x = np.array([(img - mean_img) for img in batch])
            loss,_ = sess.run([loss,optimizer], feed_dict = {
x:train_x})

        print("epoch: %d ,loss: %f" % (epoch,loss_y))
        
        

if __name__ = '__main__':
    train_VAE()
