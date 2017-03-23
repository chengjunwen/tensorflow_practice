import numpy as np
import math
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib
import matplotlib.pyplot as plt

def get_mnist_data():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)                                     
    return mnist
def get_noise_image(input, noise_ratio):
### denoise
    noise_image = np.random.uniform(-0.5,0.5,
                        (input.get_shape())).astype('float32')
    return noise_ratio*noise_image + (1-noise_ratio)*input


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

def show_img_square(test_data,test_y, mean_img,nimgs=10):
    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(nimgs):
        axs[0][i].imshow(
                np.reshape(test_data[i, :,:,:], (28, 28)))
        axs[1][i].imshow(
                np.reshape([test_y[i, :,:,:] + mean_img], (28, 28)))

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

