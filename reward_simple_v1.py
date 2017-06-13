#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE

    s = 1/(1+np.exp(-x))

    ### END YOUR CODE

    return s

def generate_batch(batch_size, option, sd_range):


    if (option == 1):


        data_x = []
        data_y = []

        for i in range(batch_size):
            samp1 = np.random.random_integers(-10, 10)
            samp2 = np.random.random_integers(-10, 10)

            #print samp1
            #print samp2 

            data_x.append((samp1, samp2))
            plt.scatter(samp1, samp2)
            label = np.int32(samp1 + samp2 > 0)
            data_y.append(label)


        batch_x = data_x
        batch_y =  np.int32(data_y)
        print batch_x
        return batch_x, batch_y

    else:
        data_x = []
        data_y = []

        for j in range(batch_size):

            rand_selector = random.random()
            if (rand_selector < 0.5):
                mu1 = 5
                mu2 = 5
            else:
                mu1 = -5
                mu2 = -5

            var1 = np.random.random_integers(-1*sd_range, sd_range)
            var2 = np.random.random_integers(-1*sd_range, sd_range)

            samp1 = mu1 + var1
            samp2 = mu2 + var2
            #print samp1
            #print samp2 

            data_x.append((samp1, samp2))
            plt.scatter(samp1, samp2)
            label = np.int32(samp1 + samp2 > 0)
            data_y.append(label)

        batch_x = data_x
        batch_y =  np.int32(data_y)
        print batch_x
        return batch_x, batch_y

def xavier_mat_init(matrix):
    """
    Does Xavier initialization for a given matrix.

    Arguments:
    matrix -- matrix to be initialized
    """
    m = matrix.shape[0]
    n = matrix.shape[1]

    epsilon = np.sqrt(6) / np.sqrt(m + n)

    for i in range(m):
        for j in range(n):
            matrix[i][j] = np.random.uniform(-1 * epsilon, epsilon, 1)

    return matrix

def xavier_array_init(array):
    """
    Does Xavier initialization for a given array.

    Arguments:
    array -- array to be initialized
    """
    m = array.shape[0]
    epsilon = np.sqrt(6) / np.sqrt(m)
    for i in range(m):
        array[i] = np.random.uniform(-1 * epsilon, epsilon, 1)
    return array

def simple_forward_prop(data, labels):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    data = np.matrix(data)
    labels = np.array(labels)

    print data

    m = data.shape[0]
    n = data.shape[1]

    print data.shape, m, n
    print labels.shape

    # Hyperparameters
    power_constant = 1.0/3
    row = 0.5
    lamb = 0.01

    # Set up parameters
    W1 = np.ones((n, n))
    W2 = np.ones(n)
    
    # Initialization
    
    # Simple method
    # W1 = (-1) * W1/ np.linalg.norm(W1)
    # W2 = (-1) * W2/ np.linalg.norm(W2)
    
    # Xavier
    W1 = xavier_mat_init(W1)
    W2 = xavier_array_init(W2)
    print "W1 pre learning :"
    print W1
    print "W2 pre learning :"
    print W2
    
    error_count = 0
    general_count = 0

    for iter_num in range(m):
        
        x0 = data[iter_num, :]
        y0 = labels[iter_num]

        # print "x0 shape :", x0.shape
        # print "yo shape :", y0.shape
        # print "y0 :", y0
        # print "W1 shape: ", W1.shape
        # print "W2 shape: ", W2.shape
        # print "W2 shape[0]: ", W2.shape[0]

        # Forward prop
        z1 = np.dot(x0, W1) # + b1
        h1  = sigmoid(z1)
        x1 = np.random.binomial(1, h1) # roll dice based on h

        z2 = np.dot(x1, W2) # + b2
        h2 = sigmoid(z2)
        yHat = np.random.binomial(1, h2)

        # Error and reward signal calculation
        error = np.power(np.abs(y0 - yHat), power_constant)
        reward = 1 - error
        
        if error > 0:
            error_count += 1
        general_count += 1

        # print "yHat ", yHat
        print "reward", reward, "error", error, "running average error rate", 1.0 * error_count / general_count
        
        # print "h2 :", h2
        # print "z2 :", z2
        
        # print "z1 :", z1
        # print "h1 :", h1
        # print "x1 : ", x1
        
        # print "x1 shape :", x1.shape
        # print "h1 shape :", h1.shape
        # print ""

        # Reward signal propagation (single value)

        # W2 update

        for j in range(W2.shape[0]):
            W2[j] += row * reward * (yHat - h2) * x1[0, j] + lamb * row * (1 - reward) * (1 - yHat - h2) * x1[0, j]
            # b2[j] += row * reward * (yHat - h2) * x1[j] + lamb * row * (1 - reward) * (1 - yHat - h2) * x1[j]

        # W1 update
        for i in range(W1.shape[0]):    
            for j in range(W1.shape[1]):
                W1[i, j] += row * reward * (x1[0, i] - h1[0, i]) * x0[0, j] + lamb * row * (1 - reward) * (1 - x1[0, i] - h1[0, i]) * x0[0, j]

    print "post process W1"
    print W1










def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)



if __name__ == "__main__":
    batch_size = 100
    option = 2
    sd_range = 3


    batch_x, batch_y = generate_batch(batch_size, option, sd_range)
    
    simple_forward_prop(batch_x, batch_y)

