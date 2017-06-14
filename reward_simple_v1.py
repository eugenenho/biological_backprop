#!/usr/bin/env python
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

# Model design parameters
n = 2

batch_size = 50
total_iter_num = 500
m = batch_size * total_iter_num

hidden_dim = 20

# Data generation parameters
option = 3
sd_range = 3
num_complexity = 6 # SD for blobs for (10, 10), (-10, -10)

# Other parameters
threshold = 0.05

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

    elif (option == 2):
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

    else:

        data_x = []
        data_y = []

        for i in range(batch_size):
            samp1 = np.random.random_integers(-10, 10)
            samp2 = np.random.random_integers(-10, 10)

            #print samp1
            #print samp2 

            data_x.append((samp1, samp2))
            plt.scatter(samp1, samp2)
            label = np.int32(samp2 > samp1 * samp1)
            data_y.append(label)


        batch_x = np.matrix(data_x)
        batch_y =  np.array(np.int32(data_y))

        print "batch_x shape ", batch_x.shape
        print batch_x
        print batch_y

        plt.figure()
        plt.title("Quadratic dataset", fontsize='small')
        plt.scatter(batch_x[:, 0], batch_x[:, 1], marker='.', c=batch_y)
        plt.show()
       
        print batch_x
        return batch_x, batch_y
    

def generate_nonlin_data(num_features, num_samples):
    plt.subplot(326)
    plt.title("Gaussian divided into three quantiles", fontsize='small')
    X1, Y1 = make_gaussian_quantiles(mean = (1, 1), cov = 5, n_samples=num_samples, n_features=num_features, n_classes=2)
    print X1
    print Y1
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1)
    plt.show()
    return X1, Y1

def generate_lin_data(num_features, num_samples, num_complexity):
    # plt.subplot(325)
    # plt.title("Two blobs", fontsize='small')
    X1, Y1 = make_blobs(n_samples = num_samples, n_features=num_features, centers=[(10,) * num_features, (-10,) * num_features], cluster_std = num_complexity, random_state=1)
    # plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1)
    # plt.show()
    return X1, Y1

def generate_quad_data(num_features, num_samples):
    X1 = np.random.uniform(-10, 10, (num_samples, num_features))
    squared = 0.1 * X1[:, 0]**2
    Y1 = np.int32(squared < X1[:, 1])
    plt.figure()
    plt.title("Quadratic dataset", fontsize='small')
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1)
    plt.show()
    return X1, Y1

def generate_xor_data(num_features, num_samples):
    X1 = np.random.uniform(-10, 10, (num_samples, num_features))
    Y1 = np.int32(X1[:, 1] * X1[:, 0] < 0)
    plt.figure()
    plt.title("Xor dataset", fontsize='small')
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1)
    #plt.show()
    return X1, Y1


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

def simple_forward_prop(data, labels, hidden_dim, batch_size):
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

    #print data

    m = data.shape[0]
    n = data.shape[1]

    #print data.shape, m, n
    #print labels.shape

    # Hyperparameters
    power_constant = 1.0/3
    row = 0.9  #0.5
    lamb = 0.001 #0.01

    # Set up parameters
    W1 = np.zeros((n, hidden_dim))
    W2 = np.zeros(hidden_dim)
    b1 = np.zeros(hidden_dim)
    b2 = np.zeros(1)
    
    # Initialization
    
    # Simple method
    # W1 = (-1) * W1/ np.linalg.norm(W1)
    # W2 = (-1) * W2/ np.linalg.norm(W2)
    
    # Xavier
    W1 = xavier_mat_init(W1)
    W2 = xavier_array_init(W2)
    b1 = xavier_array_init(b1)
    b2 = xavier_array_init(b2)
    # print "W1 pre learning :"
    # print W1
    # print "W2 pre learning :"
    # print W2
    # print "b1 pre learning :"
    # print b1
    # print "b2 pre learning :"
    # print b2
    
    
    error_count = 0
    batch_count = 0
    error_array = []

    start_time = time.time()
    for iter_num in range(m):
        
        x0 = data[iter_num, :]
        y0 = labels[iter_num]

        # Forward prop
        z1 = np.dot(x0, W1) + b1
        h1  = sigmoid(z1)
        x1 = np.random.binomial(1, h1) # roll dice based on h

        z2 = np.dot(x1, W2) + b2
        h2 = sigmoid(z2)
        yHat = np.random.binomial(1, h2)

        # Error and reward signal calculation
        error = np.power(np.abs(y0 - yHat), power_constant)
        reward = 1 - error
        
        if error > 0: error_count += 1
        batch_count += 1

        # Reward signal propagation (single value)

        # W2 update
        for j in range(W2.shape[0]):
            W2[j] += row * reward * (yHat - h2) * x1[0, j] + lamb * row * (1 - reward) * (1 - yHat - h2) * x1[0, j]
            b2 += row * reward * (yHat - h2) * x1[0, j] + lamb * row * (1 - reward) * (1 - yHat - h2) * x1[0, j]

        # W1 update
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):    
                W1[i, j] += row * reward * (x1[0, j] - h1[0, j]) * x0[0, i] + lamb * row * (1 - reward) * (1 - x1[0, j] - h1[0, j]) * x0[0, i]
                b1[j] += row * reward * (x1[0, j] - h1[0, j]) * x0[0, i] + lamb * row * (1 - reward) * (1 - x1[0, j] - h1[0, j]) * x0[0, i]
                
        if batch_count == batch_size:
            error_array.append(1.0 * error_count / batch_count)
            batch_count = 0
            error_count = 0

    # print "Training took " + str(time.time() - start_time) + " seconds"
    # print "post process b1"
    # print b1
    # print "post process b2"
    # print b2

    axis = np.arange(len(error_array))
    
    # find the training completion point. return the batch number it took to reach completion point
    # return find_completion_point(error_array)

    plt.figure()
    plt.plot(axis, error_array,'r', label = 'batch error')
    plt.legend()
    plt.show()

    plot_boundary(data, labels, W1, W2, b1, b2)



def find_completion_point(error_array):
    array_length = len(error_array)

    # Iterate over error_array in reverse order
    for index in range(array_length - 1, -1, -1):
        if (error_array[index] > threshold):
            return index
    
    return 0        

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


def complexity_run(sd_low, sd_high):

    # Iterate over complexity
    results = []
    
    for sd in range(sd_high, sd_low - 1, -1):
        
        
        # Generate data with given complexity
        data_x, data_y = generate_lin_data(n, m, sd)

        # Iterate over 10 runs
        indiv_run_results = []
        for i in range(10):
            indiv_run_results.append(simple_forward_prop(data_x, data_y, hidden_dim, batch_size))

        # Save average to results
        print "on complexity: ", sd, "and results:", indiv_run_results
        results.append(np.mean(indiv_run_results))


    # Plot results
    plt.figure()
    plt.title("Complexity results", fontsize='small')
    plt.plot(range(sd_high, sd_low - 1, -1), results, marker='.')
    plt.show()
    
def dimension_run(dim_low, dim_high, step_size):

    # Iterate over complexity
    results = []
    
    for dim in range(dim_low, dim_high + 1, step_size):
        
        # Generate data with given complexity
        data_x, data_y = generate_lin_data(dim, m, num_complexity)

        # Iterate over 10 runs
        indiv_run_results = []
        for i in range(10):
            indiv_run_results.append(simple_forward_prop(data_x, data_y, hidden_dim, batch_size))

        # Save average to results
        print "on dim: ", dim, "and results:", indiv_run_results
        results.append(np.mean(indiv_run_results))

    # Plot results
    plt.figure()
    plt.title("Dimension results", fontsize='small')
    plt.plot(range(dim_low, dim_high + 1, step_size), results, marker='.')
    plt.show()

def net_predict(data, W1, W2, b1, b2):

    print "net_predict here"

    print "data"
    print data

    print "W1"
    print W1

    print "W2"
    print W2
    # Forward prop
    z1 = np.dot(data, W1) + b1 # [m x n x n x h] = [m x h]
    h1  = sigmoid(z1)   # [m x h]
    x1 = np.int32(h1 >= 0.5) # [m x h]
    # x1 = h1

    z2 = np.dot(x1, W2) + b2 # [m x h x h x 1]
    h2 = sigmoid(z2) # [m x 1]
    yHat = np.int32(h2 >= 0.5) # [m x 1]

    return yHat


def plot_boundary(X, Y, W1, W2, b1, b2):
    # create a mesh to plot in
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = net_predict(np.c_[xx.ravel(), yy.ravel()], W1, W2, b1, b2)

    print xx
    print yy

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    print Z
    print np.sum(Z)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')

    # Plot also the training points
    # ax.scatter(X[:, 0], X[:, 1], c=Y, marker='.')#, cmap=plt.cm.Paired)
    ax.set_title('Perceptron')
    plt.show()

if __name__ == "__main__":
    
    #complexity_run(3, 10)
    #dimension_run(2, 10, 1)

    # data_x, data_y = generate_nonlin_data(n, m)
    # data_x, data_y = generate_lin_data(n, m, num_complexity)
    data_x, data_y = generate_quad_data(n, m)
    # data_x, data_y = generate_xor_data(n, m)
    
    print "starting data generation"
    #data_x, data_y = generate_batch(batch_size * total_iter_num, option, sd_range)
    print "done with data generation"
    print data_x.shape
    print data_y.shape
    simple_forward_prop(data_x, data_y, hidden_dim, batch_size)
