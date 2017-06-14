def generate_nonlin_data(num_features, num_samples):
    plt.figure()
    csfont = {'fontname':'Times New Roman'}
    
    X1, Y1 = make_gaussian_quantiles(mean = (1, 1), cov = 5, n_samples=num_samples, n_features=num_features, n_classes=2)
    
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1, cmap=plt.cm.Paired)
    plt.savefig('non_lin_data', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return X1, Y1

def generate_lin_data(num_features, num_samples, num_complexity):
    # plt.subplot(325)
    plt.figure()
    csfont = {'fontname':'Times New Roman'}

    # plt.title("Example input data", **csfont)
    X1, Y1 = make_blobs(n_samples = num_samples, n_features=num_features, centers=[(10,) * num_features, (-10,) * num_features], cluster_std = num_complexity, random_state=1)
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    plt.xlabel('x1', **csfont)
    plt.ylabel('x2', **csfont)
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1, cmap=plt.cm.Paired)
    plt.savefig('figure1_example_data', bbox_inches='tight', pad_inches=0.1)
    return X1, Y1

def generate_quad_data(num_features, num_samples):
    X1 = np.random.uniform(-10, 10, (num_samples, num_features))
    squared = 0.1 * X1[:, 0]**2
    Y1 = np.int32(squared < X1[:, 1])
    
    plt.figure()
    csfont = {'fontname':'Times New Roman'}
    plt.xlabel('x1', **csfont)
    plt.ylabel('x2', **csfont)
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1, cmap=plt.cm.Paired)
    plt.savefig('quad_data', bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    return X1, Y1

def generate_xor_data(num_features, num_samples):
    X1 = np.random.uniform(-10, 10, (num_samples, num_features))
    Y1 = np.int32(X1[:, 1] * X1[:, 0] < 0)
    
    plt.figure()
    csfont = {'fontname':'Times New Roman'}
    plt.xlabel('x1', **csfont)
    plt.ylabel('x2', **csfont)
    plt.scatter(X1[:, 0], X1[:, 1], marker='.', c=Y1, cmap=plt.cm.Paired)
    plt.savefig('xor_data', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return X1, Y1


# data_x, data_y = generate_nonlin_data(n, m)
# data_x, data_y = generate_lin_data(n, m, num_complexity)
# data_x, data_y = generate_quad_data(n, m)
# data_x, data_y = generate_xor_data(n, m)


####### plot code
plt.figure()
plt.plot(axis, error_array, '0.2')

csfont = {'fontname':'Times New Roman'}
plt.xlabel('Number of iterations', **csfont)
plt.ylabel('Error rate per batch', **csfont)
plt.savefig('error_rate_chart', bbox_inches='tight', pad_inches=0.1)
plt.show()
