
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

    csfont = {'fontname':'Times New Roman'}
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('on')

    ax.set_xlabel('x1', **csfont)
    ax.set_ylabel('xy', **csfont)
    plt.savefig('boundary', bbox_inches='tight', pad_inches=0.1)

    # Plot also the training points
    # ax.scatter(X[:, 0], X[:, 1], c=Y, marker='.')#, cmap=plt.cm.Paired)
    #ax.set_title('Perceptron')
    plt.show()