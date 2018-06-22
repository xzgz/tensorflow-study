from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def visualize(embed, x_test):

    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)

    # two ways of visualization: leave with original scale
    feat = embed
    ax_min = np.min(embed, 0)
    ax_max = np.max(embed, 0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x_test[i], zoom=0.6, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.xticks([]), plt.yticks([])
    plt.title('Embedding from the last layer of the network')
    plt.show()


def show_data(data, head, padsize = 1, padval = 0):
    data = np.array(data)
    data = data.astype('float')
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    #     print 'n', n
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    #     print 'padding', padding
    #     print 'data.shape0', data.shape
    data = np.pad(data, padding, mode = 'constant', constant_values = (padval, padval))

    # tile the filters into an image
    #     print 'data.shape1', data.shape
    #     print 'data.shape[1:]', data.shape[1:]
    ndim = data.ndim
    data = data.reshape((n, n) + data.shape[1:])
    #     print 'data.shape2', data.shape
    #     print 'ndim', ndim

    #     print 'kk', (0, 2, 1, 3) + tuple(range(4, ndim + 1)), tuple(range(4, ndim + 1))
    data = data.transpose((0, 2, 1, 3) + tuple(range(4, ndim + 1)))
    #     print 'data.shape3', data.shape

    #     print 'data.shape[4:]', data.shape[4:]
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #     print 'data.shape4', data.shape
    plt.figure()
    plt.title(head)
    #     plt.imshow(data)
    plt.imshow(data, cmap='gray')
    plt.axis('off')


if __name__ == "__main__":

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_test = mnist.test.images
    x_test = x_test.reshape([-1, 28, 28])

    embed = np.fromfile('embed.txt', dtype=np.float32)
    embed = embed.reshape([-1, 2])

    visualize(embed, x_test)