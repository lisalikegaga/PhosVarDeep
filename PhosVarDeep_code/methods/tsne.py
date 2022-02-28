# coding:utf-8
"""
画加入ppi特征的tsne图 与原始的正交编码特征的对比
"""
from time import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, datasets
import os
from keras.utils.np_utils import to_categorical


def plot_embedding_2d(X, y, fig, title=None, num=None):
    # %%
    # 将降维后的数据可视化,2维
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    # X = (X - x_min) / (x_max - x_min)

    # 坐标缩放到[-1,1]之间
    x_avg = (x_max + x_min) / 2
    X = 2 * (X - x_avg) / (x_max - x_min)

    # 降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, num)
    for i in range(X.shape[0]):
        # ax.text(X[i, 0], X[i, 1],str(y[i]),
        #          color=plt.cm.Set1(y[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        if int(y[i]) == 1:

            ax.scatter(X[i, 0], X[i, 1], marker='.', color='r')


        else:

            ax.scatter(X[i, 0], X[i, 1], marker='.', color='b')
    ax.scatter(X[0, 0], X[0, 1], marker='.', color='r', label='positive phospho-variants')
    ax.legend(loc='lower right')
    ax.scatter(X[0, 0], X[0, 1], marker='.', color='b', label='negative phospho-variants')
    ax.legend(loc='lower right')
    plt.xlim(-1, 1.25)
    plt.ylim(-1, 1)

    if title is not None:
        plt.title(title, fontsize=10)


def plot_embedding1_2d(X, y, fig, title=None, num=None):
    # %%
    # 将降维后的数据可视化,2维
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    # X = (X - x_min) / (x_max - x_min)

    # 坐标缩放到[-1,1]之间
    x_avg = (x_max + x_min) / 2
    X = 2 * (X - x_avg) / (x_max - x_min)

    # 降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, num)
    for i in range(X.shape[0]):
        # ax.text(X[i, 0], X[i, 1],str(y[i]),
        #          color=plt.cm.Set1(y[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        if int(y[i]) == 1:

            ax.scatter(-X[i, 0], X[i, 1] / 2, marker='.', color='r')


        else:

            ax.scatter(-X[i, 0], X[i, 1] / 2, marker='.', color='b')

    ax.scatter(-X[0, 0], X[0, 1] / 2, marker='.', color='r', label='positive phospho-variants')
    ax.legend(loc='lower right')
    ax.scatter(-X[0, 0], X[0, 1] / 2, marker='.', color='b', label='negative phospho-variants')
    ax.legend(loc='lower right')
    plt.ylim(-1, 1)
    plt.xlim(-1.25, 1.25)

    if title is not None:
        plt.title(title, fontsize=10)


def plot_embedding2_2d(X, y, fig, title=None, num=None):
    # %%
    # 将降维后的数据可视化,2维
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    # X = (X - x_min) / (x_max - x_min)

    # 坐标缩放到[-1,1]之间
    x_avg = (x_max + x_min) / 2
    X = 2 * (X - x_avg) / (x_max - x_min)

    # 降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, num)
    for i in range(X.shape[0]):
        # ax.text(X[i, 0], X[i, 1],str(y[i]),
        #          color=plt.cm.Set1(y[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        if int(y[i]) == 1:

            ax.scatter(X[i, 0], X[i, 1] / 2, marker='.', color='r')


        else:

            ax.scatter(X[i, 0], X[i, 1] / 2, marker='.', color='b')

    ax.scatter(X[0, 0], X[0, 1] / 2, marker='.', color='r', label='positive phospho-variants')
    ax.legend(loc='lower right')
    ax.scatter(X[0, 0], X[0, 1] / 2, marker='.', color='b', label='negative phospho-variants')
    ax.legend(loc='lower right')
    plt.ylim(-1, 1)
    plt.xlim(-1.25, 0.75)

    if title is not None:
        plt.title(title, fontsize=10)


def run_tsne():
    # %%
    # 加载原始数据，显示数据
    fig = plt.figure(figsize=(16, 4))

    y = np.loadtxt(folder + '/y.txt')
    original_f = np.loadtxt(folder + '/original-features.txt')
    feature_seq = np.loadtxt(folder + '/combined-features.txt')

    # t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(original_f)
    print(X_tsne.shape)
    np.savetxt(folder + '/x_tsen.txt', X_tsne)
    plot_embedding_2d(X_tsne[:, 0:2], y, fig, "t-SNE plot of original combined features", 1)


    print("Computing t-SNE embedding for feature")
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    t0 = time()
    X_tsne1 = tsne.fit_transform(feature_seq)
    print(X_tsne1.shape)
    np.savetxt(folder + '/x_tsen_feature.txt', X_tsne1)
    plot_embedding2_2d(X_tsne1[:, 0:2], y, fig, "t-SNE plot of combined features by PhosVarSNN", 2)

    plt.savefig(folder + '/2D1.png', dpi=300)


if __name__ == '__main__':
    ptm_type = 'ST'
    folder = "tsne_phosvar/test3/{:s}".format(ptm_type)
    if not os.path.exists(folder):
        os.makedirs(folder)
    run_tsne()
