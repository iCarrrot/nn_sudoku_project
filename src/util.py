import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm


def to_gray(img_path):
    return np.asarray(Image.open(img_path).convert('L'))


sudoku = np.array(
    [
        [1, 4, 5, 3, 2, 7, 6, 9, 8],
        [8, 3, 9, 6, 5, 4, 1, 2, 7],
        [6, 7, 2, 9, 1, 8, 5, 4, 3],

        [4, 9, 6, 1, 8, 5, 3, 7, 2],
        [2, 1, 8, 4, 7, 3, 9, 5, 6],
        [7, 5, 3, 2, 9, 6, 4, 8, 1],

        [3, 6, 7, 5, 4, 2, 8, 1, 9],
        [9, 8, 4, 7, 6, 1, 2, 3, 5],
        [5, 2, 1, 8, 3, 9, 7, 6, 4],
    ]
)
sudoku2 = np.array(
    [
        [1, 4, 5, 3, 2, 7, 6, 9, 7],
        [8, 3, 9, 6, 5, 4, 1, 2, 8],
        [6, 7, 2, 9, 1, 8, 5, 4, 3],

        [4, 9, 6, 1, 8, 5, 3, 7, 2],
        [2, 1, 8, 4, 7, 3, 9, 5, 6],
        [7, 5, 3, 2, 9, 6, 4, 8, 1],

        [3, 6, 7, 5, 4, 2, 8, 1, 9],
        [9, 8, 4, 7, 6, 1, 2, 3, 5],
        [5, 2, 1, 8, 3, 9, 7, 6, 4],
    ]
)


def check_sudoku(sudoku):
    ln_sp = np.arange(1, 10)
    xx, yy = np.meshgrid(ln_sp, ln_sp)
    tmp_arr = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            tmp_arr[3 * i + j] = sudoku[3 * i:3 * i + 3, 3 * j:3 * j + 3].ravel()
    return (
            np.all(np.sort(tmp_arr, axis=1) == xx)
            and np.all(np.sort(sudoku, axis=1) == xx)
            and np.all(np.sort(sudoku, axis=0) == yy)
    )


def get_pics_path(default="~/Downloads/dataset"):
    pics_path = 'drive/My Drive/dataset/'

    if not os.path.exists('drive/My Drive/dataset/'):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except KeyError:
            pics_path = os.path.expanduser(default)
    return pics_path


def get_gray_images(pics_path):
    gray_images = []
    labels = []
    with open(os.path.join(pics_path, 'labels.json')) as json_file:
        data = json.load(json_file)
    for img_name in data:
        img_path = os.path.join(pics_path, img_name)
        gray_images.append(to_gray(img_path))
        labels.append(data[img_name])

    gray_images = np.array(gray_images)
    return gray_images, labels


def get_pure_data(dataloader):
    digits, labels = [], []
    for _, (data, targets) in tqdm(enumerate(dataloader)):
        digits.append(data.cpu().numpy().reshape(-1, 28, 28))
        labels.append(targets.cpu().numpy())

    return np.vstack(digits).reshape(-1, 28 * 28), np.hstack(labels)


def plot_digits(X, y):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    plt.figure(figsize=(16, 9))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(0.1 * y[i]), fontdict={'weight': 'bold', 'size': 8})

    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_samples(X, y, n=8):
    plt.figure(figsize=(20, 6))
    ids = np.random.choice(X.shape[0], n, replace=False)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[ids[i]].reshape(28, 28), cmap='gray')
        plt.title(y[ids[i]])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def present_dataset(dataloader):
    digits, labels = get_pure_data(dataloader)

    plot_samples(digits, labels)
    tsne_size = 3000
    if digits.shape[0] > tsne_size:
        ids = np.random.choice(digits.shape[0], tsne_size, replace=False)
    else:
        ids = np.ones(digits.shape[0])

    transformation = TSNE(n_components=2, init='pca')
    X = transformation.fit_transform(digits[ids].reshape(tsne_size, -1))

    plot_digits(X, labels[ids])
