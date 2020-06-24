import os

import numpy as np
from PIL import Image


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

    for img_name in os.listdir(pics_path):
        if 'jpg' in img_name:
            img_path = os.path.join(pics_path, img_name)
            gray_images.append(to_gray(img_path))

    gray_images = np.array(gray_images)
    return gray_images
