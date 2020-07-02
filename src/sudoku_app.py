import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from src.net import get_dataloader
from src.preprocessing import split_into_cells


def get_predictions(model, device, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.cpu().numpy().reshape((9, 9)), output.cpu().numpy()


def find_errors(grid, i, j):
    related = []
    for k in range(9):
        if grid[i, j] == grid[i, k] and j < k:
            related.append([(i, k), (i, j)])

        if grid[i, j] == grid[k, j] and i < k:
            related.append([(k, j), (i, j)])

    for k, l in itertools.product(range(3), range(3)):
        ci, cj = 3 * (i // 3) + k, 3 * (j // 3) + l
        if grid[i, j] == grid[ci, cj] and (ci > i or (cj > j and ci == i)) and [(ci, cj), (i, j)] not in related:
            related.append([(ci, cj), (i, j)])

    return related


def process_grid(grid_cells, y, model, threshold=0.95):
    grid_dataloader = get_dataloader(grid_cells, y)
    predicted, ppbs = get_predictions(model, 'cuda', grid_dataloader)
    alternative_values = ppbs.argsort(axis=1)[:, -2].reshape((9, 9))

    ppbs = np.e ** (ppbs)
    print(predicted)
    associated_cells = []
    for i, j in itertools.product(range(9), range(9)):
        for cells in find_errors(predicted, i, j):
            associated_cells.append(cells)

    if len(associated_cells) == 0:
        return []

    error_chances = []
    cell_counts = {}
    for (i1, j1), (i2, j2) in associated_cells:
        if (i1, j1) not in cell_counts:
            cell_counts[(i1, j1)] = 0
        if (i2, j2) not in cell_counts:
            cell_counts[(i2, j2)] = 0

        cell_counts[(i1, j1)] += 1
        cell_counts[(i2, j2)] += 1
        predicted_digit = predicted[i1, j1]
        error_chances.append([ppbs[9 * i1 + j1, predicted_digit], ppbs[9 * i2 + j2, predicted_digit]])

    constraints_cover = np.zeros(len(cell_counts))
    for k in cell_counts:
        i, j = k
        # print(k, cell_counts[k], ppbs[9 * i + j, predicted[i, j]], ppbs[9 * i + j, alternative_values[i, j]],
        # alternative_values[i, j])
    error_chances = np.asarray(error_chances)
    order = error_chances.min(axis=1).argsort()[::-1]

    i = 1
    response = {'m': []}
    for index in order:
        if error_chances[index, :].mean() < threshold:
            break

        print(associated_cells[index], error_chances[index, :])

        if i == 1:
            response['o'] = associated_cells[index]
        else:
            response['m'].append(associated_cells[index])
        i += 1
    return response


def mark_error(img, errors, color=None, thick=50):
    cell_width = img.shape[0] // 9
    if color is None:
        color = (np.random.choice(255), np.random.choice(255), np.random.choice(255))

    for j, i in errors:
        img = cv2.rectangle(img,
                            (i * cell_width, j * cell_width),
                            ((i + 1) * cell_width, (j + 1) * cell_width),
                            color,
                            thick)

    return img


def check(instance, model, threshold):
    img, y = instance
    grid_cells = split_into_cells(img)

    empties = [(i // 9, i % 9) for i in range(81) if grid_cells[i].mean() > 252]

    if len(empties) > 0:
        error_dict = {'e': empties}

    else:
        error_dict = process_grid(grid_cells, y, model, threshold=threshold)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if len(error_dict) > 0:
        for error_type, cells in error_dict.items():
            if error_type == 'e':
                final_img = mark_error(rgb_img, cells, color=(0, 0, 255))

            if error_type == 'o':
                final_img = mark_error(rgb_img, cells, color=(255, 0, 0))

            if error_type == 'm':
                for c in cells:
                    rgb_img = mark_error(rgb_img, c, thick=40)

    final_img = rgb_img

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(final_img)
    plt.show()

    if len(error_dict) == 0:
        # plt.imshow(rgb_img)
        # plt.show()
        print('This grid has no errors!')
        return
