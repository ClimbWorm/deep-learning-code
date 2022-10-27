import numpy as np
import matplotlib.pyplot as plt
import random

def imshow(image, ax=None, title=None, normalize=True):
    """helper function: Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.savefig('test loader.jpg')

    return ax


def show_example(X, y, label, grid=(3, 3)):
    y_label = [i for i, tag in enumerate(y) if label == tag]
    random.shuffle(y_label)

    rows = grid[0]
    columns = grid[1]

    fig, axes = plt.subplots(rows, columns)
    fig.set_figheight(7)
    fig.set_figwidth(7)

    for row in axes:
        for col in row:
            col.imshow(X[y_label.pop()])
    plt.show()