from pennylane import numpy as np

# PyTorch
import torch

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


def plot_classification(
    model, X, X_train, X_test, y_train, y_test, accuracy_on_test_data
):
    # directory for coloring the plot
    classes = list(set(list(y_train) + list(y_test)))
    # print(classes)
    n_classes = len(classes)

    colors = cm.get_cmap("rainbow", n_classes)

    fig, ax = plt.subplots()

    # draw decision boundaries

    factor = 1.1
    x_min = np.min(X[:, 0]) * factor
    x_max = np.max(X[:, 0]) * factor
    y_min = np.min(X[:, 1]) * factor
    y_max = np.max(X[:, 1]) * factor
    print(x_min, x_max, y_min, y_max)

    # generate grid
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    print(grid_points)

    # import random
    # evaluate decision function for each grid point
    # def fun(point): # TODO replace with QNN -> evaluate classification for given point
    # return (point - [x_min, y_min]) / [x_max-x_min, y_max-y_min]
    #    return random.uniform(0, 1)

    _, grid_results = torch.max(
        model(torch.Tensor(grid_points)), 1
    )  # np.asarray([model(torch.Tensor(point)) for point in grid_points]) #model(torch.Tensor(grid_points))
    print(grid_results)
    # Z = np.tanh(grid_results[:, 1] - grid_results[:, 0])
    # Z = Z.reshape(xx.shape)
    Z = grid_results.reshape(xx.shape)

    ax.contourf(
        xx, yy, Z, levels=n_classes - 1, linestyles=["-"], cmap="winter", alpha=0.3
    )

    # draw training data
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )
    # mark train data
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c="b", s=50, marker="x"
    )  # , label="train data")

    # scatter test set
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )

    # create legend elements
    # classes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=i,
            markerfacecolor=colors(i),
            markersize=10,
            ls="",
        )
        for i in range(n_classes)
    ]
    # training element crosses
    legend_elements = legend_elements + [
        Line2D(
            [0],
            [0],
            marker="x",
            color="b",
            label="train data",
            markerfacecolor="g",
            markersize=8,
            ls="",
        )
    ]

    ax.legend(handles=legend_elements)

    ax.set_title(
        "Classification \naccuracy on test data={}".format(accuracy_on_test_data)
    )

    plt.show()
