# implementation based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html


from pennylane import numpy as np

# PyTorch
import torch

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


def plot_classification(
    model, X, X_train, X_test, y_train, y_test, accuracy_on_test_data, resolution
):
    """
    visualize the classification results of the given model

    model: network
    X: all data elements
    X_train: training data
    X_test: test data
    y_train: training labels
    y_test: test labels
    accuracy_on_test_data: (previously computed) accuracy of the models predictions on test data
    resolution: how many evaluations of the classifier in each dimension
    """
    # number of classes
    classes = list(set(list(y_train) + list(y_test)))
    n_classes = len(classes)

    # color map for scatter plots
    colors = cm.get_cmap("rainbow", n_classes)

    # figure and subplot
    fig, ax = plt.subplots()

    # draw decision boundaries

    # bounds of the figure and grid
    factor = 1.1
    x_min = np.min(X[:, 0]) * factor
    x_max = np.max(X[:, 0]) * factor
    y_min = np.min(X[:, 1]) * factor
    y_max = np.max(X[:, 1]) * factor
    print(x_min, x_max, y_min, y_max)

    # generate gridpoints

    # how fine grained the background contour should be
    x_range = x_max - x_min
    y_range = y_max - y_min
    hx = x_range / resolution
    hy = y_range / resolution

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # calculate class predictions for gridpoints
    _, grid_results = torch.max(
        model(torch.Tensor(grid_points)), 1
    )  # np.asarray([model(torch.Tensor(point)) for point in grid_points]) #model(torch.Tensor(grid_points))

    Z = grid_results.reshape(xx.shape)

    # draw class predictions for grid as background
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
    ax.scatter(X_train[:, 0], X_train[:, 1], c="b", s=50, marker="x")

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

    # legend elements for classes
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
    # legend element for training element crosses
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

    # add legend elements to plot
    ax.legend(handles=legend_elements)

    # set pot title
    ax.set_title(
        "Classification \naccuracy on test data={}".format(accuracy_on_test_data)
    )

    return fig
