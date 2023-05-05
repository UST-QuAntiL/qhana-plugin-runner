# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor


def add_background(
    points: np.array,
    resolution: int,
    predictor: Callable[[Tensor], Tensor],
    scatter: go.Scatter,
    two_classes: bool = False,
    label_to_int: dict = None,
) -> go.Figure:
    # Prep for grid (heatmap)
    # Get min and max of each dimension (here x and y) and write them into vector
    min_vec = points.min(axis=0)
    max_vec = points.max(axis=0)

    # The plotly zooms a little further out (autozoom), such that the points at the edge of the plot are perfectly visible.
    # If we do not account for this, there would be an margin of empty space between the points and the edge of the plot.
    # Therefore, we adapt the min and max values and thus, stretching the heatmap to the edge.
    max_values = (
        np.array([np.abs(min_vec), np.abs(max_vec)]).max(axis=0) * 0.2 + 1
    )  # +1 to avoid the zero case
    min_vec -= max_values
    max_vec += max_values

    # Compute ranges (still in vector)
    range_vec = max_vec - min_vec

    # Subtract 1 from the resolution, we add it later on again.
    # This lets us fill the whole screen, on low resolutions
    resolution -= 1
    h_vec = range_vec / resolution  # h_vec contains the step size for x and y direction

    # get x and y coordinates of a grid.
    # We need to determine the label of each point in the grid to create the heatmap
    max_vec += h_vec  # +h adds the previously subtracted resolution
    x_grid = np.arange(min_vec[0], max_vec[0], h_vec[0])
    y_grid = np.arange(min_vec[1], max_vec[1], h_vec[1])

    # Determine labels of points in grid for heatmap
    Z = []
    for y_v in y_grid:
        temp = []
        for x_v in x_grid:
            temp.append([x_v, y_v])
        Z.append(predictor(temp))

    Z = np.array(Z)

    if two_classes:
        # Create contours
        background = go.Figure(
            go.Contour(
                z=Z + 1,
                x=x_grid,
                y=y_grid,
                showscale=False,
                colorscale=list(px.colors.qualitative.D3),
                zmin=0,
                zmax=10,
                hoverinfo="skip",
                opacity=0.55,
                line_width=1,
            )
        )
    else:
        # Create heatmap
        background = go.Figure(
            go.Heatmap(
                z=Z,
                x=x_grid,
                y=y_grid,
                showscale=False,
                colorscale=list(px.colors.qualitative.D3),
                zmin=0,
                zmax=9,
                hoverinfo="skip",
                opacity=0.55,
            )
        )

    # Give markers a slightly thicker border, since their background will most likely have the same color.
    # Note, background is due to the heatmap
    scatter.update_traces(
        marker=dict(
            line=dict(
                width=2.5,
            )
        ),
    )
    # Correct colors of different labels. 0 gets the first color, 1 the second and so on
    # Thus the color match with the heatmap colors
    for sca_plt in scatter.data:
        label = int(
            sca_plt.legendgroup[0]
            if label_to_int is None
            else label_to_int[", ".join(sca_plt.legendgroup.split(", ")[:-1])]
        )
        sca_plt.update(
            marker=dict(
                color=px.colors.qualitative.D3[label],
            )
        )

    # Combine both heatmap and scatter plot
    # layout=fig.layout keeps the description of the legend
    scatter = go.Figure(data=background.data + scatter.data, layout=scatter.layout)

    # Set x- and y-axes correctly, in case the background still is not large enough for autozoom
    scatter.update_xaxes(range=[x_grid.min(), x_grid.max()])
    scatter.update_yaxes(range=[y_grid.min(), y_grid.max()])

    return scatter


def plot_data(
    train_data: Tensor,
    train_id_list: list,
    train_labels: Tensor,
    test_data: Tensor,
    test_id_list: list,
    test_labels: Tensor,
    resolution: int = 0,
    predictor: Callable[[Tensor], Tensor] = None,
    only_first_100: bool = True,
    title: str = "",
    label_to_int: dict = None,
) -> go.Scatter:
    # Prepare data
    dim = len(train_data[0])
    train_end = 100 if only_first_100 else len(train_data)
    test_end = 100 if only_first_100 else len(test_data)

    train_data = np.array(train_data[:train_end])
    test_data = np.array(test_data[:test_end])
    points = np.concatenate((train_data, test_data), axis=0)

    dimensions = ["x", "y", "z"][:dim]
    ids = train_id_list[:train_end] + test_id_list[:test_end]

    train_labels = [str(el) for el in train_labels[:train_end]]
    test_labels = [str(el) for el in test_labels[:test_end]]
    labels = train_labels + test_labels

    data_frame_content = dict(
        ID=ids,
        size=[10] * len(ids),
        label=labels,
        type=["train"] * len(train_data) + ["test"] * len(test_data),
    )
    for d, d_name in enumerate(dimensions):
        data_frame_content[d_name] = points[:, d]

    df = pd.DataFrame(data_frame_content)

    # Create plots
    if dim >= 3:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            hover_name="ID",
            size="size",
            color="label",
            symbol="type",
        )
    elif dim == 2:
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )

        if resolution > 0 and predictor is not None:
            fig = add_background(
                points,
                resolution,
                predictor,
                fig,
                two_classes=len(set(train_labels)) == 2,
                label_to_int=label_to_int,
            )
    else:
        df["y"] = [0] * len(df["x"])
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )

    fig.update_layout(
        dict(
            font=dict(
                size=15,
            ),
        )
    )
    fig.update_layout(
        dict(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=30,
                ),
            )
        )
    )

    return fig


def plot_confusion_matrix(y_true: list, y_pred: list, labels: list) -> go.Figure:
    labels.sort()
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels).T

    labels = [str(label) for label in labels]

    df_content = dict()
    for label, v in zip(labels, conf_matrix):
        df_content[label] = [str(el) for el in v]
    df = pd.DataFrame(df_content)
    df.index = pd.Index(labels, name="True label")
    df.columns = pd.Index(labels, name="Predicted label")

    fig = px.imshow(
        df,
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Aggrnyl,
    )
    fig.update_layout(
        dict(
            font=dict(
                size=18,
            ),
        )
    )
    fig.update_layout(
        dict(
            title=dict(
                text="Confusion Matrix",
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=30,
                ),
            )
        )
    )
    fig.update_xaxes(dict(titlefont=dict(size=18), tickfont=dict(size=18)))
    fig.update_yaxes(dict(titlefont=dict(size=18), tickfont=dict(size=18)))

    return fig
