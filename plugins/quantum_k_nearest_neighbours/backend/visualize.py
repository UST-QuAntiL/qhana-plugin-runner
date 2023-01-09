import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix


def get_id_list(id_to_idx: dict) -> list:
    ids = ["id"] * len(id_to_idx)
    for id, idx in id_to_idx.items():
        ids[idx] = id
    return ids


def add_background(points, resolution, predictor, scatter, two_classes=False):
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
        label = int(sca_plt.legendgroup[0])
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
    train_data,
    train_id_to_idx,
    train_labels,
    test_data,
    test_id_to_idx,
    test_labels,
    resolution=0,
    predictor=None,
    only_first_100=True,
    title="",
):
    # Prepare data
    dim = len(train_data[0])
    train_end = 100 if only_first_100 else len(train_data)
    test_end = 100 if only_first_100 else len(test_data)

    train_data = np.array(train_data)[:train_end]
    test_data = np.array(test_data)[:test_end]
    points = np.concatenate((train_data, test_data), axis=0)

    dimensions = ["x", "y", "z"][:dim]
    ids = (
        get_id_list(train_id_to_idx)[:train_end] + get_id_list(test_id_to_idx)[:test_end]
    )

    train_labels = [str(int(el)) for el in train_labels[:train_end]]
    test_labels = [str(int(el)) for el in test_labels[:test_end]]
    labels = train_labels[:train_end] + test_labels[:test_end]

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


def plot_confusion_matrix(y_true, y_pred, labels: list):
    labels.sort()
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels).T

    labels = [str(label) for label in labels]

    df_content = dict()
    for label, v in zip(labels, conf_matrix):
        df_content[label] = [str(el) for el in v]
    df = pd.DataFrame(df_content)
    df.index = pd.Index(labels, name="True label")
    df.columns = pd.Index(
        labels, name="Predicted label"
    )

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
