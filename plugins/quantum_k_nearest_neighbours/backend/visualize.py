import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def get_id_list(id_to_idx: dict) -> list:
    ids = ["id"]*len(id_to_idx)
    for id, idx in id_to_idx.items():
        ids[idx] = id
    return ids


def add_heatmap(points, resolution, predictor, scatter):
    # Prep for grid (heatmap)
    # Get min and max of each dimension (here x and y) and write them into vector
    min_vec = points.min(axis=0)
    max_vec = points.max(axis=0)
    max_v = max_vec.max() * 0.1
    min_vec -= max_v
    max_vec += max_v

    # Compute ranges (still in vector)
    range_vec = max_vec - min_vec

    # Subtract 1 from the resolution, we add it later on again, but doing it like this, fills the whole screen
    # with the heatmap and leaves no blank border
    # resolution -= 1
    h_vec = range_vec / resolution  # h_vec contains the step size for x and y direction

    # get x and y coordinates of a grid.
    # We need to determine the label of each point in the grid to create the heatmap
    # max_vec += h_vec  # +h adds the previously subtracted resolution
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

    # Create heatmap
    heatmap = go.Figure(go.Heatmap(
        z=Z,
        x=x_grid,
        y=y_grid,
        showscale=False,
        colorscale=px.colors.qualitative.D3,
        zmin=0,
        zmax=10,
        hoverinfo="skip",
        opacity=0.75,
    ))

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

    scatter = go.Figure(data=heatmap.data + scatter.data, layout=scatter.layout)

    return scatter


def plot_data(train_data, train_id_to_idx, train_labels, test_data, test_id_to_idx, test_labels, resolution=0, predictor=None, only_first_100=True):
    # Prepare data
    dim = len(train_data[0])
    train_end = 100 if only_first_100 else len(train_data)
    test_end = 100 if only_first_100 else len(test_data)

    train_data = np.array(train_data)[:train_end]
    test_data = np.array(test_data)[:test_end]
    points = np.concatenate((train_data, test_data), axis=0)

    dimensions = ["x", "y", "z"][:dim]
    ids = get_id_list(train_id_to_idx)[:train_end] + get_id_list(test_id_to_idx)[:test_end]

    train_labels = [str(int(el)) for el in train_labels[:train_end]]
    test_labels = [str(int(el)) for el in test_labels[:test_end]]
    labels = train_labels[:train_end] + test_labels[:test_end]

    data_frame_content = dict(
        ID=ids,
        size=[10] * len(ids),
        label=labels,
        type=["train"]*len(train_data) + ["test"]*len(test_data),
    )
    for d, d_name in enumerate(dimensions):
        data_frame_content[d_name] = points[:, d]

    df = pd.DataFrame(data_frame_content)

    # Create plots
    if dim >= 3:
        return px.scatter_3d(
            df, x="x", y="y", z="z", hover_name="ID", size="size", color="label", symbol="type"
        )
    elif dim == 2:
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )

        if resolution > 0 and predictor is not None:
            fig = add_heatmap(points, resolution, predictor, fig)

        return fig
    else:
        df["y"] = [0]*len(df["x"])
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )
