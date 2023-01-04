import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def plot_data(train_data, train_id_to_idx, train_labels, test_data, test_id_to_idx, test_labels, resolution=0, predictor=None, only_first_100=True):
    dim = len(train_data[0])
    train_end = len(train_data)
    test_end = len(test_data)
    if only_first_100:
        train_end = min(train_end, 100)
        test_end = min(test_end, 100)
    if dim >= 3:
        points_x = []
        points_y = []
        points_z = []
        ids = []
        labels = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(id)
            labels.append(str(int(train_labels[idx])))
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(id)
            labels.append(str(int(test_labels[idx])))
            type.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "z": points_z,
                "ID": ids,
                "size": [10]*len(ids),
                "label": labels,
                "type": type,
            }
        )
        return px.scatter_3d(
            df, x="x", y="y", z="z", hover_name="ID", size="size", color="label", symbol="type"
        )
    elif dim == 2:
        points_x = []
        points_y = []
        ids = []
        labels = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(id)
            labels.append(str(int(train_labels[idx])))
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(id)
            labels.append(str(int(test_labels[idx])))
            type.append("test")

        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
                "label": labels,
                "type": type,
            }
        )
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )


        if resolution > 0 and predictor is not None:
            points_x = np.array(points_x)
            points_y = np.array(points_y)

            # Prep for grid (heatmap)
            x_min = points_x.min()
            x_max = points_x.max()
            y_min = points_y.min()
            y_max = points_y.max()

            x_range = x_max - x_min
            y_range = y_max - y_min
            # Subtract 1 from the resolution, we add it later on again, but doing it like this, fills the whole screen
            # with the heatmap and leaves no blank border
            resolution -= 1
            hx = x_range / resolution
            hy = y_range / resolution

            # get x and y coordinates of a grid.
            # We need to determine the label of each point in the grid to create the heatmap
            x_grid = np.arange(x_min, x_max + hx, hx)   # +hx adds the previously subtracted resolution
            y_grid = np.arange(y_min, y_max + hy, hy)   # +hy adds the previously subtracted resolution

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
            ))

            # Give markers a slightly thicker border, since their background will most likely have the same color.
            # Note, background is due to the heatmap
            fig.update_traces(
                marker=dict(
                    line=dict(
                        width=2.5,
                    )
                ),
            )
            # Correct colors of different labels. 0 gets the first color, 1 the second and so on
            # Thus the color match with the heatmap colors
            for sca_plt in fig.data:
                label = int(sca_plt.legendgroup[0])
                sca_plt.update(
                    marker=dict(
                        color=px.colors.qualitative.D3[label],
                    )
                )

        # Combine both heatmap and scatter plot
        # layout=fig.layout keeps the description of the legend
        fig = go.Figure(data=heatmap.data + fig.data, layout=fig.layout)

        return fig
    else:
        points_x = []
        ids = []
        labels = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            ids.append(id)
            labels.append(str(int(train_labels[idx])))
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            ids.append(id)
            labels.append(str(int(test_labels[idx])))
            type.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": [0] * len(ids),
                "ID": ids,
                "size": [10] * len(ids),
                "label": labels,
                "type": type,
            }
        )
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )
