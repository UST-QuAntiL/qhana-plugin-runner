import pandas as pd
import plotly.express as px


def plot_data(train_data, train_id_to_idx, train_labels, test_data, test_id_to_idx, test_labels, only_first_100=True):
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
        label = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(id)
            label.append(f"{int(train_labels[idx])}")
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(id)
            label.append(f"{int(test_labels[idx])}")
            type.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "z": points_z,
                "ID": ids,
                "size": [10]*len(ids),
                "label": label,
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
        label = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(id)
            label.append(f"{int(train_labels[idx])}")
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(id)
            label.append(f"{int(test_labels[idx])}")
            type.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
                "label": label,
                "type": type,
            }
        )
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )
    else:
        points_x = []
        ids = []
        label = []
        type = []
        for id, idx in train_id_to_idx.items():
            if idx >= train_end:
                break
            point = train_data[idx]
            points_x.append(float(point[0]))
            ids.append(id)
            label.append(f"{int(train_labels[idx])}")
            type.append('train')
        for id, idx in test_id_to_idx.items():
            if idx >= test_end:
                break
            point = test_data[idx]
            points_x.append(float(point[0]))
            ids.append(id)
            label.append(f"{int(test_labels[idx])}")
            type.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": [0] * len(ids),
                "ID": ids,
                "size": [10] * len(ids),
                "label": label,
                "type": type,
            }
        )
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="type"
        )