import pandas as pd
import plotly.express as px


def plot_data(train_data, train_labels, test_data, test_labels, only_first_100=True):
    dim = len(train_data[0]["point"])
    label_dict = {}
    for label_entry in train_data:
        label_dict[label_entry["ID"]] = int(label_entry["label"])
    for label_entry in test_labels:
        label_dict[label_entry["ID"]] = int(label_entry["label"])
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
        test_or_train = []
        for i in range(train_end):
            entity = train_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append('train')
        for i in range(test_end):
            entity = test_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            points_z.append(float(point[2]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "z": points_z,
                "ID": ids,
                "size": [10]*len(ids),
                "label": label,
                "test_or_train": test_or_train,
            }
        )
        return px.scatter_3d(
            df, x="x", y="y", z="z", hover_name="ID", size="size", color="label", symbol="test_or_train"
        )
    elif dim == 2:
        points_x = []
        points_y = []
        ids = []
        label = []
        test_or_train = []
        for i in range(train_end):
            entity = train_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append('train')
        for i in range(test_end):
            entity = test_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            points_y.append(float(point[1]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append('test')
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
                "label": label,
                "test_or_train": test_or_train,
            }
        )
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="test_or_train"
        )
    else:
        points_x = []
        ids = []
        label = []
        test_or_train = []
        for i in range(train_end):
            entity = train_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append("train")
        for i in range(test_end):
            entity = test_data[i]
            point = entity["point"]
            points_x.append(float(point[0]))
            ids.append(entity["ID"])
            label.append(label_dict[entity["ID"]])
            test_or_train.append("test")
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": [0] * len(ids),
                "ID": ids,
                "size": [10] * len(ids),
                "label": label,
                "test_or_train": test_or_train,
            }
        )
        return px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label", symbol="test_or_train"
        )
