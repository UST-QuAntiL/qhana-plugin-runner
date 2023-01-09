import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix


def get_id_list(id_to_idx: dict) -> list:
    ids = ["id"] * len(id_to_idx)
    for id, idx in id_to_idx.items():
        ids[idx] = id
    return ids


def plot_data(
    train_data,
    train_id_to_idx,
    train_labels,
    test_data,
    test_id_to_idx,
    test_labels,
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


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred).T

    df_content = dict()
    for idx, v in enumerate(conf_matrix):
        df_content[str(idx)] = [str(el) for el in v]
    df = pd.DataFrame(df_content)
    df.index = pd.Index([str(idx) for idx in range(len(conf_matrix))], name="True label")
    df.columns = pd.Index(
        [str(idx) for idx in range(len(conf_matrix))], name="Predicted label"
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
