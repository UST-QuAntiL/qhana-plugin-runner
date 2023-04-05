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

import pandas as pd
import plotly.express as px
import numpy as np


def plot_data(
    points,
    id_list,
    labels,
    only_first_100=True,
    title="",
):
    # Prepare data
    dim = len(points[0])
    data_end = 100 if only_first_100 else len(points)

    points = np.array(points[:data_end])

    dimensions = ["x", "y", "z"][:dim]
    ids = id_list[:data_end]

    labels = [str(el) for el in labels[:data_end]]

    data_frame_content = dict(
        ID=ids,
        size=[10] * len(ids),
        label=labels,
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
        )
    elif dim == 2:
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label"
        )
    else:
        df["y"] = [0] * len(df["x"])
        fig = px.scatter(
            df, x="x", y="y", hover_name="ID", size="size", color="label"
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
