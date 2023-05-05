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
import plotly.graph_objects as go
import numpy as np
from networkx import spring_layout
from .max_cut_solver import create_graph


def plot_data(
    points,
    ids,
    labels,
    title="",
):
    # Prepare data
    dim = len(points[0])

    dimensions = ["x", "y", "z"][:dim]

    labels = [str(el) for el in labels]

    data_frame_content = dict(
        ID=ids,
        size=[10] * len(ids),
        label=labels,
    )
    for d, d_name in enumerate(dimensions):
        data_frame_content[d_name] = points[:, d]

    df = pd.DataFrame(data_frame_content)

    # Create plots
    fig = px.scatter(df, x="x", y="y", hover_name="ID", size="size", color="label")

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


def get_width(weight, min_weight=1, max_weight=1, min_width=0.75, max_width=5):
    if min_weight == max_weight:
        return (max_width - min_width) / 2.0
    return (weight - min_weight) / max_weight * (max_width - min_width) + min_width


def make_edge(x, y, text, weight, min_weight=1, max_weight=1, showlegend=False):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+text",
        line=dict(
            width=get_width(weight, min_weight=min_weight, max_weight=max_weight),
            color="cornflowerblue",
        ),
        text=(["", text, ""]),
        name="Edges",
        showlegend=showlegend,
        legendgroup="edges",
    )


def plot_graph(
    adjacency_matrix: np.array,
    labels,
    title="",
):
    min_weight = np.min(adjacency_matrix[np.nonzero(adjacency_matrix)])
    max_weight = np.max(adjacency_matrix[np.nonzero(adjacency_matrix)])
    graph = create_graph(adjacency_matrix)
    positions = np.array(list(spring_layout(graph).values()))

    # Create figure
    fig = go.Figure()

    # For each edge, make an edge_trace, append to list
    showlegend = True
    for edge in graph.edges():
        if graph.edges()[edge]["weight"] != 0:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = positions[char_1]
            x1, y1 = positions[char_2]

            trace = make_edge(
                [x0, (x0 + x1) / 2.0, x1],
                [y0, (y0 + y1) / 2.0, y1],
                str(graph.edges()[edge]["weight"]),
                weight=graph.edges()[edge]["weight"],
                min_weight=min_weight,
                max_weight=max_weight,
                showlegend=showlegend,
            )
            fig.add_trace(trace)
            showlegend = False

    fig.add_traces(
        plot_data(positions, list(range(len(positions))), labels, title=title).data
    )

    fig.update_layout(
        font=dict(size=15),
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
