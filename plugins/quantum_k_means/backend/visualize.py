# Copyright 2022 QHAna plugin runner contributors.
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


def plot_data(data_points, clusters, only_first_100=True):
    if only_first_100:
        data_points = data_points[:100]

    dim = len(data_points[0]["point"])
    cluster_dict = {}
    for cluster_entry in clusters:
        cluster_dict[cluster_entry["ID"]] = int(cluster_entry["cluster"])

    points = np.empty((len(data_points), dim))
    ids = []
    cluster = []
    for i, entity in enumerate(data_points):
        points[i] = np.array(entity["point"], dtype=float)
        ids.append(entity["ID"])
        cluster.append(cluster_dict[entity["ID"]])

    if dim >= 3:
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "ID": ids,
                "size": [10] * len(ids),
                "cluster": cluster,
            }
        )
        return px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            hover_name="ID",
            size="size",
            color="cluster",
            symbol="cluster",
        )
    else:
        if dim == 2:
            points_y = points[:, 1]
        else:
            points_y = [0] * len(points)
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
                "cluster": cluster,
            }
        )
        return px.scatter(
            df,
            x="x",
            y="y",
            hover_name="ID",
            size="size",
            color="cluster",
            symbol="cluster",
        )
