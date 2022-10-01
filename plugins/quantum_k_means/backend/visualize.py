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


def plot_data(data_points, clusters, only_first_100=True):
    dim = len(data_points[0]["point"])
    cluster_dict = {}
    for cluster_entry in clusters:
        cluster_dict[cluster_entry["ID"]] = int(cluster_entry["cluster"])
    if dim == 3:
        points_x = []
        points_y = []
        points_z = []
        ids = []
        cluster = []
        if only_first_100:
            for i in range(min(len(data_points), 100)):
                entity = data_points[i]
                point = entity["point"]
                points_x.append(float(point[0]))
                points_y.append(float(point[1]))
                points_z.append(float(point[2]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        else:
            for entity in data_points:
                point = entity["point"]
                points_x.append(float(point[0]))
                points_y.append(float(point[1]))
                points_z.append(float(point[2]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "z": points_z,
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
    elif dim == 2:
        points_x = []
        points_y = []
        ids = []
        cluster = []
        if only_first_100:
            for i in range(min(len(data_points), 100)):
                entity = data_points[i]
                point = entity["point"]
                points_x.append(float(point[0]))
                points_y.append(float(point[1]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        else:
            for entity in data_points:
                point = entity["point"]
                points_x.append(float(point[0]))
                points_y.append(float(point[1]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        df = pd.DataFrame(
            {
                "x": points_x,
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
    else:
        points_x = []
        ids = []
        cluster = []
        if only_first_100:
            for i in range(min(len(data_points), 100)):
                entity = data_points[i]
                point = entity["point"]
                points_x.append(float(point[0]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        else:
            for entity in data_points:
                point = entity["point"]
                points_x.append(float(point[0]))
                ids.append(entity["ID"])
                cluster.append(cluster_dict[entity["ID"]])
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": [0] * len(ids),
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
