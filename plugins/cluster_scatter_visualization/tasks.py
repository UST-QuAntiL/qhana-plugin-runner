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

import muid
import pandas as pd
import plotly.express as px
from tempfile import SpooledTemporaryFile
from celery.utils.log import get_task_logger
from requests.exceptions import HTTPError
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from . import ClusterScatterVisualization


TASK_LOGGER = get_task_logger(__name__)


class PlotNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(
    name=f"{ClusterScatterVisualization.instance.identifier}.generate_plot", bind=True
)
def generate_plot(self, entity_url: str, clusters_url: str, hash: str) -> str:

    TASK_LOGGER.info(
        f"Generating plot for entites {entity_url} and clusters {clusters_url}..."
    )
    try:
        with open_url(entity_url) as url:
            entities = url.json()
    # An entity URL is required, or no plot can be generated
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Entity URL: {entity_url}")
        DataBlob.set_value(
            ClusterScatterVisualization.instance.identifier,
            hash,
            "",
        )
        PluginState.delete_value(
            ClusterScatterVisualization.instance.identifier, hash, commit=True
        )
        return "Invalid Entity URL!"

    clusters = []
    # Only check the cluster url as valid, if one is provided
    if clusters_url is not None:
        try:
            with open_url(clusters_url) as url:
                clusters = url.json()
        except HTTPError:
            TASK_LOGGER.error(f"Invalid Cluster URL: {clusters_url}")
            DataBlob.set_value(
                ClusterScatterVisualization.instance.identifier,
                hash,
                "",
            )
            PluginState.delete_value(
                ClusterScatterVisualization.instance.identifier, hash, commit=True
            )
            return "Invalid Cluster URL!"

    # Setup all necessary lists and the max_cluster counter
    pt_x_list = [0 for _ in range(0, len(entities))]
    pt_y_list = [0 for _ in range(0, len(entities))]
    pt_z_list = [0 for _ in range(0, len(entities))]
    label_list = [0 for _ in range(0, len(entities))]
    id_list = [x for x in range(0, len(entities))]
    size_list = [10 for _ in range(0, len(entities))]
    max_cluster = 0

    # Check if provided data is three-dimensional
    do_3d = "dim2" in entities[0]

    # Load all entity points into their respective lists
    for pt in entities:
        idx = int(pt["ID"])
        pt_x_list[idx] = pt["dim0"]
        pt_y_list[idx] = pt["dim1"]
        if do_3d:
            pt_z_list[idx] = pt["dim2"]

    # Load all clusters in to the label_list and update max_cluster
    for cl in clusters:
        label_list[int(cl["ID"])] = cl["label"]
        max_cluster = max(max_cluster, cl["label"])

    df = pd.DataFrame(
        {
            "ID": [f"Point {x}" for x in id_list],
            "x": pt_x_list,
            "y": pt_y_list,
            "z": pt_z_list,
            "Cluster ID": [str(x) for x in label_list],
            "size": size_list,
        }
    )

    # 2D Plot generation
    if not do_3d:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            size="size",
            hover_name="ID",
            color="Cluster ID",
            hover_data={"size": False},
        )
    # 3D Plot generation
    else:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            size="size",
            hover_name="ID",
            color="Cluster ID",
            hover_data={"size": False},
        )

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = str.encode(fig.to_html(full_html=False), encoding="utf-8")

    DataBlob.set_value(ClusterScatterVisualization.instance.identifier, hash, html_bytes)
    PluginState.delete_value(
        ClusterScatterVisualization.instance.identifier, hash, commit=True
    )

    return "Created plot!"


@CELERY.task(
    name=f"{ClusterScatterVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(PlotNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(self, db_id: str, entity_url: str, clusters_url: str, hash: str) -> str:
    if not (
        plot := DataBlob.get_value(ClusterScatterVisualization.instance.identifier, hash)
    ):
        if not (
            task_id := PluginState.get_value(
                ClusterScatterVisualization.instance.identifier, hash
            )
        ):
            task_result = generate_plot.s(entity_url, clusters_url, hash).apply_async()
            PluginState.set_value(
                ClusterScatterVisualization.instance.identifier,
                hash,
                task_result.id,
                commit=True,
            )
        raise PlotNotFinishedError()
    with SpooledTemporaryFile() as output:
        output.write(plot)
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"plot_{hash}.svg", "image/html", "text/html"
        )
    return "Created plot!"
