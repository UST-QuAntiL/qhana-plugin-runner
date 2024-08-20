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


from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

import muid
from sklearn import svm
from . import ClusterSVM
from .schemas import InputParameters, InputParametersSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url, retrieve_filename

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(name=f"{ClusterSVM.instance.identifier}.visualization_task", bind=True)
def visualization_task(self, db_id: int) -> str:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_url = input_params.entity_url
    clusters_url = input_params.clusters_url
    do_svm = input_params.do_svm
    do_3d = input_params.do_3d
    TASK_LOGGER.info(f"Loaded input parameters from db: '{str(input_params)}'")

    # load data from file
    entity_points = open_url(entity_url).json()
    clusters = open_url(clusters_url).json()

    print(entity_url)

    pt_x_list = [0 for _ in range(0, len(entity_points))]
    pt_y_list = [0 for _ in range(0, len(entity_points))]
    label_list = [0 for _ in range(0, len(entity_points))]
    id_list = [x for x in range(0, len(entity_points))]
    size_list = [10 for _ in range(0, len(entity_points))]
    max_cluster = 0

    for pt in entity_points:
        idx = int(pt["ID"])
        pt_x_list[idx] = pt["dim0"]
        pt_y_list[idx] = pt["dim1"]

    for cl in clusters:
        label_list[int(cl["ID"])] = cl["label"]
        max_cluster = max(max_cluster, cl["label"])

    df = pd.DataFrame(
        {
            "ID": [f"Point {x}" for x in id_list],
            "x": pt_x_list,
            "y": pt_y_list,
            "Cluster ID": [str(x) for x in label_list],
            "size": size_list,
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="size",
        hover_name="ID",
        color="Cluster ID",
        hover_data={"size": False},
    )

    if do_svm:
        cluster_list = [[] for _ in range(0, max_cluster + 1)]
        for idx, label in enumerate(label_list):
            cluster_list[label].append([pt_x_list[idx], pt_y_list[idx]])

        for i, cl1 in enumerate(cluster_list):
            for j, cl2 in enumerate(cluster_list):
                if i >= j:
                    continue

                cluster_label = [i for _ in range(0, len(cl1))]
                cluster_label = cluster_label + [j for _ in range(0, len(cl2))]
                cl = cl1 + cl2

                clf = svm.SVC(kernel="linear")
                clf.fit(cl, cluster_label)

                a = -clf.coef_[0][0] / clf.coef_[0][1]
                b = clf.intercept_[0] / clf.coef_[0][1]

                x_range = [min(pt_x_list), max(pt_x_list)]

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[a * x - b for x in x_range],
                        mode="lines",
                        name=f"SVM for cluster {i} and {j}",
                        hoveron="fills",
                    )
                )

                y_max = max(pt_y_list)
                y_min = min(pt_y_list)
                padding = (y_max - y_min) * 0.05
                fig.update_layout(yaxis=dict(range=[y_min - padding, y_max + padding]))

    # fig.update_layout(showlegend=False)

    filenames_hash = get_readable_hash(retrieve_filename(clusters_url))

    info_str = f"_cluster-svm_{filenames_hash}"

    with SpooledTemporaryFile(mode="wt") as output:
        html = fig.to_html(include_plotlyjs="cdn")
        output.write(html)

        STORE.persist_task_result(
            db_id,
            output,
            f"plot{info_str}.html",
            "plot",
            "text/html",
        )

    return "Result stored in file"
