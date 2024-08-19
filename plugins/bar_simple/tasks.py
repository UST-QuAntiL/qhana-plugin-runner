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
from json import loads

from celery.utils.log import get_task_logger

import muid
from . import BarDiagram
from .schemas import InputParameters, InputParametersSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url, retrieve_filename

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(name=f"{BarDiagram.instance.identifier}.visualization_task", bind=True)
def visualization_task(self, db_id: int) -> str:
    import pandas as pd
    import plotly.express as px

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    
    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    clusters_url = input_params.clusters_url
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters_url='{clusters_url}'")

    # load data from file

    clusters = open_url(clusters_url).json()

    cluster_list = []
    amount_list = []

    for cl in clusters:
        label = cl["label"]
        if not label in cluster_list:
            cluster_list.append(label)
            amount_list.append(1)
        else:
            amount_list[cluster_list.index(label)] += 1

    df = pd.DataFrame(
        {
            "Cluster ID": cluster_list,
            "Amount": amount_list,
        }
    )

    fig = px.bar(df, x="Cluster ID", y="Amount", color="Cluster ID")
    fig.update_coloraxes(showscale=False)

    filenames_hash = get_readable_hash(retrieve_filename(clusters_url))

    info_str = f"_bar-diagram_{filenames_hash}"

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
