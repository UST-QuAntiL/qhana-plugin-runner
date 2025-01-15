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
from tempfile import SpooledTemporaryFile
from celery.utils.log import get_task_logger
from requests.exceptions import HTTPError
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from . import ConfusionMatrixVisualization
import numpy as np
from scipy.optimize import linear_sum_assignment

TASK_LOGGER = get_task_logger(__name__)


class ImageNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(
    name=f"{ConfusionMatrixVisualization.instance.identifier}.generate_table", bind=True
)
def generate_table(
    self, clusters_url1: str, clusters_url2: str, optimize: bool, hash: str
) -> str:

    TASK_LOGGER.info(
        f"Generating table for clusters {clusters_url1} and clusters {clusters_url2}..."
    )
    try:
        with open_url(clusters_url1) as url:
            clusters1 = url.json()
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Cluster URL: {clusters_url1}")
        DataBlob.set_value(
            ConfusionMatrixVisualization.instance.identifier,
            hash,
            "",
        )
        PluginState.delete_value(
            ConfusionMatrixVisualization.instance.identifier, hash, commit=True
        )
        return "Invalid Entity URL!"

    try:
        with open_url(clusters_url2) as url:
            clusters2 = url.json()
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Cluster URL: {clusters_url2}")
        DataBlob.set_value(
            ConfusionMatrixVisualization.instance.identifier,
            hash,
            "",
        )
        PluginState.delete_value(
            ConfusionMatrixVisualization.instance.identifier, hash, commit=True
        )
        return "Invalid Cluster URL!"

    confusion_dict = dict()
    wrong_ids = 0
    label_list = []

    for cl in clusters1:
        confusion_dict[cl["ID"]] = [cl["label"]]
        if cl["label"] not in label_list:
            label_list += [cl["label"]]

    for cl in clusters2:
        if cl["ID"] in confusion_dict:
            confusion_dict[cl["ID"]] += [cl["label"]]
        else:
            wrong_ids += 1

    label_list.sort()
    confusion_matrix = [[0 for _ in label_list] for _ in label_list]

    for labels in confusion_dict.values():
        if len(labels) < 2:
            wrong_ids += 1
        elif labels[0] >= len(label_list) or labels[1] >= len(label_list):
            wrong_ids += 1
        else:
            confusion_matrix[label_list.index(labels[0])][
                label_list.index(labels[1])
            ] += 1

    permutation_str = ""
    confusion_matrix = np.array(confusion_matrix)

    if optimize:
        cost_matrix = np.max(confusion_matrix) - confusion_matrix
        _, permutation = linear_sum_assignment(cost_matrix)
        confusion_matrix = confusion_matrix[:, permutation]
        permutation_str = (
            "To achieve an optimal confusion matrix, the column order is now the following: "
            + str(permutation[0])
        )
        permutation = permutation[1:]
        for i in permutation:
            permutation_str += ", " + str(i)

    context = {
        "confusion_matrix": confusion_matrix,
        "wrong_ids": wrong_ids,
        "permutation": permutation_str,
    }
    context_bytes = str.encode(str(context).replace("'", '"'), encoding="utf-8")

    DataBlob.set_value(
        ConfusionMatrixVisualization.instance.identifier, hash, context_bytes
    )
    PluginState.delete_value(
        ConfusionMatrixVisualization.instance.identifier, hash, commit=True
    )

    return "Created table!"


@CELERY.task(
    name=f"{ConfusionMatrixVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(ImageNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(self, db_id: str, entity_url: str, clusters_url: str, hash: str) -> str:
    if not (
        image := DataBlob.get_value(
            ConfusionMatrixVisualization.instance.identifier, hash
        )
    ):
        if not (
            task_id := PluginState.get_value(
                ConfusionMatrixVisualization.instance.identifier, hash
            )
        ):
            task_result = generate_table.s(entity_url, clusters_url, hash).apply_async()
            PluginState.set_value(
                ConfusionMatrixVisualization.instance.identifier,
                hash,
                task_result.id,
                commit=True,
            )
        raise ImageNotFinishedError()
    with SpooledTemporaryFile() as output:
        output.write(image)
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"circuit_{hash}.svg", "image/svg", "image/svg+xml"
        )
    return "Created Confusion Matrix Table!"
