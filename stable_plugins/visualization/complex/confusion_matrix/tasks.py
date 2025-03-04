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


from tempfile import SpooledTemporaryFile
from typing import Dict

import muid
import numpy as np
from celery.utils.log import get_task_logger
from flask.templating import render_template
from markupsafe import Markup
from requests.exceptions import HTTPError
from scipy.optimize import linear_sum_assignment

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE

from . import ConfusionMatrixVisualization

TASK_LOGGER = get_task_logger(__name__)


class ImageNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def get_confucion_matrix(  # noqa: C901
    clusters_url1: str, clusters_url2: str, optimize: bool
) -> str:
    original_clusters = {}
    predicted_clusters = {}

    with open_url(clusters_url1) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype.")
        label_column = "label"
        for ent in ensure_dict(load_entities(response, mimetype=mimetype)):
            if label_column not in ent:
                columns = list(ent.keys() - {"ID", "href"})
                if len(columns) != 1:
                    raise ValueError("Could not extract label column from entity data.")
                label_column = columns[0]
            original_clusters[ent["ID"]] = ent[label_column]
    with open_url(clusters_url2) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype.")
        for ent in ensure_dict(load_entities(response, mimetype=mimetype)):
            if label_column not in ent:
                columns = list(ent.keys() - {"ID", "href"})
                if len(columns) != 1:
                    raise ValueError("Could not extract label column from entity data.")
                label_column = columns[0]
            predicted_clusters[ent["ID"]] = ent[label_column]

    # confusion_dict has the ID as the key, and a list of one or two labels as the value
    confusion_dict = {
        k: (v, predicted_clusters.get(k)) for k, v in original_clusters.items()
    }
    wrong_ids = len(predicted_clusters.keys() - original_clusters.keys()) + len(
        original_clusters.keys() - predicted_clusters.keys()
    )
    labels = {l for l in original_clusters.values()}

    label_list = sorted(labels)
    label_index = {l: i for i, l in enumerate(label_list)}

    # Confusion matrix is a two dimensional list
    confusion_matrix = [[0 for _ in label_list] for _ in label_list]

    for original, predicted in confusion_dict.values():
        # ID was only present in clusters2 but not in the first
        if predicted not in labels:
            wrong_ids += 1
        # Add the value to the confusion_matrix
        else:
            confusion_matrix[label_index[original]][label_index[predicted]] += 1

    label_permutation: Dict[str, str] = {l: l for l in label_list}
    confusion_matrix = np.array(confusion_matrix)

    if optimize:
        cost_matrix = np.max(confusion_matrix) - confusion_matrix
        # Linear_sum_assignment solves the optimization problem
        _, permutation = linear_sum_assignment(cost_matrix)
        confusion_matrix = confusion_matrix[:, permutation]
        label_permutation = {l: label_list[i] for l, i in zip(label_list, permutation)}

    return render_template(
        "confusion_matrix_table.html",
        confusion_matrix=confusion_matrix,
        label_list=label_list,
        wrong_ids=wrong_ids,
        permutation=label_permutation,
    )


@CELERY.task(
    name=f"{ConfusionMatrixVisualization.instance.identifier}.generate_table", bind=True
)
def generate_table(
    self, clusters_url1: str, clusters_url2: str, optimize: bool, hash_: str
) -> str:

    TASK_LOGGER.info(
        f"Generating table for clusters {clusters_url1} and clusters {clusters_url2}..."
    )
    try:
        confusion_matrix_html = get_confucion_matrix(
            clusters_url1=clusters_url1, clusters_url2=clusters_url2, optimize=optimize
        )
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Cluster URL: {clusters_url1}")
        DataBlob.set_value(
            ConfusionMatrixVisualization.instance.identifier,
            hash_,
            b"",
        )
        PluginState.delete_value(
            ConfusionMatrixVisualization.instance.identifier, hash_, commit=True
        )
        return "Invalid Entity URL!"

    DataBlob.set_value(
        ConfusionMatrixVisualization.instance.identifier,
        hash_,
        confusion_matrix_html.encode(encoding="utf-8"),
    )
    PluginState.delete_value(
        ConfusionMatrixVisualization.instance.identifier, hash_, commit=True
    )

    return "Created table!"


@CELERY.task(
    name=f"{ConfusionMatrixVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(ImageNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(
    self, db_id: str, clusters_url1: str, clusters_url2: str, optimize: bool, hash_: str
) -> str:
    if not (
        table := DataBlob.get_value(
            ConfusionMatrixVisualization.instance.identifier, hash_
        )
    ):
        if not PluginState.get_value(
            ConfusionMatrixVisualization.instance.identifier, hash_
        ):
            task_result = generate_table.s(
                clusters_url1, clusters_url2, optimize, hash_
            ).apply_async()
            PluginState.set_value(
                ConfusionMatrixVisualization.instance.identifier,
                hash_,
                task_result.id,
                commit=True,
            )
        raise ImageNotFinishedError()
    standalone_html = render_template(
        "confusion_matrix_standalone.html", table=Markup(table.decode(encoding="utf-8"))
    )
    with SpooledTemporaryFile() as output:
        # Data is sligtly changed when it is saved, so it needs to be adjusted a bit
        output.write(standalone_html.encode(encoding="utf-8"))
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"table_{hash_}.html", "table/html", "text/html"
        )
    return "Created Confusion Matrix Table!"
