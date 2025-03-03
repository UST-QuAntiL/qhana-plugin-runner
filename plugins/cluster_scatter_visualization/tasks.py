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

from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Dict, Optional, Tuple, Union

import muid
import pandas as pd
import plotly.express as px
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import ClusterScatterVisualization

TASK_LOGGER = get_task_logger(__name__)


class PlotNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def _get_plot(
    entity_url: str, clusters_url: Optional[str], full_html: bool
) -> Tuple[str, str]:
    """Generate a scatter plot from the given url.

    Args:
        entity_url (str): the url containing entity coordinates
        clusters_url (str|None): an optional url containing entity cluster labels
        full_html (bool): if True, produce a standalone html page, else produce
            an embeddable html snippet.

    Returns:
        Tuple[str, str]: html_content, filename
    """
    is_3d = False
    entities: Dict[str, Dict[str, Union[str, float, int, None]]] = {}
    with open_url(entity_url) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype.")
        name = retrieve_filename(response)
        for ent in ensure_array(load_entities(response, mimetype=mimetype)):
            diagram_ent: Dict[str, Union[str, float, int, None]] = {
                "ID": ent.ID,
                "name": ent.ID,
            }
            dim = len(ent.values)
            if dim == 0:
                raise ValueError("Cannot produce scatter plot without coordinates.")
            diagram_ent["x"] = ent.values[0]
            if dim == 1:
                diagram_ent["y"] = 0
            if dim > 1:
                diagram_ent["y"] = ent.values[1]
            if dim > 2:
                diagram_ent["z"] = ent.values[2]
                is_3d = True
            diagram_ent["href"] = ent.href
            entities[ent.ID] = diagram_ent

    if clusters_url:
        with open_url(clusters_url) as response:
            mimetype = get_mimetype(response)
            if mimetype is None:
                raise ValueError("Could not determine mimetype.")
            label_column = "label"
            for ent in ensure_dict(load_entities(response, mimetype=mimetype)):
                diag_ent: Optional[Dict[str, Union[str, float, int, None]]] = (
                    entities.get(ent["ID"], None)
                )
                if diag_ent is None:
                    continue
                if ent["href"]:
                    diag_ent["href"] = ent["href"]
                if label_column not in ent:
                    entity_columns = list(ent.keys() - {"ID", "href"})
                    if len(entity_columns) != 1:
                        raise ValueError(
                            f"Unable to determine label column from {entity_columns}!"
                        )
                    label_column = entity_columns[0]
                diag_ent["label"] = str(ent[label_column])

    ent_list = list(entities.values())

    df = pd.DataFrame(
        {
            "ID": [e["ID"] for e in ent_list],
            "name": [e["name"] for e in ent_list],
            "URL": [e["href"] for e in ent_list],
            "x": [e["x"] for e in ent_list],
            "y": [e["y"] for e in ent_list],
            "z": [e["z"] if is_3d else 0 for e in ent_list],
            "Cluster ID": [e.get("label", "0") for e in ent_list],
            "size": [10 for _ in ent_list],
        }
    )

    if is_3d:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            size="size",
            hover_name="name",
            color="Cluster ID",
            symbol="Cluster ID",
            hover_data={"size": False},
        )
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            size="size",
            hover_name="name",
            color="Cluster ID",
            symbol="Cluster ID",
            hover_data={"size": False},
        )

    return fig.to_html(full_html=full_html), Path(name).stem


@CELERY.task(
    name=f"{ClusterScatterVisualization.instance.identifier}.generate_plot", bind=True
)
def generate_plot(self, entity_url: str, clusters_url: Optional[str], hash_: str) -> str:

    TASK_LOGGER.info(
        f"Generating plot for entites {entity_url} and clusters {clusters_url}..."
    )
    diagram, _ = _get_plot(
        entity_url=entity_url, clusters_url=clusters_url, full_html=False
    )

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = diagram.encode(encoding="utf-8")

    DataBlob.set_value(ClusterScatterVisualization.instance.identifier, hash_, html_bytes)
    PluginState.delete_value(
        ClusterScatterVisualization.instance.identifier, hash_, commit=True
    )

    return "Created plot!"


@CELERY.task(name=f"{ClusterScatterVisualization.instance.identifier}.process", bind=True)
def process(self, db_id: str, entity_url: str, clusters_url: str) -> str:
    diagram, name = _get_plot(
        entity_url=entity_url, clusters_url=clusters_url, full_html=True
    )

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = diagram.encode(encoding="utf-8")

    with SpooledTemporaryFile() as output:
        output.write(html_bytes)
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"plot_{name}.html", "image/html", "text/html"
        )
    return "Created plot!"
