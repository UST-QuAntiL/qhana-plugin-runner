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
from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np

# import muid
from celery.utils.log import get_task_logger
from requests import HTTPError

from ripser import ripser
from persim import plot_diagrams

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    load_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import TDAVisualization

TASK_LOGGER = get_task_logger(__name__)


class PlotNotFinishedError(Exception):
    pass


# def get_readable_hash(s: str) -> str:
#     return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")

def _get_plot_as_html(
        persistence_dgm: List[np.ndarray],full_html: bool
) -> str:
    """Generate a html representation of a given persistence diagram.

    Args:
        persistence_dgm: list of nd.arrays containing the points in the persistence diagram for each 
            homology dimension
        full_html (bool): if True, produce a standalone html page, else produce
            an embeddable html snippet.

    Returns:
        str: html_content
    """
    
    fig = go.Figure()

    for dim, diagram in enumerate(persistence_dgm):
        if len(diagram) == 0:
            continue

        births = diagram[:, 0]
        deaths = diagram[:, 1]

        fig.add_trace(
            go.Scatter(
                x=births,
                y=deaths,
                mode="markers",
                name=f"H{dim}",
            )
        )

    # find maximum death value for axis limits
    max_val = max(
        np.max(diagram[np.isfinite(diagram)])
        for diagram in persistence_dgm
        if len(diagram) > 0
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Diagonal",
        )
    )

    fig.update_layout(
        title="Persistence Diagram",
        xaxis_title="Birth",
        yaxis_title="Death",
    )

    return fig.to_html(full_html=True)


def _get_plot(
    entity_url: str, homology_dimension: Optional[int], full_html: bool
) -> Tuple[str, str]:
    """Generate a persistence diagram from the data given url.

    Args:
        entity_url (str): the url containing entity coordinates
        homology_dimension (int): the homology dimension
        full_html (bool): if True, produce a standalone html page, else produce
            an embeddable html snippet.

    Returns:
        Tuple[str, str]: html_content, filename
    """
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
            diagram_ent["data"] = ent.values
            diagram_ent["href"] = ent.href
            entities[ent.ID] = diagram_ent
    
    data_points = np.array([
        entity["data"] for entity in entities.values()
    ])
    ripser_result = ripser(data_points, maxdim=homology_dimension)
    fig_html = _get_plot_as_html(ripser_result["dgms"], full_html)

    return fig_html, Path(name).stem


@CELERY.task(
    name=f"{TDAVisualization.instance.identifier}.generate_plot", bind=True
)
def generate_plot(self, entity_url: str, homology_dimension: Optional[int], hash_: str) -> str:

    TASK_LOGGER.info(
        f"Generating persistence diagram for entites {entity_url} and homology dimension {homology_dimension}..."
    )

    try:
        diagram, _ = _get_plot(
            entity_url=entity_url, homology_dimension=homology_dimension, full_html=False
        )
    except HTTPError:
        DataBlob.set_value(TDAVisualization.instance.identifier, hash_, b"")
        PluginState.delete_value(
            TDAVisualization.instance.identifier, hash_, commit=True
        )
        return "Invalid Entity URL!"

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = diagram.encode(encoding="utf-8")

    DataBlob.set_value(TDAVisualization.instance.identifier, hash_, html_bytes)
    PluginState.delete_value(
        TDAVisualization.instance.identifier, hash_, commit=True
    )

    return "Created plot!"


@CELERY.task(name=f"{TDAVisualization.instance.identifier}.process", bind=True)
def process(self, db_id: str, entity_url: str, homology_dimension: int) -> str:
    diagram, name = _get_plot(
        entity_url=entity_url, homology_dimension=homology_dimension, full_html=True
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
