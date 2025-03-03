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
from typing import Tuple
from pathlib import Path

import muid
import pandas as pd
import plotly.express as px
from celery.utils.log import get_task_logger
from requests.exceptions import HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.requests import open_url, retrieve_filename, get_mimetype
from qhana_plugin_runner.plugin_utils.entity_marshalling import load_entities, ensure_dict
from qhana_plugin_runner.storage import STORE

from . import HistogramVisualization

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def get_diagram(data_url: str, full_html: bool) -> Tuple[str, str]:
    """Get a rendered html diagram and the file name.

    Args:
        data_url (str): the data to render
        full_html (bool): if True, produce a standalone html page, else produce
            an embeddable html snippet.

    Returns:
        Tuple[str, str]: html_content, filename
    """
    with open_url(data_url) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype.")
        name = retrieve_filename(response)
        data = next(ensure_dict(load_entities(response, mimetype=mimetype)))

    x_array = []
    y_array = []

    # X-Axis are the labels, Y-Axis the counts
    for label, value in data.items():
        if label in ("ID", "href"):
            continue
        x_array.append(label)
        if isinstance(value, str):
            value = int(value)
        y_array.append(value)

    df = pd.DataFrame(
        {
            "Values": x_array,
            "Counts": y_array,
        }
    )

    fig = px.histogram(df, x="Values", y="Counts", color="Values", text_auto=True)
    fig.update_layout(
        yaxis=dict(title=dict(text="Sum of Counts, Total: " + str(sum(y_array))))
    )
    fig.update_traces(showlegend=False)

    return fig.to_html(full_html=full_html), Path(name).stem


@CELERY.task(
    name=f"{HistogramVisualization.instance.identifier}.generate_plot", bind=True
)
def generate_plot(self, data_url: str, hash_: str) -> str:

    TASK_LOGGER.info(f"Generating histogram plot for data in {data_url}...")
    # Check that data_url is correct
    try:
        diagram, _ = get_diagram(data_url, full_html=False)
    except (HTTPError, ValueError):
        TASK_LOGGER.error(f"Invalid Data URL: {data_url}")
        DataBlob.set_value(HistogramVisualization.instance.identifier, hash_, b"")
        PluginState.delete_value(
            HistogramVisualization.instance.identifier, hash_, commit=True
        )
        return "Invalid Entity URL!"

    html_bytes = diagram.encode(encoding="utf-8")

    DataBlob.set_value(HistogramVisualization.instance.identifier, hash_, html_bytes)
    PluginState.delete_value(
        HistogramVisualization.instance.identifier, hash_, commit=True
    )

    return "Created plot!"


@CELERY.task(name=f"{HistogramVisualization.instance.identifier}.process", bind=True)
def process(self, db_id: str, data_url: str) -> str:
    diagram, name = get_diagram(data_url, full_html=True)

    with SpooledTemporaryFile() as output:
        output.write(diagram.encode(encoding="utf-8"))
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"plot_{name}.html", "image/html", "text/html"
        )
    return "Created Plot!"
