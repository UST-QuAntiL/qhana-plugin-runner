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
import plotly.graph_objects as go
from tempfile import SpooledTemporaryFile
from celery.utils.log import get_task_logger
from requests.exceptions import HTTPError
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from . import HistogramVisualization


TASK_LOGGER = get_task_logger(__name__)


class ImageNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(name=f"{HistogramVisualization.instance.identifier}.generate_image", bind=True)
def generate_image(self, data_url: str, hash: str) -> str:

    TASK_LOGGER.info(f"Generating histogram plot for data in {data_url}...")
    try:
        with open_url(data_url) as url:
            data = url.json()
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Data URL: {data_url}")
        DataBlob.set_value(
            HistogramVisualization.instance.identifier,
            hash,
            "",
        )
        PluginState.delete_value(HistogramVisualization.instance.identifier, hash, commit=True)
        return "Invalid Entity URL!"
    
    x_array = []
    y_array = []

    for label in data:
        if label != "ID":
            x_array.append(label)
            y_array.append(int(data[label]))
    
    df = pd.DataFrame(
        {
            "Values": x_array,
            "Counts": y_array,
        }
    )

    fig = px.histogram(df, x="Values", y="Counts", color="Values", text_auto=True)
    fig.update_traces(showlegend=False) 

    html_bytes = str.encode(fig.to_html(full_html=False), encoding="utf-8")

    DataBlob.set_value(HistogramVisualization.instance.identifier, hash, html_bytes)
    PluginState.delete_value(HistogramVisualization.instance.identifier, hash, commit=True)

    return "Created image of plot!"


@CELERY.task(
    name=f"{HistogramVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(ImageNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(self, db_id: str, data_url: str, hash: str) -> str:
    print("\n\n-------------------------1-------------------------\n\n")
    if not (image := DataBlob.get_value(HistogramVisualization.instance.identifier, hash)):
        if not (
            task_id := PluginState.get_value(HistogramVisualization.instance.identifier, hash)
        ):
            print("--------------------------2---------------------")
            with open_url(data_url) as url:
                data = url.json()
            for d in data:
                print(d)
            task_result = generate_image.s(data_url, hash).apply_async()
            PluginState.set_value(
                HistogramVisualization.instance.identifier,
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
    return "Created image of circuit!"
