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

import muid
import pyzx as zx
from celery.utils.log import get_task_logger
from matplotlib.figure import Figure
from requests.exceptions import HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from . import ZXCalculusVisualization

TASK_LOGGER = get_task_logger(__name__)


class ImageNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(
    name=f"{ZXCalculusVisualization.instance.identifier}.generate_image", bind=True
)
# Optimized is not required, as both images are generated
def generate_image(self, data_url: str, hash_norm: str, hash_opt: str) -> str:

    TASK_LOGGER.info(f"Generating ZXCalculus circuit for data in {data_url}...")
    # Check that Data_url is correct
    try:
        with open_url(data_url) as url:
            qasm_data = url.text
    except HTTPError:
        TASK_LOGGER.error(f"Invalid Data URL: {data_url}")
        DataBlob.set_value(ZXCalculusVisualization.instance.identifier, hash_norm, b"")
        DataBlob.set_value(ZXCalculusVisualization.instance.identifier, hash_opt, b"")
        PluginState.delete_value(
            ZXCalculusVisualization.instance.identifier, hash_norm, commit=True
        )
        return "Invalid Entity URL!"

    circuit = zx.Circuit.from_qasm(qasm_data)
    graph = circuit.to_graph()

    zx.simplify.full_reduce(graph)
    graph.normalize()

    # Normal figures
    fig_norm: Figure = zx.draw(circuit)
    with SpooledTemporaryFile() as fig:
        fig_norm.savefig(fig, format="svg")
        fig.seek(0)
        svg_norm = fig.read()

    # Optimized figures
    fig_opt: Figure = zx.draw(graph)
    with SpooledTemporaryFile() as fig:
        fig_opt.savefig(fig, format="svg")
        fig.seek(0)
        svg_opt = fig.read()

    DataBlob.set_value(ZXCalculusVisualization.instance.identifier, hash_norm, svg_norm)
    DataBlob.set_value(ZXCalculusVisualization.instance.identifier, hash_opt, svg_opt)
    PluginState.delete_value(
        ZXCalculusVisualization.instance.identifier, hash_norm, commit=True
    )

    return "Created circuits!"


@CELERY.task(
    name=f"{ZXCalculusVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(ImageNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(
    self, db_id: str, data_url: str, hash_norm: str, hash_opt: str, optimized: bool
) -> str:
    if not (
        image := DataBlob.get_value(
            ZXCalculusVisualization.instance.identifier,
            hash_opt if optimized else hash_norm,
        )
    ):
        if not (
            task_id := PluginState.get_value(
                ZXCalculusVisualization.instance.identifier,
                hash_opt if optimized else hash_norm,
            )
        ):
            task_result = generate_image.s(data_url, hash_norm, hash_opt).apply_async()
            PluginState.set_value(
                ZXCalculusVisualization.instance.identifier,
                hash_opt if optimized else hash_norm,
                task_result.id,
                commit=True,
            )
        raise ImageNotFinishedError()
    with SpooledTemporaryFile() as output:
        output.write(image)
        output.seek(0)
        new_hash = hash_opt if optimized else hash_norm
        STORE.persist_task_result(
            db_id, output, f"circuit_{new_hash}.svg", "custom/plot", "image/svg+xml"
        )
    return "Created image of circuit!"
