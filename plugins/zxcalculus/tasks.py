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
from . import ZXCalculus
from .schemas import InputParameters, InputParametersSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url, retrieve_filename

from qhana_plugin_runner.storage import STORE

import pyzx as zx
import matplotlib.pyplot as _, mpld3

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(name=f"{ZXCalculus.instance.identifier}.visualization_task", bind=True)
def visualization_task(self, db_id: int) -> str:

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    qubits = input_params.qubits
    depth = input_params.depth
    simplify = input_params.simplify

    circuit = zx.generate.cliffordT(qubits, depth)
    zx.settings.drawing_backend = 'd3'
    fig = zx.draw(circuit)
    html = mpld3.fig_to_html(fig)
    if simplify: 
        zx.simplify.full_reduce(circuit)
        circuit.normalize()
        fig = zx.draw(circuit)
        html = mpld3.fig_to_html(fig)

    with SpooledTemporaryFile(mode="wt") as output:
        output.write(html)

        STORE.persist_task_result(
            db_id,
            output,
            f"ZXCalculus_Circuit_{qubits}Qubits_{depth}Depth.html",
            "circuit",
            "text/html",
        )

    return "Result stored in file"
