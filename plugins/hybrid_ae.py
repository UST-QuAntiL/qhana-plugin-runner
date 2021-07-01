# Copyright 2021 QHAna plugin runner contributors.
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

from http import HTTPStatus
from io import StringIO
from json import dumps, loads
from typing import Optional, Dict

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask.app import Flask
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.util import MaBaseSchema, SecurityBlueprint
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from hybrid_autoencoders import simple_api
import numpy as np

_plugin_name = "hybrid-autoencoder"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HA_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Hybrid Autoencoder plugin API.",
    template_folder="hybrid_autoencoder_templates",
)


class HybridAutoencoderResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class HybridAutoencoderTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HybridAutoencoderPennylaneRequestSchema(MaBaseSchema):
    input_data = ma.fields.String(required=True, allow_none=False, load_only=True)
    number_of_qubits = ma.fields.Integer(required=True, allow_none=False, load_only=True)
    embedding_size = ma.fields.Integer(required=True, allow_none=False, load_only=True)
    qnn_name = ma.fields.String(required=True, allow_none=False, load_only=True)
    training_steps = ma.fields.Integer(required=True, allow_none=False, load_only=True)


@HA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HA_BLP.response(HTTPStatus.OK, HybridAutoencoderResponseSchema())
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return {
            "name": HybridAutoencoder.instance.name,
            "version": HybridAutoencoder.instance.version,
            "identifier": HybridAutoencoder.instance.identifier,
        }


@HA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    @HA_BLP.doc(
        responses={
            f"{HTTPStatus.OK}": {
                "description": "Micro frontend of the hello world plugin.",
                "content": {"text/html": {"schema": {"type": "string"}}},
            }
        }
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""
        return Response(
            render_template(
                "hybrid_autoencoder_template.html",
                name=HybridAutoencoder.instance.name,
                version=HybridAutoencoder.instance.version,
            )
        )


@HA_BLP.route("/process/pennylane/")
class HybridAutoencoderPennylaneAPI(MethodView):
    """Start a long running processing task."""

    @HA_BLP.response(HTTPStatus.OK, HybridAutoencoderTaskResponseSchema)
    @HA_BLP.arguments(HybridAutoencoderPennylaneRequestSchema)
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, req_dict):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=hybrid_autoencoder_pennylane_task.name,
            parameters=dumps(req_dict),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = hybrid_autoencoder_pennylane_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return {
            "name": hybrid_autoencoder_pennylane_task.name,
            "task_id": str(result.id),
            "task_result_url": url_for("tasks-api.TaskView", task_id=str(result.id)),
        }


class HybridAutoencoder(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HA_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{HybridAutoencoder.instance.identifier}.pennylane_hybrid_autoencoder_task", bind=True)
def hybrid_autoencoder_pennylane_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new hybrid autoencoder pennylane task with db id '{db_id}'")
    task_data: ProcessingTask = DB.session.execute(
        select(ProcessingTask).filter_by(id=db_id)
    ).scalar_one()

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_data_str: str = params.get("input_data", None)
    q_num: int = params.get("number_of_qubits", None)
    embedding_size: int = params.get("embedding_size", None)
    qnn_name: str = params.get("qnn_name", None)
    steps: int = params.get("training_steps", None)

    TASK_LOGGER.info(f"input_data: {input_data_str}")
    TASK_LOGGER.info(f"q_num: {q_num}")
    TASK_LOGGER.info(f"embedding_size: {embedding_size}")
    TASK_LOGGER.info(f"qnn_name: {qnn_name}")
    TASK_LOGGER.info(f"steps: {steps}")

    if None in [input_data_str, q_num, embedding_size, qnn_name, steps]:
        raise ValueError("Request is missing one or more values.")

    input_data_arr = np.genfromtxt(StringIO(input_data_str), delimiter=",")

    if input_data_arr.ndim == 1:
        input_data_arr = input_data_arr.reshape((1, -1))

    output_arr = simple_api.pennylane_hybrid_autoencoder(input_data_arr, q_num, embedding_size, qnn_name, steps)

    string_buffer = StringIO()
    np.savetxt(string_buffer, output_arr, delimiter=",")

    return string_buffer.getvalue()
