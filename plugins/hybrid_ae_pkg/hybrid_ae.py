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
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Dict, Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "hybrid-autoencoder"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HA_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Hybrid Autoencoder plugin API.",
    template_folder="hybrid_ae_templates",
)


class HybridAutoencoderTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HybridAutoencoderPennylaneRequestSchema(FrontendFormBaseSchema):
    input_data = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Input Data",
            "description": "URL to the input data.",
            "input_type": "text",
        },
    )
    number_of_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of qubits that will be used.",
            "input_type": "text",
        },
    )
    embedding_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Embedding Size",
            "description": "Size the embeddings will have (number of values).",
            "input_type": "text",
        },
    )
    qnn_name = ma.fields.String(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "QNN Name",
            "description": "Name of the QNN that will be used.",
            "input_type": "text",
        },
    )
    training_steps = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Training Steps",
            "description": "Number of training steps",
            "input_type": "text",
        },
    )


@HA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Hybrid Autoencoder",
            description="Reduces the dimensionality of a given dataset with a combination of classical and quantum neural networks.",
            name=HybridAutoencoderPlugin.instance.identifier,
            version=HybridAutoencoderPlugin.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
                ui_href=url_for(f"{HA_BLP.name}.MicroFrontend"),
                data_input=[
                    DataMetadata(
                        data_type="real-valued-entities",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="real-valued-entities",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=["dimensionality-reduction"],
        )


@HA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hybrid autoencoder plugin."""

    example_inputs = {
        "inputData": "data:text/plain,0,0,0,0,0,0,0,0,0,0",
        "numberOfQubits": 3,
        "embeddingSize": 2,
        "qnnName": "QNN3",
        "trainingSteps": 100,
    }

    @HA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hybrid autoencoder plugin."
    )
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @HA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hybrid autoencoder plugin."
    )
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        print(">>>", errors)
        schema = HybridAutoencoderPennylaneRequestSchema()
        return Response(
            render_template(
                "hybrid_ae_template.html",
                name=HybridAutoencoderPlugin.instance.name,
                version=HybridAutoencoderPlugin.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
                example_values=url_for(
                    f"{HA_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@HA_BLP.route("/process/pennylane/")
class HybridAutoencoderPennylaneAPI(MethodView):
    """Start a long running processing task."""

    @HA_BLP.response(HTTPStatus.OK, HybridAutoencoderTaskResponseSchema)
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(unknown=EXCLUDE), location="form"
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, req_dict):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=hybrid_autoencoder_pennylane_task.name,
            parameters=dumps(req_dict),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = hybrid_autoencoder_pennylane_task.s(
            db_id=db_task.id
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )


class HybridAutoencoderPlugin(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HA_BLP

    def get_requirements(self) -> str:
        # return "git+ssh://git@github.com/UST-QuAntiL/MuseEmbeddings.git@6cc2f18fdd6b9483d5aaa68d12f8e01cb6329dde#egg=hybrid_autoencoders"
        # TODO: remove dependency on the MuseEmbeddings project
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{HybridAutoencoderPlugin.instance.identifier}.pennylane_hybrid_autoencoder_task",
    bind=True,
)
def hybrid_autoencoder_pennylane_task(self, db_id: int) -> str:
    import numpy as np
    from hybrid_autoencoders import simple_api

    TASK_LOGGER.info(
        f"Starting new hybrid autoencoder pennylane task with db id '{db_id}'"
    )
    task_data: ProcessingTask = DB.session.execute(
        select(ProcessingTask).filter_by(id=db_id)
    ).scalar_one()

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_data_url: str = params.get("input_data", None)
    q_num: int = params.get("number_of_qubits", None)
    embedding_size: int = params.get("embedding_size", None)
    qnn_name: str = params.get("qnn_name", None)
    steps: int = params.get("training_steps", None)

    TASK_LOGGER.info(
        f"input_data: {input_data_url}, q_num: {q_num}, embedding_size: {embedding_size}, qnn_name: {qnn_name}, steps: {steps}"
    )

    if None in [input_data_url, q_num, embedding_size, qnn_name, steps]:
        raise ValueError("Request is missing one or more values.")

    with open_url(input_data_url, stream=True) as url_data:
        input_data_arr = np.genfromtxt(url_data.iter_lines(), delimiter=",")

    if input_data_arr.ndim == 1:
        input_data_arr = input_data_arr.reshape((1, -1))

    output_arr = simple_api.pennylane_hybrid_autoencoder(
        input_data_arr, q_num, embedding_size, qnn_name, steps
    )

    with SpooledTemporaryFile(mode="w") as output:
        np.savetxt(output, output_arr, delimiter=",")
        STORE.persist_task_result(
            db_id, output, "out.csv", "autoencoder-result", "text/csv"
        )
        output.seek(
            0
        )  # TODO remove separate output if task output is already persisted as file
        return "".join(output.readlines())
