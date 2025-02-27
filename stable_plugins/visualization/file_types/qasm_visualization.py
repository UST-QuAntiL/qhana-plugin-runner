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

import hashlib
from http import HTTPStatus
from io import BytesIO
import pathlib
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional
from celery import chain
import celery
from celery.utils.log import get_task_logger
from flask import abort, redirect, send_file
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE
from requests.exceptions import HTTPError

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState

_plugin_name = "qasm-visualization"
__version__ = "v0.3.1"
_identifier = plugin_identifier(_plugin_name, __version__)


QASM_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A visualization plugin for quantum circuits.",
    template_folder="qasm_visualization_templates",
)


class QasmInputParametersSchema(FrontendFormBaseSchema):
    data = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["text/x-qasm"],
        metadata={
            "label": "QASM File",
            "description": "The URL to a QASM file.",
        },
    )


@QASM_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QASM_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @QASM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = QasmVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.visualization,
            entry_point=EntryPoint(
                href=url_for(f"{QASM_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QASM_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="*",
                        content_type=["text/x-qasm"],
                        parameter="data",
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["image/svg+xml"],
                        required=True,
                    )
                ],
            ),
            tags=plugin.tags,
        )


@QASM_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the QASM visualization plugin."""

    @QASM_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the QASM visualization plugin."
    )
    @QASM_BLP.arguments(
        QasmInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QASM_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @QASM_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the QASM visualization plugin."
    )
    @QASM_BLP.arguments(
        QasmInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QASM_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = QasmVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response(
            render_template(
                "qasm_visualization.html",
                name=plugin.name,
                version=plugin.version,
                schema=QasmInputParametersSchema(),
                valid=valid,
                values=data,
                errors=errors,
                example_values=url_for(f"{QASM_BLP.name}.MicroFrontend"),
                get_circuit_image_url=url_for(f"{QASM_BLP.name}.get_circuit_image"),
                process=url_for(f"{QASM_BLP.name}.ProcessView"),
            )
        )


class ImageNotFinishedError(Exception):
    pass


@QASM_BLP.route("/circuits/")
@QASM_BLP.response(HTTPStatus.OK, description="Circuit image.")
@QASM_BLP.arguments(
    QasmInputParametersSchema(unknown=EXCLUDE),
    location="query",
    required=True,
)
@QASM_BLP.require_jwt("jwt", optional=True)
def get_circuit_image(data: Mapping):
    url = data.get("data", None)
    if not url:
        abort(HTTPStatus.BAD_REQUEST)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    image = DataBlob.get_value(QasmVisualization.instance.identifier, url_hash, None)
    if image is None:
        if not (
            task_id := PluginState.get_value(
                QasmVisualization.instance.identifier, url_hash, None
            )
        ):
            task_result = generate_image.s(url, url_hash).apply_async()
            PluginState.set_value(
                QasmVisualization.instance.identifier,
                url_hash,
                task_result.id,
                commit=True,
            )
        else:
            task_result = CELERY.AsyncResult(task_id)
        try:
            task_result.get(timeout=5)
            image = DataBlob.get_value(QasmVisualization.instance.identifier, url_hash)
        except celery.exceptions.TimeoutError:
            return Response("Image not yet created!", HTTPStatus.ACCEPTED)
    if not image:
        abort(HTTPStatus.BAD_REQUEST, "Invalid circuit URL!")
    return send_file(BytesIO(image), mimetype="image/svg+xml")


@QASM_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @QASM_BLP.arguments(QasmInputParametersSchema(unknown=EXCLUDE), location="form")
    @QASM_BLP.response(HTTPStatus.SEE_OTHER)
    @QASM_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        circuit_url = arguments.get("data", None)
        if circuit_url is None:
            abort(HTTPStatus.BAD_REQUEST)
        url_hash = hashlib.sha256(circuit_url.encode("utf-8")).hexdigest()
        db_task = ProcessingTask(task_name=process.name)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = process.s(
            db_id=db_task.id, url=circuit_url, hash=url_hash
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async(db_id=db_task.id, url=circuit_url, hash=url_hash)

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class QasmVisualization(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Visualizes QASM data."
    tags = ["visualization", "qasm"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        # create folder for circuit images
        pathlib.Path(__file__).parent.absolute().joinpath("img").mkdir(
            parents=True, exist_ok=True
        )

    def get_api_blueprint(self):
        return QASM_BLP

    def get_requirements(self) -> str:
        return "pylatexenc~=2.10\nqiskit~=0.43"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QasmVisualization.instance.identifier}.generate_image", bind=True)
def generate_image(self, url: str, hash: str) -> str:
    from qiskit import QuantumCircuit
    import matplotlib

    matplotlib.use("SVG")

    TASK_LOGGER.info(f"Generating image for circuit {url}...")
    try:
        with open_url(url) as qasm_response:
            circuit_qasm = qasm_response.text
    except HTTPError:
        TASK_LOGGER.error(f"Invalid circuit URL: {url}")
        DataBlob.set_value(
            QasmVisualization.instance.identifier,
            hash,
            "",
        )
        PluginState.delete_value(QasmVisualization.instance.identifier, hash, commit=True)
        return "Invalid circuit URL!"

    circuit = QuantumCircuit.from_qasm_str(circuit_qasm)
    fig = circuit.draw(output="mpl", interactive=False)
    figfile = BytesIO()
    fig.savefig(figfile, format="svg")
    figfile.seek(0)
    DataBlob.set_value(QasmVisualization.instance.identifier, hash, figfile.getvalue())
    TASK_LOGGER.info(f"Stored image of circuit {circuit.name}.")
    PluginState.delete_value(QasmVisualization.instance.identifier, hash, commit=True)

    return "Created image of circuit!"


@CELERY.task(
    name=f"{QasmVisualization.instance.identifier}.process",
    bind=True,
    autoretry_for=(ImageNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def process(self, db_id: str, url: str, hash: str) -> str:
    if not (image := DataBlob.get_value(QasmVisualization.instance.identifier, hash)):
        if not (
            task_id := PluginState.get_value(QasmVisualization.instance.identifier, hash)
        ):
            task_result = generate_image.s(url, hash).apply_async()
            PluginState.set_value(
                QasmVisualization.instance.identifier,
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
