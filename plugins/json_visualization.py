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

from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import abort, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

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
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "json-visualization"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


JSON_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A demo JSON visualization plugin.",
    template_folder="json_visualization_templates",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class JsonInputParametersSchema(FrontendFormBaseSchema):
    data = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["application/json"],
        metadata={
            "label": "JSON File",
            "description": "The URL to a JSON file.",
        },
    )


@JSON_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @JSON_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @JSON_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = JsonVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.identifier,
            version=plugin.version,
            type=PluginType.visualization,
            entry_point=EntryPoint(
                href=url_for(f"{JSON_BLP.name}.ProcessView"),
                ui_href=url_for(f"{JSON_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="*",
                        content_type=["application/json"],
                        parameter="data",
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["text/html"],
                        required=True,
                    )
                ],
            ),
            tags=[],
        )


@JSON_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the JSON visualization plugin."""

    @JSON_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the JSON visualization plugin."
    )
    @JSON_BLP.arguments(
        JsonInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @JSON_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @JSON_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the json visualization plugin."
    )
    @JSON_BLP.arguments(
        JsonInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @JSON_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        plugin = JsonVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = JsonInputParametersSchema()
        return Response(
            render_template(
                "json_visualization.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{JSON_BLP.name}.ProcessView"),
                example_values=url_for(f"{JSON_BLP.name}.MicroFrontend"),
            )
        )


@JSON_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @JSON_BLP.arguments(JsonInputParametersSchema(unknown=EXCLUDE), location="form")
    @JSON_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @JSON_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(task_name=demo_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = demo_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class JsonVisualization(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = "Visualizes JSON data."
    tags = ["visualization"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return JSON_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{JsonVisualization.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: Optional[str] = loads(task_data.parameters or "{}").get("input_str", None)
    TASK_LOGGER.info(f"Loaded input parameters from db: input_str='{input_str}'")
    if input_str is None:
        raise ValueError("No input argument provided!")
    if input_str:
        out_str = input_str.replace("input", "output")
        with SpooledTemporaryFile(mode="w") as output:
            output.write(out_str)
            STORE.persist_task_result(
                db_id, output, "out.txt", "hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"
