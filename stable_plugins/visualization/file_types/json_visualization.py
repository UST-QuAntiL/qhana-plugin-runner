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
from qhana_plugin_runner.plugin_utils.dimension_mapping import load_dimension_labels
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "json-visualization"
__version__ = "v0.3.0"
_identifier = plugin_identifier(_plugin_name, __version__)


JSON_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A demo JSON visualization plugin.",
    template_folder="json_visualization_templates",
)


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
    dimension_mapping_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/dimension-mapping",
        data_content_types=["application/json"],
        metadata={
            "label": "Dimension Mapping URL",
            "description": (
                "Optional URL to a dimension mapping file describing where the "
                "dimensions of a vector file came from. The preview then names the "
                "feature every dimension attribute belongs to."
            ),
            "related_to": "data",
            "relation": "pre",
            "related_include_self": True,
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
            name=plugin.name,
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
                    ),
                    InputDataMetadata(
                        data_type="entity/dimension-mapping",
                        content_type=["application/json"],
                        parameter="dimensionMappingUrl",
                        required=False,
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="custom/hello-world-output",
                        content_type=["text/html"],
                        required=True,
                    )
                ],
            ),
            tags=plugin.tags,
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
        return self.render(request.args, errors, False)

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
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
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
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{JSON_BLP.name}.ProcessView"),
                dimension_labels_url=url_for(f"{JSON_BLP.name}.get_dimension_labels"),
                example_values=url_for(f"{JSON_BLP.name}.MicroFrontend"),
            )
        )


@JSON_BLP.route("/dimension-labels/")
@JSON_BLP.response(HTTPStatus.OK, description="Feature name of every dimension.")
@JSON_BLP.arguments(
    JsonInputParametersSchema(partial=True, unknown=EXCLUDE),
    location="query",
    required=True,
)
@JSON_BLP.require_jwt("jwt", optional=True)
def get_dimension_labels(data: Mapping):
    """Name the feature every dimension of a dimension mapping file came from.

    The micro frontend calls this when a dimension mapping is selected and
    annotates the matching attributes of the preview with the returned names.
    Each name ends with the column the dimension had in the input vector, so the
    dimensions of one multi dimensional feature stay distinguishable.
    """
    dimension_mapping_url = data.get("dimension_mapping_url", None)
    if not dimension_mapping_url:
        return Response(dumps({}), mimetype="application/json")
    try:
        labels = load_dimension_labels(
            dimension_mapping_url, include_source_dimension=True
        )
    except (HTTPError, ValueError):
        abort(HTTPStatus.BAD_REQUEST, "Invalid dimension mapping URL!")
    return Response(dumps(labels), mimetype="application/json")


@JSON_BLP.route("/process/")
class ProcessView(
    MethodView
):  # FIXME decide on a somewhat useful implementation for this (or remove completely!)
    """Start a long running processing task."""

    @JSON_BLP.arguments(JsonInputParametersSchema(unknown=EXCLUDE), location="form")
    @JSON_BLP.response(HTTPStatus.SEE_OTHER)
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
    tags = ["visualization", "json"]

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
                db_id, output, "out.txt", "custom/hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"
