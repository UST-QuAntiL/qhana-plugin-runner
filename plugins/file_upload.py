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
from werkzeug.utils import secure_filename

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "file-upload"
__version__ = "v0.2.0"
_identifier = plugin_identifier(_plugin_name, __version__)


FILE_UPLOAD_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="File upload plugin API.",
    template_folder="file_upload_templates",
)


class FileUploadParametersSchema(FrontendFormBaseSchema):
    data_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"label": "Data Type", "description": "Semantic of the data"},
    )
    content_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Content Type",
            "description": "Encoding of the data (mimetype)",
        },
    )


@FILE_UPLOAD_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @FILE_UPLOAD_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @FILE_UPLOAD_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = FileUpload.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{FILE_UPLOAD_BLP.name}.ProcessView"),
                ui_href=url_for(f"{FILE_UPLOAD_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["*"],
                        required=True,
                    )
                ],
            ),
            tags=FileUpload.instance.tags,
        )


@FILE_UPLOAD_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @FILE_UPLOAD_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @FILE_UPLOAD_BLP.arguments(
        FileUploadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @FILE_UPLOAD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @FILE_UPLOAD_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @FILE_UPLOAD_BLP.arguments(
        FileUploadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @FILE_UPLOAD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = FileUpload.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = FileUploadParametersSchema()
        return Response(
            render_template(
                "file_upload_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{FILE_UPLOAD_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{FILE_UPLOAD_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@FILE_UPLOAD_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @FILE_UPLOAD_BLP.arguments(
        FileUploadParametersSchema(unknown=EXCLUDE), location="form"
    )
    @FILE_UPLOAD_BLP.response(HTTPStatus.SEE_OTHER)
    @FILE_UPLOAD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(task_name=demo_task.name, parameters="")
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = demo_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))

        try:
            if "file" not in request.files:
                raise ValueError("No file in request.")

            file = request.files["file"]

            if file.filename == "":
                raise ValueError("No file selected.")

            if file:
                filename = secure_filename(file.filename)

                STORE.persist_task_result(
                    db_task.id,
                    file.stream,
                    filename,
                    arguments["data_type"],
                    arguments["content_type"],
                )
        except Exception as err:
            db_task.parameters = dumps({"error": str(err)})

        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class FileUpload(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Uploads files to use in the workflow."
    tags = ["data-loading"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return FILE_UPLOAD_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{FileUpload.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    error_message: Optional[str] = loads(task_data.parameters or "{}").get("error", None)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    if error_message is not None:
        raise Exception(error_message)

    return "File uploaded."
