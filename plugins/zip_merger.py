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
import json
from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional
from zipfile import ZipFile

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "zip-merger"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


ZIP_MERGER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Zip merger plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParametersSchema(FrontendFormBaseSchema):
    zip1_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="any",
        data_content_types="application/zip",
        metadata={
            "label": "Zip 1 URL",
            "description": "URL to the first zip file.",
            "input_type": "text",
        },
    )
    zip2_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="any",
        data_content_types="application/zip",
        metadata={
            "label": "Zip 2 URL",
            "description": "URL to the second zip file.",
            "input_type": "text",
        },
    )


@ZIP_MERGER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ZIP_MERGER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @ZIP_MERGER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Zip merger endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Zip merger",
            description=ZipMerger.instance.description,
            name=ZipMerger.instance.name,
            version=ZipMerger.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{ZIP_MERGER_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{ZIP_MERGER_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="any",
                        content_type=["application/zip"],
                        required=True,
                        parameter="zip1Url",
                    ),
                    InputDataMetadata(
                        data_type="any",
                        content_type=["application/zip"],
                        required=True,
                        parameter="zip2Url",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="any", content_type=["application/zip"], required=True
                    )
                ],
            ),
            tags=ZipMerger.instance.tags,
        )


@ZIP_MERGER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the zip merger plugin."""

    @ZIP_MERGER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the zip merger plugin."
    )
    @ZIP_MERGER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ZIP_MERGER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @ZIP_MERGER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Wu Palmer plugin."
    )
    @ZIP_MERGER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ZIP_MERGER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=ZipMerger.instance.name,
                version=ZipMerger.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{ZIP_MERGER_BLP.name}.CalcSimilarityView"),
            )
        )


@ZIP_MERGER_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @ZIP_MERGER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @ZIP_MERGER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @ZIP_MERGER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name, parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class ZipMerger(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Merges two zip files into one zip file."
    tags = ["utility"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ZIP_MERGER_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{ZipMerger.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new zip merger task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    zip1_url: Optional[str] = loads(task_data.parameters or "{}").get("zip1_url", None)
    TASK_LOGGER.info(f"Loaded input parameters from db: zip1_url='{zip1_url}'")
    zip2_url: Optional[str] = loads(task_data.parameters or "{}").get("zip2_url", None)
    TASK_LOGGER.info(f"Loaded input parameters from db: zip2_url='{zip2_url}'")

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    merged_zip_file = ZipFile(tmp_zip_file, "w")

    # load data from the files

    for file, file_name in get_files_from_zip_url(zip1_url):
        merged_zip_file.writestr(file_name, file.read())

    for file, file_name in get_files_from_zip_url(zip2_url):
        merged_zip_file.writestr(file_name, file.read())

    merged_zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "merged.zip",
        "any",
        "application/zip",
    )

    return "Result stored in file"
