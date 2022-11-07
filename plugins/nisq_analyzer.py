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
from os import environ

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import abort, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "nisq-analyzer"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


NISQ_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="NISQ Analyzer Plugin.",
)

class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class NisqAnalyzerParametersSchema(FrontendFormBaseSchema):
    results = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Result as String",
            "description": "The analysis result as string.",
            "input_type": "text",
        },
    )


@NISQ_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @NISQ_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @NISQ_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = NisqAnalyzer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{NISQ_BLP.name}.ProcessView"),
                ui_href=plugin.url,
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="nisq-analyzer-result",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=NisqAnalyzer.instance.tags,
        )


@NISQ_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @NISQ_BLP.arguments(NisqAnalyzerParametersSchema(unknown=EXCLUDE), location="form")
    @NISQ_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @NISQ_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the task."""
        db_task = ProcessingTask(task_name=store_results_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = store_results_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class NisqAnalyzer(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = "Provides the NISQ Analyzer UI."
    tags = ["nisq-analyzer"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        self.url = app.config.get("NISQ_ANALYZER_UI_URL")
        self.url = environ.get("NISQ_ANALYZER_UI_URL", self.url)
        self.url += "/#/algorithms"

    def get_api_blueprint(self):
        return NISQ_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{NisqAnalyzer.instance.identifier}.store_results_task", bind=True)
def store_results_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new store results task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    results: Optional[str] = loads(task_data.parameters or "{}").get("results", None)
    TASK_LOGGER.info(f"Loaded input parameters from db: results='{results}'")

    if not results:
        raise ValueError("No input argument provided!")

    with SpooledTemporaryFile(mode="w") as output:
        output.write(results)
        STORE.persist_task_result(
            db_id, output, "nisq_analysis.json", "nisq-analyzer-result", "application/json"
        )

    return "result: " + repr(results)
