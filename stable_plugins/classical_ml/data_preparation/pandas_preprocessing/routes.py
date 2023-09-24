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

import os
from tempfile import NamedTemporaryFile
from http import HTTPStatus
from typing import Mapping, Optional

from pathlib import Path

from celery.canvas import chain
from flask import send_file, Response, redirect, Markup
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import PDPreprocessing_BLP, PDPreprocessing
from .schemas import (
    FirstInputParametersSchema,
    SecondInputParametersSchema,
    TaskResponseSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    InputDataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

from .tasks import first_task, second_task

from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


@PDPreprocessing_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PDPreprocessing_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """pandas-preprocessing endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Pandas Preprocessing",
            description=PDPreprocessing.instance.description,
            name=PDPreprocessing.instance.name,
            version=PDPreprocessing.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{PDPreprocessing_BLP.name}.FirstProcessView"),
                ui_href=url_for(f"{PDPreprocessing_BLP.name}.FirstMicroFrontend"),
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="",
                        content_type=[
                            "text/csv",
                        ],
                        required=True,
                    ),
                ],
            ),
            tags=PDPreprocessing.instance.tags,
        )


@PDPreprocessing_BLP.route("/ui/")
class FirstMicroFrontend(MethodView):
    """Micro frontend for the pandas preprocessing plugin."""

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        FirstInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        FirstInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = FirstInputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {}

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=PDPreprocessing.instance.name,
                version=PDPreprocessing.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{PDPreprocessing_BLP.name}.FirstProcessView"),
            )
        )


@PDPreprocessing_BLP.route("/process/")
class FirstProcessView(MethodView):
    """Start a long running processing task."""

    @PDPreprocessing_BLP.arguments(
        FirstInputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PDPreprocessing_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=first_task.name,
            parameters=FirstInputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # next step
        step_id = "2"
        href = url_for(
            f"{PDPreprocessing_BLP.name}.SecondProcessView",
            db_id=db_task.id,
            step_id=step_id,
            _external=True,
        )
        ui_href = url_for(
            f"{PDPreprocessing_BLP.name}.SecondMicroFrontend",
            db_id=db_task.id,
            step_id=step_id,
            _external=True,
        )

        # all tasks need to know about db id to load the db entry
        task: chain = first_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@PDPreprocessing_BLP.route("/<int:db_id>/<int:step_id>/ui/")
class SecondMicroFrontend(MethodView):
    """Micro frontend for the pandas preprocessing plugin."""

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        SecondInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int, step_id: int):
        """Return the micro frontend."""
        return self.render(request.form, db_id, step_id, errors)

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        SecondInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int, step_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, step_id, errors)

    def render(self, data: Mapping, db_id: int, step_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        schema = SecondInputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {
            fields["threshold"].data_key: 0,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "second_template.html",
                name=PDPreprocessing.instance.name,
                version=PDPreprocessing.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(
                    f"{PDPreprocessing_BLP.name}.FinalProcessView",
                    db_id=db_id,
                    step_id=step_id,
                ),
                add_step=url_for(
                    f"{PDPreprocessing_BLP.name}.SecondProcessView",
                    db_id=db_id,
                    step_id=step_id,
                ),
                pandas_html=Markup(db_task.data["pandas_html"]),
                columns_and_rows_html=Markup(db_task.data["columns_and_rows_html"]),
                columns_as_optionlist=Markup(db_task.data["columns_as_optionlist"]),
            )
        )


@PDPreprocessing_BLP.route("/<int:db_id>/<int:step_id>/process/final/")
class FinalProcessView(MethodView):
    """Start a long running processing task."""

    @PDPreprocessing_BLP.arguments(
        SecondInputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PDPreprocessing_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int, step_id: int):
        """Start the calculation task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = SecondInputParametersSchema().dumps(arguments)

        db_task.clear_previous_step()
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = second_task.s(
            db_id=db_task.id, step_id=step_id
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@PDPreprocessing_BLP.route("/<int:db_id>/<int:step_id>/process/second/")
class SecondProcessView(MethodView):
    """Start a long running processing task."""

    @PDPreprocessing_BLP.arguments(
        SecondInputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PDPreprocessing_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int, step_id: int):
        """Start the calculation task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = SecondInputParametersSchema().dumps(arguments)

        # next step
        next_step_id = str(step_id + 1)
        href = url_for(
            f"{PDPreprocessing_BLP.name}.FinalProcessView",
            db_id=db_task.id,
            step_id=next_step_id,
            _external=True,
        )
        ui_href = url_for(
            f"{PDPreprocessing_BLP.name}.SecondMicroFrontend",
            db_id=db_task.id,
            step_id=next_step_id,
            _external=True,
        )

        db_task.clear_previous_step()
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = second_task.s(db_id=db_task.id, step_id=step_id) | add_step.s(
            db_id=db_task.id,
            step_id=next_step_id,
            href=href,
            ui_href=ui_href,
            prog_value=(step_id / (step_id + 1)) * 100,
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
