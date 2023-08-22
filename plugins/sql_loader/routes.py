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

from http import HTTPStatus
from typing import Mapping, Optional
from json import dumps

from celery.canvas import chain
from celery.utils.log import get_task_logger
from celery.exceptions import TimeoutError as CeleryTimeoutError
from flask import Response, redirect, Markup
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import SQLLoader_BLP, SQLLoaderPlugin
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
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

from .tasks import (
    first_task,
    second_task,
    get_second_task_html
)

TASK_LOGGER = get_task_logger(__name__)


@SQLLoader_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @SQLLoader_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """db_manager endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="DB Manager",
            description=SQLLoaderPlugin.instance.description,
            name=SQLLoaderPlugin.instance.identifier,
            version=SQLLoaderPlugin.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{SQLLoader_BLP.name}.FirstProcessView"),
                ui_href=url_for(f"{SQLLoader_BLP.name}.FirstMicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        "entity/vectors",
                        content_type=["application/json"],
                        required=False,
                        parameter="dbHost",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                        ],
                        required=True,
                    ),
                ],
            ),
            tags=SQLLoaderPlugin.instance.tags,
        )


@SQLLoader_BLP.route("/ui/")
class FirstMicroFrontend(MethodView):
    """Micro frontend for the db manager plugin."""

    @SQLLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the db manager plugin."
    )
    @SQLLoader_BLP.arguments(
        FirstInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @SQLLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the db manager plugin."
    )
    @SQLLoader_BLP.arguments(
        FirstInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = FirstInputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {
            fields["db_host"].data_key: "localhost",
            fields["db_port"].data_key: -1,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "first_template.html",
                name=SQLLoaderPlugin.instance.name,
                version=SQLLoaderPlugin.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{SQLLoader_BLP.name}.FirstProcessView"),
            )
        )


@SQLLoader_BLP.route("/process/")
class FirstProcessView(MethodView):
    """Start a long running processing task."""

    @SQLLoader_BLP.arguments(FirstInputParametersSchema(unknown=EXCLUDE), location="form")
    @SQLLoader_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=first_task.name,
            parameters=FirstInputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # next step
        step_id = "2.0"
        href = url_for(
            f"{SQLLoader_BLP.name}.SecondProcessView",
            db_id=db_task.id,
            step_id=step_id,
            _external=True,
        )
        ui_href = url_for(
            f"{SQLLoader_BLP.name}.SecondMicroFrontend",
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


@SQLLoader_BLP.route("/<int:db_id>/<float:step_id>/ui/")
class SecondMicroFrontend(MethodView):
    """Micro frontend for the db manager plugin."""

    @SQLLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the db manager plugin."
    )
    @SQLLoader_BLP.arguments(
        SecondInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int, step_id: float):
        """Return the micro frontend."""
        return self.render(request.form, db_id, step_id, errors)

    @SQLLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the db manager plugin."
    )
    @SQLLoader_BLP.arguments(
        SecondInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int, step_id: float):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, step_id, errors)

    def render(self, data: Mapping, db_id: int, step_id: float, errors: dict):
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
            # fields["str_list"].data_key: ["localhost", "localhost", "peter"],
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "second_template.html",
                name=SQLLoaderPlugin.instance.name,
                version=SQLLoaderPlugin.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(
                    f"{SQLLoader_BLP.name}.SecondProcessView",
                    db_id=db_id,
                    step_id=step_id,
                ),
                get_pd_html=url_for(
                    f"{SQLLoader_BLP.name}.GetPDHTML",
                    db_id=db_id,
                    step_id=step_id,
                ),
                additional_info=Markup(db_task.data["db_tables_and_columns"]),
                checkbox_list=Markup(db_task.data["checkbox_list"]),
            )
        )


@SQLLoader_BLP.route("/<int:db_id>/<float:step_id>/ui/pd_html")
class GetPDHTML(MethodView):
    @SQLLoader_BLP.html_response(
        HTTPStatus.OK, description="Returns a query result in html form"
    )
    @SQLLoader_BLP.arguments(
        SecondInputParametersSchema(unknown=EXCLUDE),
        location="query",
    )
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def get(self, arguments, db_id: int, step_id: float):
        try:
            return get_second_task_html.s(db_id=db_id, arguments=SecondInputParametersSchema().dumps(arguments)).apply_async().get(timeout=15)
        except CeleryTimeoutError:
            return "Query timed out"


@SQLLoader_BLP.route("/<int:db_id>/<float:step_id>-process/")
class SecondProcessView(MethodView):
    """Start a long running processing task."""

    @SQLLoader_BLP.arguments(
        SecondInputParametersSchema(unknown=EXCLUDE), location="form",
    )
    @SQLLoader_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @SQLLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int, step_id: float):
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
        task: chain = second_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
