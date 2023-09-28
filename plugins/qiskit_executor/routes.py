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
from http import HTTPStatus
from typing import Mapping, Optional

from celery.canvas import chain
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from celery.utils.log import get_task_logger

from . import QISKIT_EXECUTOR_BLP, QiskitExecutor
from .schemas import (
    CircuitSelectionParameterSchema,
    BackendSelectionParameterSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    OutputDataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    InputDataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

from .tasks import prepare_task, execution_task

TASK_LOGGER = get_task_logger(__name__)


@QISKIT_EXECUTOR_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Qiskit executor endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Qiskit Executor",
            description=QiskitExecutor.instance.description,
            name=QiskitExecutor.instance.name,
            version=QiskitExecutor.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{QISKIT_EXECUTOR_BLP.name}.CalcView"),
                ui_href=url_for(f"{QISKIT_EXECUTOR_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        parameter="circuit",
                    ),
                    InputDataMetadata(
                        data_type="provenance/execution-options",
                        content_type=[
                            "text/csv",
                            "application/json",
                            "application/X-lines+json",
                        ],
                        required=False,
                        parameter="executionOptions",
                    ),
                ],
                data_output=[
                    OutputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        name="result-counts.json",
                    ),
                    OutputDataMetadata(
                        data_type="provenance/trace",
                        content_type=["application/json"],
                        required=True,
                        name="result-trace.json",
                    ),
                    OutputDataMetadata(
                        data_type="provenance/execution-options",
                        content_type=["application/json"],
                        required=True,
                        name="execution-options.json",
                    ),
                ],
            ),
            tags=QiskitExecutor.instance.tags,
        )


@QISKIT_EXECUTOR_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the qiskit executor plugin."""

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the qiskit executor plugin.",
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        CircuitSelectionParameterSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the qiskit executor plugin.",
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        CircuitSelectionParameterSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        data_dict = dict(data)
        fields = CircuitSelectionParameterSchema().fields

        # define default values
        default_values = {
            fields["shots"].data_key: 1024,
        }

        if "IBMQ_BACKEND" in os.environ:
            default_values[fields["backend"].data_key] = os.environ["IBMQ_BACKEND"]

        if "IBMQ_TOKEN" in os.environ:
            default_values[fields["ibmqToken"].data_key] = "****"
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QiskitExecutor.instance.name,
                version=QiskitExecutor.instance.version,
                schema=CircuitSelectionParameterSchema(),
                valid=valid,
                values=data_dict,
                errors=errors,
                process=url_for(f"{QISKIT_EXECUTOR_BLP.name}.CalcView"),
            )
        )


@QISKIT_EXECUTOR_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @QISKIT_EXECUTOR_BLP.arguments(
        CircuitSelectionParameterSchema(unknown=EXCLUDE), location="form"
    )
    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.SEE_OTHER)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the circuit execution task."""
        db_task = ProcessingTask(
            task_name=execution_task.name,
            data=CircuitSelectionParameterSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        if arguments.backend != "":
            # start the execution task directly
            task: chain = execution_task.s(db_id=db_task.id) | save_task_result.s(
                db_id=db_task.id
            )
        else:
            # start the backend selection task (which then starts the execution task)
            step_id = "backend-selection"
            href = url_for(
                f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionStepView",
                db_id=db_task.id,
                _external=True,
            )
            ui_href = url_for(
                f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionStepFrontend",
                db_id=db_task.id,
                _external=True,
            )
            task: chain = prepare_task.s(db_id=db_task.id) | add_step.s(
                db_id=db_task.id,
                step_id=step_id,
                href=href,
                ui_href=ui_href,
                prog_value=50,
            )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/backend-selection-ui/")
class BackendSelectionStepFrontend(MethodView):
    """Micro frontend for the backend selection step of the qiskit executor plugin."""

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the backend selection step."
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        BackendSelectionParameterSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(db_id, request.args, errors, False)

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the backend selection step."
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        BackendSelectionParameterSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=True,
    )
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, db_id, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(db_id, request.form, errors, not errors)

    def render(self, db_id, data: Mapping, errors: dict, valid: bool):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            raise KeyError(msg)

        ibmq_token = CircuitSelectionParameterSchema().loads(db_task.data).ibmqToken
        data_dict = dict(data)
        fields = BackendSelectionParameterSchema().fields

        # define default values
        default_values = {}

        if "IBMQ_TOKEN" in os.environ:
            default_values[fields["ibmqToken"].data_key] = "****"
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QiskitExecutor.instance.name,
                version=QiskitExecutor.instance.version,
                schema=BackendSelectionParameterSchema(),
                valid=valid,
                values=data_dict,
                errors=errors,
                process=url_for(
                    f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionStepView", db_id=db_id
                ),
            )
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/backend-selection-process")
class BackendSelectionStepView(MethodView):
    """Start a long running processing task."""

    @QISKIT_EXECUTOR_BLP.arguments(
        BackendSelectionParameterSchema(unknown=EXCLUDE), location="form"
    )
    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.SEE_OTHER)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the circuit execution task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            raise KeyError(msg)

        task_data = CircuitSelectionParameterSchema().loads(db_task.data)
        task_data.backend = arguments.backend
        db_task.data = CircuitSelectionParameterSchema().dumps(task_data)

        db_task.clear_previous_step()
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = execution_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
