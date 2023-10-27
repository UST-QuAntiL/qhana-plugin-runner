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
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from celery.utils.log import get_task_logger

from qhana_plugin_runner.plugin_utils.metadata_utils import parse_execution_options
from qhana_plugin_runner.util.logging import redact_log_data
from . import QISKIT_EXECUTOR_BLP, QiskitExecutor
from .schemas import (
    BackendParameterSchema,
    CircuitParameters,
    CircuitParameterSchema,
    AuthenticationParameterSchema,
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
from .tasks import start_execution

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
        CircuitParameterSchema(
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
        CircuitParameterSchema(
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
        fields = CircuitParameterSchema().fields

        # define default values
        default_values = {}

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
                schema=CircuitParameterSchema(),
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
        CircuitParameterSchema(unknown=EXCLUDE), location="form"
    )
    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.SEE_OTHER)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: CircuitParameters):
        """Start the circuit execution task."""
        options = {}
        if arguments.executionOptions:
            options = parse_execution_options(arguments.executionOptions)
            TASK_LOGGER.info(f"Loaded execution options: {redact_log_data(options)}")

        if not arguments.shots:
            arguments.shots = options.get("shots", 1024)
        if not arguments.ibmqToken:
            arguments.ibmqToken = options.get("ibmqToken", None)
        if not arguments.backend:
            arguments.backend = options.get("backend", None)
        progress_target = 2
        if arguments.backend and arguments.ibmqToken:
            progress_target = 1

        db_task = ProcessingTask(
            task_name=start_execution.name,
            data={
                "parameters": CircuitParameterSchema().dumps(arguments),
                "options": options,
            },
            progress_value=0,
            progress_target=progress_target,
            progress_unit="steps",
        )
        db_task.save(commit=True)

        # generate urls for celery task
        auth_href = url_for(
            f"{QISKIT_EXECUTOR_BLP.name}.AuthenticationView",
            db_id=db_task.id,
            _external=True,
        )
        auth_ui_href = url_for(
            f"{QISKIT_EXECUTOR_BLP.name}.AuthenticationFrontend",
            db_id=db_task.id,
            _external=True,
        )
        backend_href = url_for(
            f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionView",
            db_id=db_task.id,
            _external=True,
        )
        backend_ui_href = url_for(
            f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionFrontend",
            db_id=db_task.id,
            _external=True,
        )

        db_task.data["urls"] = {
            "authentication": auth_href,
            "authentication_ui": auth_ui_href,
            "backend_selection": backend_href,
            "backend_selection_ui": backend_ui_href,
        }

        db_task.save(commit=True)

        start_execution.s(db_id=db_task.id).apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/authentication-step-ui/")
class AuthenticationFrontend(MethodView):
    """Micro frontend for the backend selection step of the qiskit executor plugin."""

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the backend selection step."
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        AuthenticationParameterSchema(
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
        AuthenticationParameterSchema(
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

        params: CircuitParameters = CircuitParameterSchema().loads(
            db_task.data["parameters"]
        )
        fields = AuthenticationParameterSchema().fields

        # define default values
        default_values = {}

        if params.backend:
            default_values[fields["backend"].data_key] = params.backend

        if "IBMQ_TOKEN" in os.environ:
            default_values[fields["ibmqToken"].data_key] = "****"
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")

        return Response(
            render_template(
                "simple_template.html",
                name=QiskitExecutor.instance.name,
                version=QiskitExecutor.instance.version,
                schema=AuthenticationParameterSchema(),
                valid=valid,
                values=default_values,
                errors=errors,
                process=url_for(
                    f"{QISKIT_EXECUTOR_BLP.name}.AuthenticationView", db_id=db_id
                ),
            )
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/authentication-step-process")
class AuthenticationView(MethodView):
    """Start a long running processing task."""

    @QISKIT_EXECUTOR_BLP.arguments(
        AuthenticationParameterSchema(unknown=EXCLUDE), location="form"
    )
    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.SEE_OTHER)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the circuit execution task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            raise KeyError(msg)

        params: CircuitParameters = CircuitParameterSchema().loads(
            db_task.data["parameters"]
        )
        params.ibmqToken = arguments.ibmqToken
        if (not params.backend) and arguments.backend:
            params.backend = arguments.backend
        db_task.data["parameters"] = CircuitParameterSchema().dumps(params)

        db_task.clear_previous_step()
        db_task.save(commit=True)

        start_execution.s(db_id=db_task.id).apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/backend-selection-ui/")
class BackendSelectionFrontend(MethodView):
    """Micro frontend for the backend selection step of the qiskit executor plugin."""

    @QISKIT_EXECUTOR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the backend selection step."
    )
    @QISKIT_EXECUTOR_BLP.arguments(
        BackendParameterSchema(
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
        BackendParameterSchema(
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

        backend_parameter_schema = BackendParameterSchema()
        backend_parameter_schema.fields["backend"].metadata["datalist"] = db_task.data[
            "backend_names"
        ]

        return Response(
            render_template(
                "simple_template.html",
                name=QiskitExecutor.instance.name,
                version=QiskitExecutor.instance.version,
                schema=backend_parameter_schema,
                values=dict(data),
                valid=valid,
                errors=errors,
                process=url_for(
                    f"{QISKIT_EXECUTOR_BLP.name}.BackendSelectionView", db_id=db_id
                ),
            )
        )


@QISKIT_EXECUTOR_BLP.route("/<int:db_id>/backend-selection-process")
class BackendSelectionView(MethodView):
    """Start a long running processing task."""

    @QISKIT_EXECUTOR_BLP.arguments(
        BackendParameterSchema(unknown=EXCLUDE), location="form"
    )
    @QISKIT_EXECUTOR_BLP.response(HTTPStatus.SEE_OTHER)
    @QISKIT_EXECUTOR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the circuit execution task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            raise KeyError(msg)

        params: CircuitParameters = CircuitParameterSchema().loads(
            db_task.data["parameters"]
        )
        params.backend = arguments.backend
        db_task.data["parameters"] = CircuitParameterSchema().dumps(params)

        db_task.clear_previous_step()
        db_task.save(commit=True)

        start_execution.s(db_id=db_task.id).apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
