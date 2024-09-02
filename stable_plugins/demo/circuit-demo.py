# Copyright 2024 QHAna plugin runner contributors.
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

from enum import Enum
from http import HTTPStatus
from textwrap import dedent
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from flask import abort, jsonify, redirect
from flask.app import Flask
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE, fields

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    MaBaseSchema,
    OutputDataMetadata,
    PluginDependencyMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    PluginUrl,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.interop import (
    call_plugin_endpoint,
    get_plugin_endpoint,
    get_task_result_no_wait,
    monitor_external_substep,
    monitor_result,
    subscribe,
)
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STEPS_CHANGED,
    save_task_error,
    save_task_result,
)
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "circuit-demo"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)


CIRCUIT_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Demo of a quantum circuit using circuit executor plugins.",
)


class BELL_STATES(Enum):
    PHI = "PHI"
    PHI_MINUS = "PHI_MINUS"
    PSI = "PSI"
    PSI_MINUS = "PSI_MINUS"


class CircuitDemoParametersSchema(FrontendFormBaseSchema):
    state = EnumField(
        BELL_STATES,
        metadata={
            "label": "Select Bell State",
            "input_type": "select",
        },
    )
    executor = PluginUrl(
        required=True,
        plugin_tags=["circuit-executor", "qasm-2"],
        metadata={
            "label": "Select Circuit Executor Plugin",
        },
    )


class WebhookParams(MaBaseSchema):
    source = fields.URL()
    event = fields.String()


@CIRCUIT_BLP.route("/")
class PluginView(MethodView):
    """Plugin Metadata resource."""

    @CIRCUIT_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @CIRCUIT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = CircuitDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{CIRCUIT_BLP.name}.ProcessView"),
                ui_href=url_for(f"{CIRCUIT_BLP.name}.MicroFrontend"),
                plugin_dependencies=[
                    PluginDependencyMetadata(
                        required=True,
                        parameter="executor",
                        tags=["circuit-executor", "qasm-2"],
                    ),
                ],
                data_input=[],
                data_output=[
                    OutputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        name="result-counts.json",
                    ),
                    OutputDataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        name="circuit.qasm",
                    ),
                ],
            ),
            tags=CircuitDemo.instance.tags,
        )


@CIRCUIT_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the circuit demo plugin."""

    example_inputs = {
        "state": "PHI",
    }

    @CIRCUIT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the circuit demo plugin."
    )
    @CIRCUIT_BLP.arguments(
        CircuitDemoParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CIRCUIT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @CIRCUIT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the circuit demo plugin."
    )
    @CIRCUIT_BLP.arguments(
        CircuitDemoParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CIRCUIT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = CircuitDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = CircuitDemoParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{CIRCUIT_BLP.name}.ProcessView"),
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{CIRCUIT_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@CIRCUIT_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @CIRCUIT_BLP.arguments(CircuitDemoParametersSchema(unknown=EXCLUDE), location="form")
    @CIRCUIT_BLP.response(HTTPStatus.SEE_OTHER)
    @CIRCUIT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the circuit demo task."""
        state: Optional[BELL_STATES] = arguments.get("state", None)
        if not state:
            state = BELL_STATES.PHI

        circuit_url = url_for(
            f"{CIRCUIT_BLP.name}.{CircuitView.__name__}",
            bell_state=state.value,
            _external=True,
        )
        options_url = url_for(
            f"{CIRCUIT_BLP.name}.{ExecutionOptionsView.__name__}", _external=True
        )
        db_task = ProcessingTask(
            task_name=circuit_demo_task.name,
            parameters=CircuitDemoParametersSchema().dumps(arguments),
        )
        db_task.save()
        DB.session.flush()
        continue_url = url_for(
            f"{CIRCUIT_BLP.name}.{ContinueProcessView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data = {
            "circuit_url": circuit_url,
            "options_url": options_url,
            "continue_url": continue_url,
        }
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task = circuit_demo_task.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CIRCUIT_BLP.route("/continue/<int:db_id>/")
class ContinueProcessView(MethodView):
    """Restart long running task that was blocked by an ongoing plugin computation."""

    @CIRCUIT_BLP.arguments(WebhookParams(partial=True), location="query")
    @CIRCUIT_BLP.response(HTTPStatus.NO_CONTENT)
    def post(self, params: dict, db_id: int):
        """Check for updates in plugin computation and resume processing."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND)

        if task_data.task_name != circuit_demo_task.name:
            # processing task is from another plugin, cannot resume
            abort(HTTPStatus.NOT_FOUND)

        if not isinstance(task_data.data, dict):
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        event_source = params.get("source", None)
        event_type = params.get("event", None)

        result_url = task_data.data.get("result_url")

        if event_source != result_url:
            abort(HTTPStatus.NOT_FOUND)

        if not result_url or task_data.is_finished:
            abort(HTTPStatus.NOT_FOUND)

        task = check_executor_result_task.s(db_id=db_id, event_type=event_type)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@CIRCUIT_BLP.route("/options/")
class ExecutionOptionsView(MethodView):
    """Get the execution options."""

    def get(self):
        """Get the requested execution options."""

        return jsonify({"ID": "1", "shots": 2048, "statevector": False})


@CIRCUIT_BLP.route("/circuit/<string:bell_state>/")
class CircuitView(MethodView):
    """Get the bell state circuit."""

    def get(self, bell_state: str):
        """Get the requested circuit."""

        state = BELL_STATES(bell_state)

        qasm_str = ""

        if state == BELL_STATES.PHI:  # start with |00>
            qasm_str = dedent(
                """
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                creg meas[2];
                h q[0];
                cx q[0],q[1];
                barrier q[0],q[1];
                measure q[0] -> meas[0];
                measure q[1] -> meas[1];
                """
            ).lstrip()
        if state == BELL_STATES.PHI_MINUS:  # start with |01>
            qasm_str = dedent(
                """
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                creg meas[2];
                x q[0];
                h q[0];
                cx q[0],q[1];
                barrier q[0],q[1];
                measure q[0] -> meas[0];
                measure q[1] -> meas[1];
                """
            ).lstrip()
        elif state == BELL_STATES.PSI:  # start with |10>
            qasm_str = dedent(
                """
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                creg meas[2];
                x q[1];
                h q[0];
                cx q[0],q[1];
                barrier q[0],q[1];
                measure q[0] -> meas[0];
                measure q[1] -> meas[1];
                """
            ).lstrip()
        elif state == BELL_STATES.PSI_MINUS:  # start with |11>
            qasm_str = dedent(
                """
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                creg meas[2];
                x q[0];
                x q[1];
                h q[0];
                cx q[0],q[1];
                barrier q[0],q[1];
                measure q[0] -> meas[0];
                measure q[1] -> meas[1];
                """
            ).lstrip()

        return Response(
            qasm_str,
            HTTPStatus.OK,
            mimetype="text/x-qasm",
        )


class CircuitDemo(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "A demo plugin implementing circuits for the bell states and executing them using a circuit executor."
    tags = ["circuit-demo", "demo"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return CIRCUIT_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{CircuitDemo.instance.identifier}.circuit_demo_task", bind=True)
def circuit_demo_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new circuit demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = CircuitDemoParametersSchema().loads(task_data.parameters or "{}")
    state: Optional[BELL_STATES] = params.get("state", None)
    executor: Optional[str] = params.get("executor", None)
    if state is None:
        state = BELL_STATES.PHI
    if executor is None:
        task_data.add_task_log_entry(
            "No executor plugin specified, aborting task.", commit=True
        )
        raise ValueError(
            "Cannot execute a quantum circuit without a circuit executor plugin specified."
        )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: bell state='{state}'; executor='{executor}'"
    )

    endpoint = get_plugin_endpoint(executor)

    circuit_url = task_data.data["circuit_url"]
    options_url = task_data.data["options_url"]
    continue_url = task_data.data["continue_url"]

    result_url = call_plugin_endpoint(
        endpoint, {"circuit": circuit_url, "executionOptions": options_url}
    )

    task_data.add_task_log_entry(f"Awaiting circuit execution result at {result_url}")
    task_data.data["result_url"] = result_url
    task_data.save(commit=True)  # commit to save result url to DB

    subscribed = subscribe(
        result_url=result_url, webhook_url=continue_url, events=["steps", "status"]
    )
    task_data.data["subscribed"] = subscribed
    if subscribed:
        task_data.add_task_log_entry("Subscribed to events from external task.")
    else:
        task_data.add_task_log_entry("Event subscription failed!")

    task_data.save(commit=True)

    app = current_app._get_current_object()
    TASK_DETAILS_CHANGED.send(app, task_id=task_data.id)

    if not subscribed:
        return self.replace(
            monitor_result.s(
                result_url=result_url, webhook_url=continue_url, monitor="all"
            )
        )


def add_new_substep(task_data: ProcessingTask, steps: list) -> Optional[int]:
    last_step = None
    if task_data.steps:
        last_step = task_data.steps[-1]
    current_step = steps[-1] if steps else None
    if current_step:
        step_id = current_step.get("stepId")
        if step_id:
            step_id = f"executor.{step_id}"
        else:
            step_id = f"executor.{len(steps)}"

        if not current_step.get("cleared", False):
            if (
                last_step
                and not last_step.cleared
                and last_step.step_id == step_id
                and last_step.href == current_step["href"]
            ):
                # new step and last step are identical, assume duplicate request and do nothing
                return None
            external_step_id = step_id if step_id else len(steps) - 1
            task_data.clear_previous_step()
            task_data.add_next_step(
                href=current_step["href"],
                ui_href=current_step["uiHref"],
                step_id=step_id,
                commit=True,
            )

            app = current_app._get_current_object()
            TASK_STEPS_CHANGED.send(app, task_id=task_data.id)

            return external_step_id
        elif current_step.get("cleared", False) and task_data.has_uncleared_step:
            if last_step and last_step.step_id == step_id:
                task_data.clear_previous_step(commit=True)
                app = current_app._get_current_object()
                TASK_STEPS_CHANGED.send(app, task_id=task_data.id)
    return None


@CELERY.task(
    name=f"{CircuitDemo.instance.identifier}.check_executor_result_task", bind=True
)
def check_executor_result_task(self, db_id: int, event_type: Optional[str]):
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to save results!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    result_url = task_data.data.get("result_url")
    continue_url = task_data.data["continue_url"]
    subscribed = task_data.data["subscribed"]

    if result_url is None:
        raise ValueError(f"No result URL present in task data with id {db_id}")

    status, result = get_task_result_no_wait(result_url)

    if status == "FAILURE":
        if "--- Circuit Executor Log ---" not in task_data.task_log:
            task_data.add_task_log_entry(
                f"--- Circuit Executor Log ---\n{result.get('log', '')}\n--- END ---\n",
                commit=True,
            )
        raise ValueError("Circuit executor failed to execute the circuit!")
    elif status == "PENDING" and event_type != "status":
        steps = result.get("steps", [])
        external_step_id = add_new_substep(task_data, steps)
        if external_step_id and not subscribed:
            return self.replace(
                # wait for external substep to clear
                monitor_external_substep.s(
                    result_url=result_url,
                    webhook_url=continue_url,
                    substep=external_step_id,
                )
            )
        elif not subscribed:
            return self.replace(
                # wait for external substep or status change
                monitor_result.s(
                    result_url=result_url, webhook_url=continue_url, monitor="all"
                )
            )
    elif status == "SUCCESS" and event_type == "status":
        if "result" in task_data.data:
            return  # already checking for result, prevent duplicate task scheduling!

        task_data.data["result"] = result
        task_data.save(commit=True)

        return self.replace(
            circuit_demo_result_task.si(db_id=db_id) | save_task_result.s(db_id=db_id)
        )
    else:
        raise ValueError(f"Unknown task status {status}!")


@CELERY.task(
    name=f"{CircuitDemo.instance.identifier}.circuit_demo_result_task", bind=True
)
def circuit_demo_result_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Saving circuit demo task results with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to save results!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    circuit_url = task_data.data.get("circuit_url")
    if circuit_url and isinstance(circuit_url, str):
        STORE.persist_task_result(
            db_id,
            circuit_url,
            "circuit.qasm",
            "executable/circuit",
            "text/x-qasm",
            storage_provider="url_file_store",
        )

    outputs = task_data.data.get("result", {}).get("outputs", [])

    for out in outputs:
        if out.get("name", "").startswith(("result-counts", "result-statevector")):
            name = out.get("name", "")
            url = out.get("href", "")
            data_type = out.get("dataType", "")
            content_type = out.get("contentType", "")
            STORE.persist_task_result(
                db_id,
                url,
                name,
                data_type,
                content_type,
                storage_provider="url_file_store",
            )

    return "Successfully saved circuit executor task result!"
