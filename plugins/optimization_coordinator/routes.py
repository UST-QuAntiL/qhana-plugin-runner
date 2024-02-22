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
from logging import Logger
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    WebhookParams,
    WebhookParamsSchema,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.interop import get_task_result_no_wait
from qhana_plugin_runner.tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STATUS_CHANGED,
    TASK_STEPS_CHANGED,
    save_task_error,
)

from . import OPTIMIZER_BLP, Optimizer
from .schemas import OptimizerSetupTaskInputData, OptimizerSetupTaskInputSchema
from .tasks import (
    add_plugin_entrypoint_task,
    check_minimizer_steps,
    check_of_steps,
    handle_minimizer_result,
    handle_of_result,
)


@OPTIMIZER_BLP.route("/")
class MetadataView(MethodView):
    """
    Plugins collection resource.

    A View that handles the plugins metadata.
    """

    @OPTIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self) -> PluginMetadata:
        """
        Optimizer endpoint returning the plugin metadata.

        Returns:
            PluginMetadata: The metadata of the optimizer plugin.
        """
        return PluginMetadata(
            title="Optimizer plugin",
            description=Optimizer.instance.description,
            name=Optimizer.instance.name,
            version=Optimizer.instance.version,
            type=PluginType.processing,
            tags=Optimizer.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupProcessStep.__name__}"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupMicroFrontend.__name__}"
                ),  # URL for the first micro frontend endpoint
                data_input=[],
                data_output=[
                    DataMetadata(  # FIXME
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                ],
            ),
        )


@OPTIMIZER_BLP.route("/ui-setup/")
class OptimizerSetupMicroFrontend(MethodView):
    """
    Micro frontend for selection of:
    1. objective-function plugin
    2. dataset
    3. minimizer plugin
    4. target variable

    This class is responsible for the handling of the setup UI.
    """

    example_inputs = {}

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the objective-function and dataset selection in the optimizer plugin.",
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors) -> Response:
        """
        Return the micro frontend.

        Args:
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the micro frontend.
        """

        return self.render(request.args, errors)

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the objective-function and dataset selection in the optimizer plugin.",
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors) -> Response:
        """
        Return the micro frontend with prerendered inputs.

        Args:
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the micro frontend with prerendered inputs.
        """

        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict) -> Response:
        """
        Render the UI for the plugin setup.

        Args:
            data (Mapping): The input data for the setup.
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the plugin setup UI.
        """
        schema = OptimizerSetupTaskInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Optimizer.instance.name,
                version=Optimizer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupProcessStep.__name__}"
                ),
                example_values=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupMicroFrontend.__name__}",
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTIMIZER_BLP.route("/process-setup/")
class OptimizerSetupProcessStep(MethodView):
    """
    UI input processing ressource.

    A View that handles handles the processing of the UI input data.
    """

    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(unknown=EXCLUDE), location="form"
    )
    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: OptimizerSetupTaskInputData) -> Response:
        """Handle POST requests for the setup UI.

        This method handles the POST requests for the setup UI.
        It saves the input data to the database, gets the metadata of the plugins and saves the endpoints to the database.
        It also invokes the objective function plugin.

        Args:
            arguments (OptimizerSetupTaskInputData): The input data of the setup UI.

        Returns:
            Response: A redirect to the task view.
        """
        db_task = ProcessingTask(
            task_name="optimize",
        )

        assert isinstance(db_task.data, dict)  # type checker assertion

        db_task.data["task_state"] = "setup"

        # save the input data to the database
        db_task.data["features_url"] = arguments.features
        db_task.data["target_url"] = arguments.target

        db_task.save()
        DB.session.flush()

        # save the callback endpoint for the minimizer plugins to the database
        minimizer_webhook = url_for(
            f"{OPTIMIZER_BLP.name}.{MinimizerWebhook.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["minimizer_webhook"] = minimizer_webhook
        db_task.data["minimizer_plugin_url"] = arguments.minimizer_plugin_selector

        # get the objective function plugin metadata and save the endpoints to the database
        of_webhook = url_for(
            f"{OPTIMIZER_BLP.name}.{ObjectiveFunctionWebhook.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["of_webhook"] = of_webhook
        db_task.data["of_plugin_url"] = arguments.objective_function_plugin_selector

        db_task.save(commit=True)

        task = add_plugin_entrypoint_task.s(
            db_id=db_task.id,
            plugin_url=arguments.objective_function_plugin_selector,
            webhook_url=of_webhook,
            step_id="of_setup",
            task_log="Prepare to setup objective function.",
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


#### Webhooks ##################################################################


@OPTIMIZER_BLP.route("/task/<int:db_id>/objective-function-webhook/")
class ObjectiveFunctionWebhook(MethodView):
    """Webhook receiving updates of the objective function."""

    @OPTIMIZER_BLP.arguments(WebhookParamsSchema(partial=True), location="query")
    @OPTIMIZER_BLP.response(HTTPStatus.NO_CONTENT)
    def post(self, params: WebhookParams, db_id: int):
        """
        Handle webhook updates of the objective function.

        Args:
            params ({'source': '{url}', 'event': '{event type}'}): standard webhook data for task update subscriptions.
            db_id (str): The ID of the task.
        """
        event_type = params["event"]

        if event_type not in ("status", "steps"):
            # event is not interesting here
            abort(HTTPStatus.NOT_FOUND, message="wrong event")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)  # type checker assertion

        task_state = db_task.data.get("task_state")

        if db_task.task_name != "optimize" or task_state is None:
            # wrong task name or missing task state data
            abort(HTTPStatus.NOT_FOUND, message="wrong task name/type")

        if task_state != "setup" and db_task.data["of_task_url"] != params["source"]:
            # wrong webhook source
            abort(
                HTTPStatus.NOT_FOUND,
                message=f"wrong task phase or source url (phase={task_state})",
            )

        of_task_url = params["source"]

        app = current_app._get_current_object()

        # handle status events

        if event_type == "status":
            status, result = get_task_result_no_wait(of_task_url)
            if status == "FAILURE" and db_task.data.get("of_success") is not False:
                old_status = db_task.status

                db_task.task_status = "FAILURE"
                db_task.data["of_success"] = False
                db_task.add_task_log_entry(f"Objective function FAILED! ({of_task_url})")
                if task_state not in ("setup-minimizer", "minimize") and (
                    log := result.get("log")
                ):
                    db_task.add_task_log_entry(
                        f"--- objective function log ---\n{log}\n--- end objective function log ---"
                    )

                db_task.save(commit=True)

                TASK_DETAILS_CHANGED.send(app, task_id=db_id)
                if old_status != db_task.status:
                    TASK_STATUS_CHANGED.send(app, task_id=db_id)

            if status == "SUCCESS":
                task = handle_of_result.s(db_id=db_id)
                task.link_error(save_task_error.s(db_id=db_task.id))
                task.apply_async()
            return "", HTTPStatus.NO_CONTENT

        # only step events get past this line

        if task_state in ("minimizer_init", "minimizer_setup", "minimize"):
            # do not handle step events of the objective function during this phase
            # TODO: check how such events should be handled (and by which plugin!)
            return "", HTTPStatus.NO_CONTENT

        if task_state == "setup":
            db_task.data["of_task_url"] = params["source"]
            db_task.data["task_state"] = "of_setup"
            db_task.clear_previous_step()
            db_task.save(commit=True)
            TASK_STEPS_CHANGED.send(app, task_id=db_id)

        # start task to handle of-step updates
        task = check_of_steps.s(db_id=db_id)
        task.apply_async()

        return "", HTTPStatus.NO_CONTENT


@OPTIMIZER_BLP.route("/task/<int:db_id>/minimizer-webhook/")
class MinimizerWebhook(MethodView):
    """Webhook receiving updates of the minimizer."""

    @OPTIMIZER_BLP.arguments(WebhookParamsSchema(partial=True), location="query")
    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    def post(self, params: dict, db_id: int):
        """
        Handle webhook updates of the minimizer.

        Args:
            params ({'source': '<url>', 'event': '<event type>'}): standard webhook data for task update subscriptions.
            db_id (str): The ID of the task.
        """
        event_type = params["event"]

        if event_type not in ("status", "steps"):
            # event is not interesting here
            abort(HTTPStatus.NOT_FOUND, message="wrong event")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)  # type checker assertion

        task_state = db_task.data.get("task_state")

        if db_task.task_name != "optimize" or task_state is None:
            # wrong task name or missing task state data
            abort(HTTPStatus.NOT_FOUND, message="wrong task name/type")

        if task_state in ("setup", "of_setup", "of_cleanup"):
            # wrong task phase
            abort(HTTPStatus.NOT_FOUND, message=f"Wrong task phase: {task_state}")

        if (
            task_state != "minimizer_init"
            and db_task.data["minimizer_task_url"] != params["source"]
        ):
            abort(
                HTTPStatus.NOT_FOUND, message="Wrong source url"
            )  # wrong webhook source

        minimizer_task_url = params["source"]

        app = current_app._get_current_object()

        # handle status events

        if event_type == "status":
            status, result = get_task_result_no_wait(minimizer_task_url)
            if status == "FAILURE" and db_task.data.get("minimizer_success") is not False:
                old_status = db_task.status

                db_task.task_status = "FAILURE"
                db_task.data["minimizer_success"] = False
                db_task.add_task_log_entry(f"Minimizer FAILED! ({minimizer_task_url})")
                if log := result.get("log"):
                    db_task.add_task_log_entry(
                        f"--- minimizer log ---\n{log}\n--- end minimizer log ---"
                    )

                db_task.save(commit=True)

                TASK_DETAILS_CHANGED.send(app, task_id=db_id)
                if old_status != db_task.status:
                    TASK_STATUS_CHANGED.send(app, task_id=db_id)

            if status == "SUCCESS":
                task = handle_minimizer_result.s(db_id=db_id)
                task.link_error(save_task_error.s(db_id=db_task.id))
                task.apply_async()
            return "", HTTPStatus.NO_CONTENT

        # only step events get past this line

        if task_state == "minimizer_init":
            db_task.data["minimizer_task_url"] = params["source"]
            db_task.data["task_state"] = "minimizer_setup"
            db_task.clear_previous_step()
            db_task.save(commit=True)
            TASK_STEPS_CHANGED.send(app, task_id=db_id)

        # start task to handle minimizer-step updates
        task = check_minimizer_steps.s(db_id=db_id)
        task.apply_async()

        return "", HTTPStatus.NO_CONTENT
