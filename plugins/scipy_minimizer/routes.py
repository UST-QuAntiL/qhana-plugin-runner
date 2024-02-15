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

from flask import Response, redirect, render_template, request, url_for
from flask.globals import current_app
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskUpdateSubscription
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    save_task_error,
    save_task_result,
)

from . import SCIPY_MINIMIZER_BLP, ScipyMinimizer
from .interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from .schemas import (
    MinimizerEnum,
    MinimizerSetupTaskInputData,
    MinimizerSetupTaskInputSchema,
    MinimizerTaskResponseSchema,
    MinimizeSchema,
)
from .tasks import minimize_task


@SCIPY_MINIMIZER_BLP.route("/")
class MetadataView(MethodView):
    """Plugins collection resource."""

    @SCIPY_MINIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Optimizer endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Scipy Minimizer Plugin",
            description=ScipyMinimizer.instance.description,
            name=ScipyMinimizer.instance.name,
            version=ScipyMinimizer.instance.version,
            type=PluginType.processing,
            tags=ScipyMinimizer.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{SCIPY_MINIMIZER_BLP.name}.{MinimizerSetupProcessStep.__name__}"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{SCIPY_MINIMIZER_BLP.name}.{MinimizerSetupMicroFrontend.__name__}"
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


@SCIPY_MINIMIZER_BLP.route("/ui-setup/")
class MinimizerSetupMicroFrontend(MethodView):
    """Micro frontend for the minimization method selection."""

    example_inputs = {
        "method": MinimizerEnum.nelder_mead,
    }

    @SCIPY_MINIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the minimization method selection.",
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackUrl):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @SCIPY_MINIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the minimization method selection.",
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackUrl):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackUrl):
        schema = MinimizerSetupTaskInputSchema()
        callback_schema = CallbackUrlSchema()

        if not data:
            data = self.example_inputs

        process_url = url_for(
            f"{SCIPY_MINIMIZER_BLP.name}.{MinimizerSetupProcessStep.__name__}",
            **callback_schema.dump(callback),
        )

        example_url = url_for(
            f"{SCIPY_MINIMIZER_BLP.name}.{MinimizerSetupMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )
        return Response(
            render_template(
                "simple_template.html",
                name=ScipyMinimizer.instance.name,
                version=ScipyMinimizer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,
                example_values=example_url,
            ),
        )


@SCIPY_MINIMIZER_BLP.route("/process-setup/")
class MinimizerSetupProcessStep(MethodView):
    """Callback to the coordinator."""

    @SCIPY_MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(unknown=EXCLUDE), location="form"
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True
    )
    @SCIPY_MINIMIZER_BLP.response(HTTPStatus.OK, MinimizerTaskResponseSchema())
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: MinimizerSetupTaskInputData, callback: CallbackUrl):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="minimizer_task",
        )
        db_task.data["method"] = arguments.method.value

        db_task.save()
        DB.session.flush()

        # add callback as webhook subscriber subscribing to all updates
        subscription = TaskUpdateSubscription(
            db_task,
            webhook_href=callback.callback,
            task_href=url_for(
                "tasks-api.TaskView", task_id=str(db_task.id), _external=True
            ),
            event_type=None,
        )

        subscription.save()

        db_task.add_next_step(
            href=url_for(
                f"{SCIPY_MINIMIZER_BLP.name}.{MinimizationEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
            ui_href=url_for(
                f"{SCIPY_MINIMIZER_BLP.name}.{MinimizationMicroFrontend.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
            step_id="minimize",
        )

        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_task.id)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@SCIPY_MINIMIZER_BLP.route("/task/<int:db_id>/ui-minimize/")
class MinimizationMicroFrontend(MethodView):
    """Micro frontend for the minimize step."""

    @SCIPY_MINIMIZER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the minimize step."
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        MinimizeSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id=db_id)

    @SCIPY_MINIMIZER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the minimize step."
    )
    @SCIPY_MINIMIZER_BLP.arguments(
        MinimizeSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @SCIPY_MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id=db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = ScipyMinimizer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = MinimizeSchema()

        process_url = url_for(
            f"{SCIPY_MINIMIZER_BLP.name}.{MinimizationEndpoint.__name__}", db_id=db_id
        )
        example_values_url = url_for(
            f"{SCIPY_MINIMIZER_BLP.name}.{MinimizationMicroFrontend.__name__}",
            db_id=db_id,
        )

        return Response(
            render_template(
                "simple_template.html",
                name=ScipyMinimizer.instance.name,
                version=ScipyMinimizer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="Pass data to the objective function.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@SCIPY_MINIMIZER_BLP.route("/task/<int:db_id>/minimize/")
class MinimizationEndpoint(MethodView):
    """Endpoint for the minimization."""

    @SCIPY_MINIMIZER_BLP.arguments(MinimizeSchema(unknown=EXCLUDE), location="form")
    def post(self, input_data: dict, db_id: int):
        """Minimize the objective function."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)

        db_task.data["objective_function_task"] = input_data["objective_function"]
        initial_weights_url = input_data.get("initial_weights")
        if initial_weights_url:
            db_task.data["initial_weights_url"] = initial_weights_url

        db_task.clear_previous_step()

        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_task.id)

        task = minimize_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
