# # Copyright 2023 QHAna plugin runner contributors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


from http import HTTPStatus
from logging import Logger
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from flask import Response, redirect, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.optimizer.interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from plugins.optimizer.interaction_utils.tasks import make_callback
from plugins.optimizer.minimizer import MINIMIZER_BLP, Minimizer
from plugins.optimizer.minimizer.schemas import (
    MinimizerEnum,
    MinimizerSetupTaskInputData,
    MinimizerSetupTaskInputSchema,
    MinimizerTaskResponseSchema,
)
from plugins.optimizer.shared.schemas import (
    MinimizerCallbackData,
    MinimizerCallbackSchema,
    MinimizerInputData,
    MinimizerInputSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from .tasks import minimize_task


@MINIMIZER_BLP.route("/")
class MetadataView(MethodView):
    """Plugins collection resource."""

    @MINIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Optimizer endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Minimizer plugin",
            description=Minimizer.instance.description,
            name=Minimizer.instance.name,
            version=Minimizer.instance.version,
            type=PluginType.processing,
            tags=Minimizer.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{MINIMIZER_BLP.name}.{MinimizerSetupProcessStep.__name__}"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{MINIMIZER_BLP.name}.{MinimizerSetupMicroFrontend.__name__}"
                ),  # URL for the first micro frontend endpoint
                data_input=[],
                data_output=[
                    DataMetadata(
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


@MINIMIZER_BLP.route("/ui-setup/")
class MinimizerSetupMicroFrontend(MethodView):
    """Micro frontend for the minimization method selection."""

    example_inputs = {
        "method": MinimizerEnum.nelder_mead,
    }

    @MINIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the minimization method selection.",
    )
    @MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackUrl):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @MINIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the minimization method selection.",
    )
    @MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackUrl):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackUrl):
        schema = MinimizerSetupTaskInputSchema()
        callback_schema = CallbackUrlSchema()

        if not data:
            data = self.example_inputs

        process_url = url_for(
            f"{MINIMIZER_BLP.name}.{MinimizerSetupProcessStep.__name__}",
            **callback_schema.dump(callback),
        )

        example_url = url_for(
            f"{MINIMIZER_BLP.name}.{MinimizerSetupMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )
        return Response(
            render_template(
                "simple_template.html",
                name=Minimizer.instance.name,
                version=Minimizer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,
                example_values=example_url,
            ),
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@MINIMIZER_BLP.route("/process-setup/")
class MinimizerSetupProcessStep(MethodView):
    """Callback to the coordinator."""

    @MINIMIZER_BLP.arguments(
        MinimizerSetupTaskInputSchema(unknown=EXCLUDE), location="form"
    )
    @MINIMIZER_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True
    )
    @MINIMIZER_BLP.response(HTTPStatus.OK, MinimizerTaskResponseSchema())
    @MINIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: MinimizerSetupTaskInputData, callback: CallbackUrl):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="minimizer_task",
        )
        db_task.data["method"] = arguments.method.value
        db_task.save(commit=True)

        minimize_endpoint = url_for(
            f"{MINIMIZER_BLP.name}.{MinimizationEndpoint.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        callback_schema = MinimizerCallbackSchema()
        callback_data = callback_schema.dump(
            MinimizerCallbackData(
                minimize_endpoint_url=minimize_endpoint,
            )
        )
        make_callback(callback.callback_url, callback_data)


@MINIMIZER_BLP.route("<int:db_id>/minimize/")
class MinimizationEndpoint(MethodView):
    """Endpoint for the minimization."""

    @MINIMIZER_BLP.arguments(MinimizerInputSchema(unknown=EXCLUDE), location="json")
    def post(self, input_data: MinimizerInputData, db_id: int):
        """Minimize the objective function."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)
        if input_data.callback_url:
            db_task.data["status_changed_callback_urls"] = [input_data.callback_url]
        db_task.data["calc_loss_endpoint_url"] = input_data.calc_loss_endpoint_url
        if input_data.calc_gradient_endpoint_url:
            db_task.data[
                "calc_gradient_endpoint_url"
            ] = input_data.calc_gradient_endpoint_url
        schema = MinimizerInputSchema()
        serialized_input_data = schema.dump(input_data)
        db_task.data["x0"] = serialized_input_data["x0"]
        db_task.data["task_view"] = url_for(
            "tasks-api.TaskView", task_id=db_task.id, _external=True
        )

        db_task.save(commit=True)
        task = minimize_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
