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

import numpy as np
from flask import Response, abort, redirect, render_template, request, url_for
from flask.globals import current_app
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import (
    ProcessingTask,
    TaskLink,
    TaskUpdateSubscription,
)
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import HINGELOSS_BLP, HingeLoss
from .interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from .schemas import (
    EvaluateRequestSchema,
    EvaluateSchema,
    HyperparamterInputData,
    HyperparamterInputSchema,
    LossResponseSchema,
    PassDataSchema,
    WeightsResponseSchema,
)
from .tasks import clear_task_data, hinge_loss, load_data, load_data_from_db


@HINGELOSS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=HingeLoss.instance.name,
            description=HingeLoss.instance.description,
            name=HingeLoss.instance.identifier,
            version=HingeLoss.instance.version,
            type=PluginType.processing,
            tags=HingeLoss.instance.tags,
            links=[],
            entry_point=EntryPoint(
                href=url_for(f"{HINGELOSS_BLP.name}.{SetupProcess.__name__}"),
                ui_href=url_for(
                    f"{HINGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
                ),
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
        )


@HINGELOSS_BLP.route("/ui-hyperparameter/")
class HyperparameterSelectionMicroFrontend(MethodView):
    """Micro frontend for the hyperparameter selection."""

    example_inputs = {
        "c": 1.0,
    }

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @HINGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackUrl):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @HINGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackUrl):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackUrl):
        plugin = HingeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparamterInputSchema()
        callback_schema = CallbackUrlSchema()

        # set default values if not present
        if not data:
            data = {"c": 1.0}
        process_url = url_for(
            f"{HINGELOSS_BLP.name}.{SetupProcess.__name__}",
            # forward the callback url to the processing step
            **callback_schema.dump(callback),
        )
        example_values_url = url_for(
            f"{HINGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )

        return Response(
            render_template(
                "simple_template.html",
                name=HingeLoss.instance.name,
                version=HingeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="The Regularization Strength of the hinge loss.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@HINGELOSS_BLP.route("/hyperparameter/")
class SetupProcess(MethodView):
    """Save the hyperparameters to the database."""

    @HINGELOSS_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True
    )
    @HINGELOSS_BLP.response(HTTPStatus.SEE_OTHER)
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackUrl):
        """Start the invoked task."""
        # create new db_task
        db_task = ProcessingTask(
            task_name="hinge-loss",
        )
        db_task.data["c"] = arguments.c
        db_task.data["weights"] = -1

        # add callback as webhook subscriber subscribing to all updates
        subscription = TaskUpdateSubscription(
            db_task,
            webhook_href=callback.callback_url,
            task_href=url_for(
                "tasks-api.TaskView", task_id=str(db_task.id), _external=True
            ),
            event_type=None,
        )

        db_task.save()
        DB.session.flush()

        weights_link = TaskLink(
            db_task,
            type="of-weights",
            href=url_for(
                f"{HINGELOSS_BLP.name}.{WeightsEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        calc_loss_link = TaskLink(
            db_task,
            type="of-evaluate",
            href=url_for(
                f"{HINGELOSS_BLP.name}.{CalcLossEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        DB.session.add(weights_link)
        DB.session.add(calc_loss_link)

        subscription.save()

        db_task.add_next_step(
            href=url_for(
                f"{HINGELOSS_BLP.name}.{PassDataEndpoint.__name__}", db_id=db_task.id
            ),
            ui_href=url_for(
                f"{HINGELOSS_BLP.name}.{PassDataMicroFrontend.__name__}", db_id=db_task.id
            ),
            step_id="pass_data",
            commit=True,
        )

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_task.id)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@HINGELOSS_BLP.route("/task/<int:db_id>/ui-pass-data/")
class PassDataMicroFrontend(MethodView):
    """Micro frontend for the pass_data step."""

    example_inputs = {
        "c": 1.0,
    }

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pass_data step."
    )
    @HINGELOSS_BLP.arguments(
        PassDataSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id=db_id)

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pass_data step."
    )
    @HINGELOSS_BLP.arguments(
        PassDataSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id=db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = HingeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = PassDataSchema()

        process_url = url_for(
            f"{HINGELOSS_BLP.name}.{PassDataEndpoint.__name__}", db_id=db_id
        )
        example_values_url = url_for(
            f"{HINGELOSS_BLP.name}.{PassDataMicroFrontend.__name__}", db_id=db_id
        )

        return Response(
            render_template(
                "simple_template.html",
                name=HingeLoss.instance.name,
                version=HingeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="Pass data to the objective function.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@HINGELOSS_BLP.route("/task/<int:db_id>/pass-data/")
class PassDataEndpoint(MethodView):
    """Endpoint to load the features and target data."""

    @HINGELOSS_BLP.arguments(
        PassDataSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @HINGELOSS_BLP.response(HTTPStatus.SEE_OTHER)
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int):
        """Load features and target data."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)

        db_task.data["features_url"] = input_data["features"]
        db_task.data["target_url"] = input_data["target"]

        db_task.save(commit=True)

        task = load_data.s(db_id=db_id) | add_step.si(
            db_id=db_id,
            step_id="evaluate",
            href=url_for(
                f"{HINGELOSS_BLP.name}.{EvaluateEndpoint.__name__}", db_id=db_id
            ),
            ui_href=url_for(
                f"{HINGELOSS_BLP.name}.{EvaluateMicroFrontend.__name__}", db_id=db_id
            ),
            task_log="Finished loading data.",
        )
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@HINGELOSS_BLP.route("/task/<int:db_id>/ui-evaluate/")
class EvaluateMicroFrontend(MethodView):
    """Micro frontend for the evaluate step."""

    example_inputs = {
        "c": 1.0,
    }

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the evaluate step."
    )
    @HINGELOSS_BLP.arguments(
        EvaluateSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id=db_id)

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the evaluate step."
    )
    @HINGELOSS_BLP.arguments(
        EvaluateSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id=db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = HingeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = EvaluateSchema()

        process_url = url_for(
            f"{HINGELOSS_BLP.name}.{EvaluateEndpoint.__name__}", db_id=db_id
        )
        example_values_url = url_for(
            f"{HINGELOSS_BLP.name}.{EvaluateMicroFrontend.__name__}", db_id=db_id
        )

        return Response(
            render_template(
                "simple_template.html",
                name=HingeLoss.instance.name,
                version=HingeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="Complete the objective function task and clean up resources.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@HINGELOSS_BLP.route("/task/<int:db_id>/evaluate/")
class EvaluateEndpoint(MethodView):
    """Endpoint to complete the objective function task."""

    @HINGELOSS_BLP.arguments(
        EvaluateSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @HINGELOSS_BLP.response(HTTPStatus.SEE_OTHER)
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int):
        """Complete the objective function task."""
        task = clear_task_data.s(db_id=db_id) | save_task_result.s(db_id=db_id)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


#### Task Specific Endpoints ###################################################


@HINGELOSS_BLP.route("/task/<int:db_id>/weights/")
class WeightsEndpoint(MethodView):
    """Endpoint for the number of weights."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, WeightsResponseSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, db_id: int) -> dict:

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        return {"weights": db_task.data.get("weights", -1)}


@HINGELOSS_BLP.route("/task/<int:db_id>/loss/")
class CalcLossEndpoint(MethodView):
    """Endpoint for the loss calculation."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @HINGELOSS_BLP.arguments(
        EvaluateRequestSchema(unknown=EXCLUDE), location="json", required=True
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int) -> dict:
        """Calculate the loss given the specific weights."""

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)

        weights = np.array(input_data["weights"])
        features = load_data_from_db(db_task.data["features_key"])
        target = load_data_from_db(db_task.data["target_key"])

        loss = hinge_loss(
            X=features,
            y=target,
            w=weights,
            C=db_task.data["c"],
        )
        return {"loss": loss}
